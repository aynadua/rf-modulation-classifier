"""Command line entry point: ``rfmc <stage>``.

The stages run in order and each writes its own artefact, so a failed run can
be resumed rather than restarted:

    rfmc data       # generate train/val/test IQ datasets
    rfmc baseline   # fit and score the cumulant + gradient-boosting model
    rfmc train      # fit the CNN
    rfmc evaluate   # score the CNN on the held-out test split
    rfmc export     # ONNX export, int8 quantisation, latency benchmark
    rfmc report     # collect everything into reports/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rfmc import baseline, evaluate, plots
from rfmc.config import Config
from rfmc.dataset import IQDataset, build_all
from rfmc.modulators import SCHEMES

REPORTS = Path("reports")
MODELS = Path("models")


def _data_path(cfg: Config, split: str) -> Path:
    return Path(cfg.data.out_dir) / f"{split}.npz"


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {path}; run `{hint}` first")


def cmd_data(cfg: Config, args: argparse.Namespace) -> None:
    paths = build_all(cfg)
    for split, path in paths.items():
        data = IQDataset.load(path)
        size_mb = path.stat().st_size / 1e6
        print(f"{split:6s} {len(data):7d} examples  {data.x.shape}  {size_mb:7.1f} MB  {path}")


def cmd_baseline(cfg: Config, args: argparse.Namespace) -> None:
    for split in ("train", "test"):
        _require(_data_path(cfg, split), "rfmc data")

    train_data = IQDataset.load(_data_path(cfg, "train"))
    test_data = IQDataset.load(_data_path(cfg, "test"))

    model, info = baseline.fit(train_data, cfg.signal.sps, seed=cfg.train.seed)
    predictions = baseline.predict(model, test_data, cfg.signal.sps)

    record = evaluate.summarise(
        "cumulant + GBM", test_data.y, predictions, test_data.snr_db, **info
    )
    baseline.save(model, MODELS / "baseline.joblib")
    evaluate.write_report(record, REPORTS / "baseline.json")
    print(
        f"baseline: overall {record['overall_accuracy']:.4f}  "
        f"SNR>=10dB {record['accuracy_high_snr_ge10db']:.4f}"
    )


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    from rfmc import train as train_mod

    for split in ("train", "val"):
        _require(_data_path(cfg, split), "rfmc data")

    result = train_mod.train(
        IQDataset.load(_data_path(cfg, "train")),
        IQDataset.load(_data_path(cfg, "val")),
        cfg,
    )
    path = train_mod.save_checkpoint(result, cfg, MODELS / "cnn.pt")
    print(
        f"best val accuracy {result.best_val_accuracy:.4f} after {result.epochs_run} epochs "
        f"({result.train_seconds:.0f}s) -> {path}"
    )


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> None:
    from rfmc import train as train_mod
    from rfmc.models import count_parameters

    _require(MODELS / "cnn.pt", "rfmc train")
    _require(_data_path(cfg, "test"), "rfmc data")

    model, blob = train_mod.load_checkpoint(MODELS / "cnn.pt")
    test_data = IQDataset.load(_data_path(cfg, "test"))
    predictions = train_mod.predict(model, test_data)

    record = evaluate.summarise(
        "CNN (fp32)",
        test_data.y,
        predictions,
        test_data.snr_db,
        n_parameters=count_parameters(model),
        epochs_run=len(blob["history"]),
        best_val_accuracy=blob["best_val_accuracy"],
    )
    evaluate.write_report(record, REPORTS / "cnn.json")
    print(
        f"CNN: overall {record['overall_accuracy']:.4f}  "
        f"SNR>=10dB {record['accuracy_high_snr_ge10db']:.4f}"
    )


def cmd_export(cfg: Config, args: argparse.Namespace) -> None:
    from rfmc import export as export_mod
    from rfmc import train as train_mod

    _require(MODELS / "cnn.pt", "rfmc train")
    _require(_data_path(cfg, "test"), "rfmc data")

    model, _ = train_mod.load_checkpoint(MODELS / "cnn.pt")
    test_data = IQDataset.load(_data_path(cfg, "test"))
    train_data = IQDataset.load(_data_path(cfg, "train"))
    n_samples = cfg.signal.n_samples

    fp32 = export_mod.export_onnx(model, n_samples, MODELS / "cnn.onnx")
    calibration = train_data.x[: args.calibration_size]
    int8 = export_mod.quantise_int8(fp32, calibration, MODELS / "cnn.int8.onnx")

    records: list[dict[str, Any]] = []
    for name, path in (("CNN (fp32 ONNX)", fp32), ("CNN (int8 ONNX)", int8)):
        predictions = export_mod.predict_onnx(export_mod.session(path), test_data.x)
        summary = evaluate.summarise(name, test_data.y, predictions, test_data.snr_db)
        summary.update(export_mod.artefact_report(path, n_samples))
        records.append(summary)
        print(
            f"{name:18s} {summary['size_kb']:8.1f} KB  "
            f"p50 {summary['latency_ms_p50']:.2f} ms  "
            f"acc {summary['overall_accuracy']:.4f}"
        )

    evaluate.write_report({"artefacts": records}, REPORTS / "export.json")


def cmd_report(cfg: Config, args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    for name in ("baseline.json", "cnn.json"):
        path = REPORTS / name
        if path.exists():
            records.append(json.loads(path.read_text()))

    export_path = REPORTS / "export.json"
    export_records = (
        json.loads(export_path.read_text())["artefacts"] if export_path.exists() else []
    )
    if not records:
        raise SystemExit("nothing to report; run `rfmc baseline` and `rfmc train` first")

    plots.accuracy_vs_snr(records + export_records, REPORTS / "accuracy_vs_snr.png")
    for record in records:
        slug = record["model"].split()[0].lower().strip("(")
        plots.confusion(record, REPORTS / f"confusion_{slug}.png")
    plots.constellations(REPORTS / "constellations.png")

    lines = [
        "# Results",
        "",
        "Generated by `rfmc report`. All numbers are on the held-out test split.",
        "",
        "## Headline",
        "",
        "| Model | Overall | SNR >= 10 dB | Size | p50 latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records + export_records:
        size = f"{record['size_kb']:.0f} KB" if "size_kb" in record else "-"
        latency = f"{record['latency_ms_p50']:.2f} ms" if "latency_ms_p50" in record else "-"
        lines.append(
            f"| {record['model']} | {record['overall_accuracy']:.4f} | "
            f"{record['accuracy_high_snr_ge10db']:.4f} | {size} | {latency} |"
        )

    lines += ["", "## Accuracy vs SNR", "", evaluate.format_snr_table(records + export_records), ""]
    lines += ["![accuracy vs snr](accuracy_vs_snr.png)", ""]
    path = REPORTS / "RESULTS.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def cmd_demo(cfg: Config, args: argparse.Namespace) -> None:
    """Classify one freshly generated burst, end to end."""
    from rfmc import train as train_mod
    from rfmc.channel import ChannelParams, apply_channel
    from rfmc.modulators import modulate

    _require(MODELS / "cnn.pt", "rfmc train")
    model, _ = train_mod.load_checkpoint(MODELS / "cnn.pt")

    rng = np.random.default_rng(args.seed)
    scheme = args.scheme or str(rng.choice(SCHEMES))
    sig = cfg.signal
    waveform = modulate(scheme, sig.n_samples // sig.sps + 4, sig.sps, rng)
    received = apply_channel(
        waveform,
        ChannelParams(snr_db=args.snr, n_taps=2, phase_off=float(rng.uniform(0, 6.28))),
        rng,
        n_out=sig.n_samples,
    )
    received = received / np.sqrt(np.mean(np.abs(received) ** 2))

    iq = np.stack([received.real, received.imag]).astype(np.float32)[None]
    data = IQDataset(x=iq, y=np.array([0]), snr_db=np.array([args.snr], dtype=np.float32))
    logits = train_mod.evaluate_logits(model, data)[0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()

    print(f"transmitted: {scheme} at {args.snr:.0f} dB SNR")
    for index in np.argsort(-probabilities)[:3]:
        print(f"  {SCHEMES[index]:6s} {probabilities[index]:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rfmc", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="path to a YAML config")
    sub = parser.add_subparsers(dest="stage", required=True)

    for name, handler in (
        ("data", cmd_data),
        ("baseline", cmd_baseline),
        ("train", cmd_train),
        ("evaluate", cmd_evaluate),
        ("report", cmd_report),
    ):
        sub.add_parser(name, help=handler.__doc__ or name).set_defaults(handler=handler)

    export_parser = sub.add_parser("export", help="ONNX export and int8 quantisation")
    export_parser.add_argument("--calibration-size", type=int, default=400)
    export_parser.set_defaults(handler=cmd_export)

    demo_parser = sub.add_parser("demo", help="classify one freshly generated burst")
    demo_parser.add_argument("--scheme", choices=SCHEMES, default=None)
    demo_parser.add_argument("--snr", type=float, default=10.0)
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.set_defaults(handler=cmd_demo)

    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    args.handler(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
