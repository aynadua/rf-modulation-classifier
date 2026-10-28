"""ONNX export, int8 quantisation and latency measurement.

A modulation classifier is only useful if it keeps up with the radio, so the
deployable artefact and its latency are part of the result, not an
afterthought. Quantisation here is *static* (post-training, with a calibration
pass over real training examples) rather than dynamic: this is a convolutional
network, and dynamic quantisation leaves the convolutions in float.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from rfmc.models import IQCNN

INPUT_NAME = "iq"
OUTPUT_NAME = "logits"


def export_onnx(model: IQCNN, n_samples: int, path: str | Path) -> Path:
    """Write an fp32 ONNX graph with a dynamic batch dimension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 2, n_samples)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


class _ArrayCalibrationReader(CalibrationDataReader):
    """Feeds calibration batches to the quantiser, one example at a time."""

    def __init__(self, x: np.ndarray) -> None:
        self._batches = iter([{INPUT_NAME: example[None].astype(np.float32)} for example in x])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)


def quantise_int8(
    fp32_path: str | Path, calibration_x: np.ndarray, path: str | Path
) -> Path:
    """Static post-training quantisation to int8, calibrated on real examples."""
    fp32_path, path = Path(fp32_path), Path(path)
    prepared = path.with_name(path.stem + ".prepared.onnx")
    quant_pre_process(str(fp32_path), str(prepared))
    quantize_static(
        str(prepared),
        str(path),
        _ArrayCalibrationReader(calibration_x),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    prepared.unlink(missing_ok=True)
    return path


def session(path: str | Path) -> ort.InferenceSession:
    """A single-threaded session, so latency numbers mean one core."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def predict_onnx(sess: ort.InferenceSession, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Class predictions from an ONNX session."""
    out = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].astype(np.float32)
        out.append(sess.run([OUTPUT_NAME], {INPUT_NAME: batch})[0])
    return np.concatenate(out).argmax(axis=1)


def benchmark(
    sess: ort.InferenceSession, n_samples: int, n_runs: int = 300, warmup: int = 30
) -> dict[str, float]:
    """Single-example, single-thread latency in milliseconds."""
    rng = np.random.default_rng(0)
    example = rng.standard_normal((1, 2, n_samples)).astype(np.float32)

    for _ in range(warmup):
        sess.run([OUTPUT_NAME], {INPUT_NAME: example})

    timings = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        sess.run([OUTPUT_NAME], {INPUT_NAME: example})
        timings[i] = (time.perf_counter() - t0) * 1e3

    return {
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_p50": float(np.percentile(timings, 50)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
    }


def artefact_report(path: str | Path, n_samples: int, **extra: Any) -> dict[str, Any]:
    """Size and latency of one exported artefact."""
    path = Path(path)
    return {
        "artefact": path.name,
        "size_kb": round(path.stat().st_size / 1024, 1),
        **benchmark(session(path), n_samples),
        **extra,
    }
