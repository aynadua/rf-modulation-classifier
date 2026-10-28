"""Dataset generation.

Two properties of this generator matter more than anything else in the repo:

1. **Every example is an independent realisation.** New random symbols, new
   channel taps, new impairments. Nothing is produced by sliding a window over
   a long recording, so no fragment of a train example can reappear in test.
   Sliding-window splits are the most common way published modulation-
   classification accuracies get inflated.
2. **Splits are generated from disjoint seeds**, so regenerating one split
   cannot silently perturb another.

Each example is normalised to unit average power. Without that, absolute
amplitude would leak the SNR straight into the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rfmc.channel import ChannelParams, apply_channel
from rfmc.config import Config
from rfmc.modulators import SCHEMES, modulate


@dataclass
class IQDataset:
    """IQ examples as ``(N, 2, L)`` float32, with labels and per-example SNR."""

    x: np.ndarray
    y: np.ndarray
    snr_db: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=self.x, y=self.y, snr_db=self.snr_db)

    @classmethod
    def load(cls, path: str | Path) -> IQDataset:
        with np.load(path) as f:
            return cls(x=f["x"], y=f["y"], snr_db=f["snr_db"])


def snr_grid(cfg: Config) -> np.ndarray:
    """The SNR points the dataset is balanced over."""
    imp = cfg.impairments
    n = int(round((imp.snr_db_max - imp.snr_db_min) / imp.snr_db_step)) + 1
    return imp.snr_db_min + imp.snr_db_step * np.arange(n)


def _draw_params(snr_db: float, cfg: Config, rng: np.random.Generator) -> ChannelParams:
    imp = cfg.impairments
    return ChannelParams(
        snr_db=snr_db,
        n_taps=int(rng.integers(1, imp.max_taps + 1)),
        rician_k_db=float(rng.uniform(imp.rician_k_db_min, imp.rician_k_db_max)),
        delay_spread=imp.delay_spread,
        cfo_norm=float(rng.uniform(-imp.cfo_norm_max, imp.cfo_norm_max)),
        phase_off=float(rng.uniform(0.0, 2.0 * np.pi)),
        timing_off=float(rng.uniform(0.0, imp.timing_off_max)),
        sro_ppm=float(rng.uniform(-imp.sro_ppm_max, imp.sro_ppm_max)),
    )


def generate_split(cfg: Config, split: str) -> IQDataset:
    """Generate one split, balanced across every (scheme, SNR) cell."""
    if split not in cfg.data.per_class_per_snr:
        raise KeyError(f"no example count configured for split {split!r}")

    per_cell = cfg.data.per_class_per_snr[split]
    snrs = snr_grid(cfg)
    rng = np.random.default_rng(cfg.data.seeds[split])

    sig = cfg.signal
    n_symbols = int(np.ceil(sig.n_samples / sig.sps))
    # Generate a margin of extra samples so the clock-drift resampler never
    # runs off the end of the waveform.
    gen_symbols = n_symbols + 4

    total = len(SCHEMES) * len(snrs) * per_cell
    x = np.empty((total, 2, sig.n_samples), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    snr_out = np.empty(total, dtype=np.float32)

    i = 0
    for label, scheme in enumerate(SCHEMES):
        for snr in snrs:
            for _ in range(per_cell):
                clean = modulate(
                    scheme, gen_symbols, sig.sps, rng, beta=sig.rrc_beta, span=sig.rrc_span
                )
                params = _draw_params(float(snr), cfg, rng)
                rx = apply_channel(clean, params, rng, n_out=sig.n_samples)

                rx = rx / np.sqrt(np.mean(np.abs(rx) ** 2))
                x[i, 0] = rx.real
                x[i, 1] = rx.imag
                y[i] = label
                snr_out[i] = snr
                i += 1

    assert i == total
    perm = rng.permutation(total)
    return IQDataset(x=x[perm], y=y[perm], snr_db=snr_out[perm])


def build_all(cfg: Config, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Generate and persist every configured split. Returns the written paths."""
    out = Path(out_dir or cfg.data.out_dir)
    paths: dict[str, Path] = {}
    for split in cfg.data.per_class_per_snr:
        ds = generate_split(cfg, split)
        path = out / f"{split}.npz"
        ds.save(path)
        paths[split] = path
    return paths
