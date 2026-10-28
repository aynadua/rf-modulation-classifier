"""Typed configuration, loaded from YAML so experiments are reproducible."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SignalConfig:
    """Waveform geometry."""

    n_samples: int = 512
    sps: int = 8
    rrc_beta: float = 0.35
    rrc_span: int = 10


@dataclass
class ImpairmentConfig:
    """Ranges the per-example channel parameters are drawn from."""

    snr_db_min: float = -10.0
    snr_db_max: float = 20.0
    snr_db_step: float = 2.0
    max_taps: int = 4
    rician_k_db_min: float = 4.0
    rician_k_db_max: float = 18.0
    delay_spread: float = 2.0
    cfo_norm_max: float = 1e-3
    timing_off_max: float = 1.0
    sro_ppm_max: float = 50.0


@dataclass
class DataConfig:
    """How many realisations to draw per class and SNR, and with which seeds."""

    per_class_per_snr: dict[str, int] = field(
        default_factory=lambda: {"train": 600, "val": 150, "test": 250}
    )
    seeds: dict[str, int] = field(default_factory=lambda: {"train": 11, "val": 22, "test": 33})
    out_dir: str = "data"


@dataclass
class TrainConfig:
    """Optimiser and schedule."""

    epochs: int = 30
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 6
    seed: int = 7
    num_workers: int = 0


@dataclass
class Config:
    signal: SignalConfig = field(default_factory=SignalConfig)
    impairments: ImpairmentConfig = field(default_factory=ImpairmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        """Load a config from YAML, falling back to defaults for absent keys."""
        if path is None:
            return cls()
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")

        sections: dict[str, Any] = {}
        for f in fields(cls):
            section = f.default_factory()  # type: ignore[misc]
            for key, value in (raw.get(f.name) or {}).items():
                if not hasattr(section, key):
                    raise ValueError(f"unknown config key {f.name}.{key}")
                setattr(section, key, value)
            sections[f.name] = section
        return cls(**sections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
