"""Figures for the README. Headless by construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rfmc.modulators import SCHEMES  # noqa: E402


def _snr_series(record: dict[str, Any]) -> tuple[list[float], list[float]]:
    items = sorted((float(k), v) for k, v in record["accuracy_by_snr"].items())
    return [k for k, _ in items], [v for _, v in items]


def accuracy_vs_snr(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Overlay every model's accuracy-vs-SNR curve."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for record in records:
        snrs, accuracies = _snr_series(record)
        ax.plot(snrs, accuracies, marker="o", markersize=4, label=record["model"])

    ax.axhline(1.0 / len(SCHEMES), linestyle="--", linewidth=1, color="grey", label="chance")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Modulation classification accuracy vs SNR ({len(SCHEMES)} classes)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def confusion(record: dict[str, Any], path: str | Path) -> Path:
    """Row-normalised confusion matrix at high SNR."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = record["confusion_matrix_high_snr"]
    labels = record["classes"]

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=140)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{record['model']}: confusion at SNR >= 10 dB")

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value >= 0.01:
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.5 else "black", fontsize=8,
                )

    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def constellations(path: str | Path, snr_db: float = 20.0) -> Path:
    """What the classifier actually sees, after symbol timing recovery."""
    import numpy as np

    from rfmc.channel import ChannelParams, apply_channel
    from rfmc.features import symbol_sync
    from rfmc.modulators import modulate

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=140)
    for ax, scheme in zip(axes.ravel(), SCHEMES, strict=True):
        waveform = modulate(scheme, 2000, 8, rng)
        received = apply_channel(
            waveform, ChannelParams(snr_db=snr_db, n_taps=1, phase_off=0.0), rng
        )
        symbols = symbol_sync(received, 8)
        symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))
        ax.scatter(symbols.real, symbols.imag, s=1.5, alpha=0.3)
        ax.set_title(scheme, fontsize=10)
        # Shared limits so the panels are directly comparable: a one-dimensional
        # constellation should look flat next to a square one, not rescaled.
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Recovered constellations at {snr_db:.0f} dB SNR (AWGN only)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
