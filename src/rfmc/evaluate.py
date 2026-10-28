"""Evaluation shared by both models.

A single top-line accuracy is close to meaningless for modulation
classification, because it is entirely determined by which SNRs you chose to
average over. Everything here is reported *per SNR*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rfmc.modulators import SCHEMES


def accuracy_by_snr(
    y_true: np.ndarray, y_pred: np.ndarray, snr_db: np.ndarray
) -> dict[float, float]:
    """Classification accuracy at each distinct SNR present in the set."""
    out: dict[float, float] = {}
    for snr in np.unique(snr_db):
        mask = snr_db == snr
        out[float(snr)] = float(np.mean(y_pred[mask] == y_true[mask]))
    return out


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Row-normalised confusion matrix (rows are true classes)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1.0)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)


def summarise(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, snr_db: np.ndarray, **extra: Any
) -> dict[str, Any]:
    """Build the metrics record written to ``reports/``."""
    by_snr = accuracy_by_snr(y_true, y_pred, snr_db)
    high = {s: a for s, a in by_snr.items() if s >= 10}
    return {
        "model": name,
        "overall_accuracy": float(np.mean(y_pred == y_true)),
        "accuracy_high_snr_ge10db": float(np.mean(list(high.values()))) if high else None,
        "accuracy_by_snr": by_snr,
        "classes": list(SCHEMES),
        "confusion_matrix_high_snr": confusion_matrix(
            y_true[snr_db >= 10], y_pred[snr_db >= 10], len(SCHEMES)
        ).tolist(),
        **extra,
    }


def write_report(record: dict[str, Any], path: str | Path) -> Path:
    """Persist one metrics record as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path


def format_snr_table(records: list[dict[str, Any]]) -> str:
    """Markdown table of accuracy vs SNR, one column per model."""
    snrs = sorted({float(s) for r in records for s in r["accuracy_by_snr"]})
    header = "| SNR (dB) | " + " | ".join(r["model"] for r in records) + " |"
    rule = "|---:|" + "---:|" * len(records)
    lines = [header, rule]
    for snr in snrs:
        cells = []
        for record in records:
            by_snr = record["accuracy_by_snr"]
            # JSON round-trips dict keys to strings, so accept either form.
            value = by_snr.get(str(snr), by_snr.get(snr, float("nan")))
            cells.append(f"{value:.3f}")
        lines.append(f"| {snr:+.0f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
