"""Metric helpers."""

import numpy as np
import pytest

from rfmc.evaluate import accuracy_by_snr, confusion_matrix, format_snr_table, summarise
from rfmc.modulators import SCHEMES


def test_accuracy_is_computed_per_snr():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    snr = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    assert accuracy_by_snr(y_true, y_pred, snr) == {0.0: pytest.approx(2 / 3), 10.0: 1.0}


def test_confusion_matrix_rows_are_normalised():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    matrix = confusion_matrix(y_true, y_pred, 2)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0, 0] == pytest.approx(0.5)
    assert matrix[1, 1] == pytest.approx(1.0)


def test_confusion_matrix_tolerates_an_absent_class():
    matrix = confusion_matrix(np.array([0]), np.array([0]), 3)
    assert np.allclose(matrix[1], 0.0)


def test_summary_carries_class_names_and_high_snr_slice():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, len(SCHEMES), n)
    snr = rng.choice([-10.0, 0.0, 10.0, 20.0], n)
    record = summarise("m", y_true, y_true.copy(), snr)
    assert record["classes"] == list(SCHEMES)
    assert record["overall_accuracy"] == 1.0
    assert record["accuracy_high_snr_ge10db"] == 1.0


def test_snr_table_renders_one_column_per_model():
    record = summarise(
        "m", np.array([0, 1]), np.array([0, 1]), np.array([0.0, 10.0])
    )
    table = format_snr_table([record])
    assert "| SNR (dB) | m |" in table
    assert table.count("\n") == 3
