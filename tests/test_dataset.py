"""Dataset tests, focused on the properties that keep the benchmark honest."""

import numpy as np
import pytest

from rfmc.config import Config
from rfmc.dataset import IQDataset, generate_split, snr_grid
from rfmc.modulators import SCHEMES


@pytest.fixture
def tiny_config():
    cfg = Config()
    cfg.impairments.snr_db_min = 0.0
    cfg.impairments.snr_db_max = 10.0
    cfg.impairments.snr_db_step = 5.0
    cfg.data.per_class_per_snr = {"train": 4, "val": 2}
    cfg.data.seeds = {"train": 1, "val": 2}
    return cfg


def test_snr_grid_endpoints_are_inclusive(tiny_config):
    assert np.array_equal(snr_grid(tiny_config), np.array([0.0, 5.0, 10.0]))


def test_split_is_balanced_over_classes_and_snr(tiny_config):
    data = generate_split(tiny_config, "train")
    assert len(data) == len(SCHEMES) * 3 * 4
    assert np.array_equal(np.bincount(data.y), np.full(len(SCHEMES), 12))
    for snr in (0.0, 5.0, 10.0):
        assert (data.snr_db == snr).sum() == len(SCHEMES) * 4


def test_every_example_has_unit_average_power(tiny_config):
    data = generate_split(tiny_config, "train")
    power = np.mean(data.x.astype(np.float64) ** 2, axis=2).sum(axis=1)
    assert np.allclose(power, 1.0, atol=1e-5)


def test_shapes_and_dtypes(tiny_config):
    data = generate_split(tiny_config, "train")
    assert data.x.shape == (len(data), 2, tiny_config.signal.n_samples)
    assert data.x.dtype == np.float32
    assert data.y.dtype == np.int64


def test_generation_is_reproducible(tiny_config):
    assert np.array_equal(
        generate_split(tiny_config, "train").x, generate_split(tiny_config, "train").x
    )


def test_splits_do_not_share_examples(tiny_config):
    """Different seeds must give genuinely different channel realisations."""
    train = generate_split(tiny_config, "train")
    val = generate_split(tiny_config, "val")
    train_rows = {row.tobytes() for row in train.x}
    assert not any(row.tobytes() in train_rows for row in val.x)


def test_unknown_split_is_rejected(tiny_config):
    with pytest.raises(KeyError):
        generate_split(tiny_config, "holdout")


def test_round_trip_through_disk(tiny_config, tmp_path):
    data = generate_split(tiny_config, "train")
    data.save(tmp_path / "x.npz")
    loaded = IQDataset.load(tmp_path / "x.npz")
    assert np.array_equal(loaded.x, data.x)
    assert np.array_equal(loaded.y, data.y)
    assert np.array_equal(loaded.snr_db, data.snr_db)
