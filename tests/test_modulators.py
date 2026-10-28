"""Waveform generation invariants."""

import numpy as np
import pytest

from rfmc.modulators import SCHEMES, constellation, modulate, rrc_filter


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constellation_has_unit_average_energy(scheme):
    points = constellation(scheme)
    assert np.isclose(np.mean(np.abs(points) ** 2), 1.0)


@pytest.mark.parametrize(
    ("scheme", "size"),
    [("BPSK", 2), ("QPSK", 4), ("8PSK", 8), ("PAM4", 4), ("16QAM", 16), ("64QAM", 64)],
)
def test_constellation_cardinality(scheme, size):
    assert len(np.unique(np.round(constellation(scheme), 9))) == size


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown scheme"):
        constellation("QAM256")


@pytest.mark.parametrize("beta", [0.0, 0.22, 0.35, 1.0])
def test_rrc_filter_is_unit_energy_and_symmetric(beta):
    taps = rrc_filter(sps=8, span=10, beta=beta)
    assert np.isclose(np.sum(taps**2), 1.0)
    assert np.allclose(taps, taps[::-1])
    assert np.all(np.isfinite(taps))


def test_rrc_rejects_out_of_range_rolloff():
    with pytest.raises(ValueError, match="roll-off"):
        rrc_filter(sps=8, span=10, beta=1.5)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_modulate_shape_and_power(scheme):
    waveform = modulate(scheme, n_symbols=64, sps=8, rng=np.random.default_rng(0))
    assert waveform.shape == (512,)
    assert np.isclose(np.mean(np.abs(waveform) ** 2), 1.0)


def test_modulate_is_deterministic_given_a_seed():
    a = modulate("QPSK", 32, 8, np.random.default_rng(5))
    b = modulate("QPSK", 32, 8, np.random.default_rng(5))
    assert np.array_equal(a, b)
