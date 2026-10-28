"""Feature-extraction tests.

The important one is :func:`test_cumulants_match_closed_form_theory`. The
normalised fourth-order cumulants of these constellations have exact known
values, derived from the moments of the constellation points:

    |C40|/C21^2    |C42|/C21^2
    BPSK   2.000         2.000
    QPSK   1.000         1.000
    8PSK   0.000         1.000     <- C40 vanishes; the classic 8PSK signature
    PAM4   1.360         1.360
    16QAM  0.680         0.680
    64QAM  0.619         0.619

Checking against these numbers tests the modulator, the Oerder-Meyr timing
recovery and the cumulant estimator jointly. If any one of the three is
wrong, the values move.
"""

import numpy as np
import pytest

from rfmc.features import (
    FEATURE_NAMES,
    cumulants,
    extract,
    extract_one,
    instantaneous_features,
    oerder_meyr_timing,
    symbol_sync,
)
from rfmc.modulators import SCHEMES, modulate

THEORETICAL = {
    "BPSK": (2.000, 2.000),
    "QPSK": (1.000, 1.000),
    "8PSK": (0.000, 1.000),
    "PAM4": (1.360, 1.360),
    "16QAM": (0.680, 0.680),
    "64QAM": (0.619, 0.619),
}


@pytest.mark.parametrize("scheme", SCHEMES)
def test_cumulants_match_closed_form_theory(scheme):
    rng = np.random.default_rng(0)
    waveform = modulate(scheme, n_symbols=4000, sps=8, rng=rng)
    # A trace of noise only, so the estimate is dominated by the constellation.
    waveform = waveform + (
        rng.standard_normal(len(waveform)) + 1j * rng.standard_normal(len(waveform))
    ) * np.sqrt(0.5 * 1e-4)

    measured = cumulants(symbol_sync(waveform, sps=8))
    expected_c40, expected_c42 = THEORETICAL[scheme]
    assert measured["c40_norm"] == pytest.approx(expected_c40, abs=0.12)
    assert measured["c42_norm"] == pytest.approx(expected_c42, abs=0.12)


def test_c20_separates_real_from_complex_constellations():
    """|C20|/C21 is ~1 for one-dimensional constellations and ~0 otherwise."""
    rng = np.random.default_rng(1)
    for scheme in ("BPSK", "PAM4"):
        value = cumulants(symbol_sync(modulate(scheme, 3000, 8, rng), 8))["c20_norm"]
        assert value > 0.8, scheme
    for scheme in ("QPSK", "8PSK", "16QAM", "64QAM"):
        value = cumulants(symbol_sync(modulate(scheme, 3000, 8, rng), 8))["c20_norm"]
        assert value < 0.2, scheme


def test_cumulants_are_invariant_to_phase_rotation():
    rng = np.random.default_rng(2)
    waveform = modulate("16QAM", 3000, 8, rng)
    plain = cumulants(symbol_sync(waveform, 8))
    rotated = cumulants(symbol_sync(waveform * np.exp(1j * 1.234), 8))
    for key in ("c40_norm", "c42_norm"):
        assert plain[key] == pytest.approx(rotated[key], abs=0.05)


def test_cumulants_are_invariant_to_gain():
    rng = np.random.default_rng(3)
    waveform = modulate("QPSK", 3000, 8, rng)
    plain = cumulants(symbol_sync(waveform, 8))
    scaled = cumulants(symbol_sync(waveform * 7.5, 8))
    for key in plain:
        assert plain[key] == pytest.approx(scaled[key], abs=1e-6)


@pytest.mark.parametrize("delay", [0.0, 1.0, 2.5, 5.0, 7.0])
def test_oerder_meyr_recovers_a_known_timing_offset(delay):
    """The estimate must track where the symbol instants actually land.

    :func:`modulate` places symbols at sample indices that are multiples of
    ``sps``. Advancing the signal by ``delay`` samples moves those instants to
    ``-delay`` modulo ``sps``, and that is what the estimator should report.
    """
    from rfmc.channel import resample_fractional

    rng = np.random.default_rng(4)
    waveform = modulate("QPSK", 600, 8, rng)
    delayed = resample_fractional(waveform, 4000, offset=delay, rate_error=0.0)

    estimate = oerder_meyr_timing(delayed, sps=8)
    expected = (-delay) % 8
    # The estimator is only defined modulo one symbol period.
    error = (estimate - expected) % 8
    assert min(error, 8 - error) < 0.6


def test_cumulants_degrade_gracefully_on_a_dead_signal():
    assert cumulants(np.zeros(64, dtype=complex))["c42_norm"] == 0.0


def test_instantaneous_features_are_finite():
    rng = np.random.default_rng(5)
    values = instantaneous_features(modulate("64QAM", 512, 8, rng))
    assert all(np.isfinite(v) for v in values.values())


def test_extract_returns_the_declared_feature_vector():
    rng = np.random.default_rng(6)
    iq = np.stack(
        [modulate("QPSK", 64, 8, rng).real, modulate("QPSK", 64, 8, rng).imag]
    ).astype(np.float32)
    assert extract_one(iq, 8).shape == (len(FEATURE_NAMES),)
    assert extract(np.stack([iq, iq]), 8).shape == (2, len(FEATURE_NAMES))
