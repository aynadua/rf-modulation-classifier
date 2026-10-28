"""Impairment-chain invariants. These are the properties the dataset relies on."""

import numpy as np
import pytest

from rfmc.channel import (
    ChannelParams,
    add_awgn,
    apply_channel,
    multipath_taps,
    resample_fractional,
)
from rfmc.modulators import modulate


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 10.0, 20.0])
def test_awgn_hits_the_requested_snr(snr_db):
    rng = np.random.default_rng(0)
    signal = modulate("QPSK", 2000, 8, rng)
    noise = add_awgn(signal, snr_db, rng) - signal
    measured = 10 * np.log10(np.mean(np.abs(signal) ** 2) / np.mean(np.abs(noise) ** 2))
    assert measured == pytest.approx(snr_db, abs=0.3)


def test_resampler_is_the_identity_with_no_offset_or_drift():
    x = np.exp(2j * np.pi * 0.01 * np.arange(600))
    out = resample_fractional(x, 500, offset=0.0, rate_error=0.0)
    assert np.allclose(out, x[:500], atol=1e-12)


def test_resampler_tracks_a_known_integer_shift():
    x = np.exp(2j * np.pi * 0.013 * np.arange(600))
    out = resample_fractional(x, 400, offset=3.0, rate_error=0.0)
    assert np.allclose(out, x[3:403], atol=1e-10)


def test_multipath_taps_have_unit_total_power():
    rng = np.random.default_rng(1)
    for n_taps in (1, 2, 4, 8):
        taps = multipath_taps(ChannelParams(snr_db=0.0, n_taps=n_taps), rng)
        assert len(taps) == n_taps
        assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_apply_channel_returns_the_requested_length():
    rng = np.random.default_rng(2)
    signal = modulate("16QAM", 128, 8, rng)
    params = ChannelParams(snr_db=5.0, n_taps=3, cfo_norm=1e-3, timing_off=0.4, sro_ppm=30)
    assert len(apply_channel(signal, params, rng, n_out=512)) == 512


def test_carrier_offset_preserves_power():
    from rfmc.channel import apply_carrier_offset

    rng = np.random.default_rng(3)
    signal = modulate("8PSK", 128, 8, rng)
    rotated = apply_carrier_offset(signal, 1e-3, 0.9)
    assert np.isclose(np.mean(np.abs(rotated) ** 2), np.mean(np.abs(signal) ** 2))
