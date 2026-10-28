"""Propagation and receiver impairments applied to a complex baseband waveform.

The impairments are applied in the order a real link imposes them:

    pulse-shaped signal -> multipath -> clock error -> carrier error -> AWGN

Noise is added last because it enters at the receiver front end, after the
channel has already coloured the signal. Getting this order wrong produces a
dataset that is quietly easier than reality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelParams:
    """Per-realisation impairment settings.

    Attributes:
        snr_db: signal-to-noise ratio in dB, measured before AWGN is added.
        n_taps: number of multipath taps (1 disables multipath).
        rician_k_db: Rician K-factor in dB. ``-inf`` gives a Rayleigh channel.
        delay_spread: exponential power-delay-profile decay constant, in samples.
        cfo_norm: carrier frequency offset normalised to the sample rate
            (cycles per sample).
        phase_off: constant carrier phase offset in radians.
        timing_off: fractional sample timing offset.
        sro_ppm: sample-rate (clock) offset in parts per million.
    """

    snr_db: float
    n_taps: int = 1
    rician_k_db: float = 10.0
    delay_spread: float = 2.0
    cfo_norm: float = 0.0
    phase_off: float = 0.0
    timing_off: float = 0.0
    sro_ppm: float = 0.0


def multipath_taps(params: ChannelParams, rng: np.random.Generator) -> np.ndarray:
    """Draw complex channel taps with an exponential power-delay profile.

    The taps are scaled to unit total power, so multipath reshapes the signal
    without changing its average power -- otherwise it would shift the
    effective SNR and confound the SNR sweep.
    """
    if params.n_taps <= 1:
        return np.array([1.0 + 0j])

    delays = np.arange(params.n_taps)
    power = np.exp(-delays / max(params.delay_spread, 1e-9))

    real = rng.standard_normal(params.n_taps)
    imag = rng.standard_normal(params.n_taps)
    diffuse = (real + 1j * imag) / np.sqrt(2.0)
    taps = np.sqrt(power) * diffuse

    if np.isfinite(params.rician_k_db):
        # A line-of-sight component on the first tap, at the requested K-factor.
        k = 10.0 ** (params.rician_k_db / 10.0)
        los_scale = np.sqrt(k / (k + 1.0)) * np.sqrt(power[0])
        taps[0] = los_scale * np.exp(1j * rng.uniform(0, 2 * np.pi)) + taps[0] / np.sqrt(k + 1.0)

    return taps / np.sqrt(np.sum(np.abs(taps) ** 2))


def apply_multipath(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolve with the channel and keep the first ``len(x)`` samples."""
    if len(taps) == 1:
        return x * taps[0]
    return np.convolve(x, taps)[: len(x)]


def resample_fractional(
    x: np.ndarray, n_out: int, offset: float, rate_error: float, half_width: int = 8
) -> np.ndarray:
    """Resample ``x`` at times ``k * (1 + rate_error) + offset`` for k < n_out.

    This models the receiver's sampling clock being both misaligned (``offset``)
    and slightly wrong in rate (``rate_error``), using a Hann-windowed sinc
    interpolator. ``half_width`` sets the interpolator's one-sided support.
    """
    t = np.arange(n_out) * (1.0 + rate_error) + offset
    base = np.floor(t).astype(np.int64)
    taps = np.arange(-half_width + 1, half_width + 1)

    idx = base[:, None] + taps[None, :]
    frac = t[:, None] - idx

    weights = np.sinc(frac) * np.hanning(2 * half_width + 2)[1:-1][None, :]
    # Preserve DC gain despite the window truncating the sinc.
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    clipped = np.clip(idx, 0, len(x) - 1)
    return np.sum(x[clipped] * weights, axis=1)


def apply_carrier_offset(x: np.ndarray, cfo_norm: float, phase_off: float) -> np.ndarray:
    """Rotate the signal by a residual carrier frequency and phase offset."""
    n = np.arange(len(x))
    return x * np.exp(1j * (2.0 * np.pi * cfo_norm * n + phase_off))


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN so that the result has the requested SNR."""
    signal_power = np.mean(np.abs(x) ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) * np.sqrt(
        noise_power / 2.0
    )
    return x + noise


def apply_channel(
    x: np.ndarray, params: ChannelParams, rng: np.random.Generator, n_out: int | None = None
) -> np.ndarray:
    """Apply the full impairment chain and return ``n_out`` samples."""
    n_out = n_out if n_out is not None else len(x)

    y = apply_multipath(x, multipath_taps(params, rng))
    y = resample_fractional(y, n_out, params.timing_off, params.sro_ppm * 1e-6)
    y = apply_carrier_offset(y, params.cfo_norm, params.phase_off)
    return add_awgn(y, params.snr_db, rng)
