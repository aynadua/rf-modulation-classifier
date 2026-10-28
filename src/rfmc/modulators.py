"""Baseband modulators: constellations, pulse shaping, and symbol generation.

Every constellation is normalised to unit average symbol energy, so the signal
power at the output of :func:`modulate` is independent of the scheme. That
matters: if 64QAM arrived at the channel with more power than BPSK, a
classifier could separate the two on amplitude alone and the whole benchmark
would be meaningless.
"""

from __future__ import annotations

import numpy as np

# Supported schemes, in a fixed order. The integer index into this tuple is the
# class label used everywhere downstream; do not reorder without regenerating
# the datasets.
SCHEMES: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "PAM4", "16QAM", "64QAM")


def _normalise(points: np.ndarray) -> np.ndarray:
    """Scale a constellation to unit average symbol energy."""
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def _square_qam(order: int) -> np.ndarray:
    """Square QAM constellation of the given order (must be a perfect square)."""
    side = int(round(np.sqrt(order)))
    if side * side != order:
        raise ValueError(f"{order}-QAM is not square")
    levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
    real, imag = np.meshgrid(levels, levels)
    return _normalise((real + 1j * imag).ravel())


def constellation(scheme: str) -> np.ndarray:
    """Return the unit-energy constellation points for ``scheme``."""
    if scheme == "BPSK":
        return _normalise(np.array([1.0, -1.0], dtype=np.complex128))
    if scheme == "QPSK":
        return _normalise(np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128))
    if scheme == "8PSK":
        return np.exp(2j * np.pi * np.arange(8) / 8)
    if scheme == "PAM4":
        return _normalise(np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.complex128))
    if scheme in ("16QAM", "64QAM"):
        return _square_qam(int(scheme.removesuffix("QAM")))
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def rrc_filter(sps: int, span: int, beta: float) -> np.ndarray:
    """Root-raised-cosine impulse response, normalised to unit energy.

    Args:
        sps: samples per symbol.
        span: filter length in symbols (the taps cover ``span * sps + 1`` samples).
        beta: roll-off factor in [0, 1].
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("roll-off beta must lie in [0, 1]")
    n = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64)
    t = n / sps  # time in symbol periods
    h = np.empty_like(t)

    # The closed form has removable singularities at t = 0 and t = +/- 1/(4*beta);
    # both are filled with their analytic limits.
    zero = np.isclose(t, 0.0)
    h[zero] = 1.0 + beta * (4.0 / np.pi - 1.0)

    if beta > 0:
        quarter = np.isclose(np.abs(t), 1.0 / (4.0 * beta))
        h[quarter] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
    else:
        quarter = np.zeros_like(t, dtype=bool)

    rest = ~(zero | quarter)
    tr = t[rest]
    numerator = np.sin(np.pi * tr * (1.0 - beta)) + 4.0 * beta * tr * np.cos(
        np.pi * tr * (1.0 + beta)
    )
    denominator = np.pi * tr * (1.0 - (4.0 * beta * tr) ** 2)
    h[rest] = numerator / denominator

    return h / np.linalg.norm(h)


def modulate(
    scheme: str,
    n_symbols: int,
    sps: int,
    rng: np.random.Generator,
    *,
    beta: float = 0.35,
    span: int = 10,
) -> np.ndarray:
    """Generate a pulse-shaped complex baseband waveform.

    Returns ``n_symbols * sps`` complex samples with unit average power, after
    discarding the filter transient at both ends.
    """
    points = constellation(scheme)
    # Generate extra symbols so that trimming the pulse-shaping transient still
    # leaves a full-length waveform.
    pad = span
    symbols = rng.choice(points, size=n_symbols + 2 * pad)

    upsampled = np.zeros((n_symbols + 2 * pad) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols

    taps = rrc_filter(sps, span, beta)
    shaped = np.convolve(upsampled, taps, mode="same")

    waveform = shaped[pad * sps : (pad + n_symbols) * sps]
    power = np.mean(np.abs(waveform) ** 2)
    if power <= 0:
        raise RuntimeError("generated an all-zero waveform")
    return waveform / np.sqrt(power)
