"""Classical feature extraction -- the baseline the CNN has to beat.

This is deliberately a *strong* baseline, not a straw man. Higher-order
cumulants only separate modulation schemes when they are computed on
symbol-rate samples: on the 8x-oversampled, RRC-shaped waveform the sample
distribution is driven by the pulse shape and looks nearly Gaussian whatever
the constellation. So the pipeline first recovers symbol timing blindly with
the Oerder-Meyr estimator, decimates to one sample per symbol, and only then
computes cumulants.

Two invariances are built in on purpose, because the receiver does not know
the carrier phase or the absolute gain:

* cumulants are divided by the appropriate power of C21, making them
  scale invariant;
* only cumulant *magnitudes* are used, making them invariant to a constant
  phase rotation.

Sixth-order cumulants are omitted. Their sample variance at a 512-sample
window is large enough to cost more than the extra class separation is worth.

Reference for the cumulant set: Swami & Sadler, "Hierarchical digital
modulation classification using cumulants", IEEE Trans. Comms, 2000.
Reference for the spectral/instantaneous features: Azzouz & Nandi,
"Automatic Modulation Recognition of Communication Signals", 1996.
Reference for the timing estimator: Oerder & Meyr, "Digital filter and square
timing recovery", IEEE Trans. Comms, 1988.
"""

from __future__ import annotations

import numpy as np

from rfmc.channel import resample_fractional

FEATURE_NAMES: tuple[str, ...] = (
    "c20_norm",
    "c40_norm",
    "c41_norm",
    "c42_norm",
    "gamma_max",
    "sigma_aa",
    "sigma_a",
    "kurtosis_a",
    "sigma_ap",
    "sigma_dp",
    "sigma_af",
    "kurtosis_f",
    "papr_db",
    "zero_ratio",
)


def oerder_meyr_timing(x: np.ndarray, sps: int) -> float:
    """Blind symbol-timing estimate, in samples, from the squared-magnitude tone.

    Squaring the signal produces a spectral line at the symbol rate whose phase
    carries the timing offset. Returns an offset in ``[0, sps)``.
    """
    n = (len(x) // sps) * sps
    if n == 0:
        return 0.0
    k = np.arange(n)
    tone = np.sum(np.abs(x[:n]) ** 2 * np.exp(-2j * np.pi * k / sps))
    if tone == 0:
        return 0.0
    epsilon = -np.angle(tone) / (2.0 * np.pi)
    return float((epsilon * sps) % sps)


def symbol_sync(x: np.ndarray, sps: int) -> np.ndarray:
    """Decimate to one sample per symbol at the blindly estimated timing phase."""
    offset = oerder_meyr_timing(x, sps)
    n_symbols = int((len(x) - offset) // sps) - 1
    if n_symbols < 8:
        return x[::sps]
    # Resample at symbol spacing by asking for every sps-th sample: the helper
    # works in units of input samples, so a rate error of (sps - 1) does it.
    return resample_fractional(x, n_symbols, offset, float(sps - 1))


def cumulants(symbols: np.ndarray) -> dict[str, float]:
    """Scale- and rotation-invariant cumulant magnitudes up to fourth order."""
    x = symbols - np.mean(symbols)
    m20 = np.mean(x**2)
    m21 = np.mean(np.abs(x) ** 2)
    m40 = np.mean(x**4)
    m41 = np.mean(x**3 * np.conj(x))
    m42 = np.mean(np.abs(x) ** 4)

    c20 = m20
    c21 = m21
    c40 = m40 - 3.0 * m20**2
    c41 = m41 - 3.0 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2.0 * m21**2

    if c21 <= 0:
        return {"c20_norm": 0.0, "c40_norm": 0.0, "c41_norm": 0.0, "c42_norm": 0.0}

    return {
        "c20_norm": float(np.abs(c20) / c21),
        "c40_norm": float(np.abs(c40) / c21**2),
        "c41_norm": float(np.abs(c41) / c21**2),
        "c42_norm": float(np.abs(c42) / c21**2),
    }


def instantaneous_features(x: np.ndarray) -> dict[str, float]:
    """Amplitude, phase and frequency statistics of the complex envelope."""
    amp = np.abs(x)
    mean_amp = np.mean(amp)
    if mean_amp <= 0:
        return dict.fromkeys(FEATURE_NAMES[4:], 0.0)

    # Normalised-centred instantaneous amplitude.
    a_cn = amp / mean_amp - 1.0
    spectrum = np.abs(np.fft.fft(a_cn)) ** 2 / len(a_cn)
    gamma_max = float(np.max(spectrum[1:]))  # skip DC, which a_cn already removes

    phase = np.unwrap(np.angle(x))
    # Strip the linear trend, which is just the residual carrier offset and
    # says nothing about the modulation.
    idx = np.arange(len(phase))
    slope, intercept = np.polyfit(idx, phase, 1)
    phase_nl = phase - (slope * idx + intercept)

    # Phase is meaningless where the envelope collapses, so weight by amplitude.
    strong = amp > 0.3 * mean_amp
    phase_strong = phase_nl[strong] if strong.sum() > 8 else phase_nl

    freq = np.diff(phase) / (2.0 * np.pi)
    freq = freq - np.mean(freq)
    freq_std = np.std(freq)
    freq_n = freq / freq_std if freq_std > 0 else freq

    a_std = np.std(a_cn)
    return {
        "gamma_max": gamma_max,
        "sigma_aa": float(np.std(np.abs(a_cn))),
        "sigma_a": float(a_std),
        "kurtosis_a": float(np.mean(a_cn**4) / a_std**4) if a_std > 0 else 0.0,
        "sigma_ap": float(np.std(np.abs(phase_strong))),
        "sigma_dp": float(np.std(phase_strong)),
        "sigma_af": float(np.std(np.abs(freq_n))),
        "kurtosis_f": float(np.mean(freq_n**4)),
        "papr_db": float(10.0 * np.log10(np.max(amp**2) / np.mean(amp**2))),
        "zero_ratio": float(np.mean(amp < 0.2 * mean_amp)),
    }


def extract_one(iq: np.ndarray, sps: int) -> np.ndarray:
    """Feature vector for a single ``(2, L)`` IQ example."""
    x = iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)
    feats = cumulants(symbol_sync(x, sps))
    feats.update(instantaneous_features(x))
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract(x: np.ndarray, sps: int) -> np.ndarray:
    """Feature matrix ``(N, F)`` for a batch of ``(N, 2, L)`` IQ examples."""
    return np.stack([extract_one(example, sps) for example in x])
