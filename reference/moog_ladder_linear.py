#!/usr/bin/env python
"""Analytic reference for the generalized Moog ladder, linear analysis.

Port of D'Angelo's moog_ladder_linear.m (see reference/upstream/PROVENANCE.md)
from Part I of "Generalized Moog Ladder Filter: Part I-Linear Analysis and
Parameterization", IEEE/ACM TASLP 22(12), 2014.

Scope: even filter orders, time-invariant parameters, float64 only. The upstream
file also implements odd orders, per-sample parameter vectors and per-sample
time-varying coefficients. This evaluation uses none of that, so none of it is
ported.

Parameter conventions, taken from the upstream file:

- k is the ABSOLUTE global feedback gain, in [0, 4] for n=4. Normalized k in
  [0,1] maps to absolute via k *= knorm_factor(n), which is 4.0 for n=4.
  Self-oscillation onset is at absolute k = 4.
- fc is the cutoff frequency of the leading poles, NOT the -3 dB frequency of
  the realized cascade. At k=0 the ratio of -3 dB frequency to fc is about
  0.4348, and it drifts with fc. That relationship is pinned by
  tests/test_linear_reference.py::test_fc_to_minus3db_ratio_is_pinned. Never
  conflate the two.
"""

import numpy as np


def aw(n, w, k):
    """Port of upstream Aw(n, w, k)."""
    return np.sqrt(
        1.0 + k ** (2.0 / n) - 2.0 * k ** (1.0 / n) * np.cos((2 * w + 1) * np.pi / n)
    )


def bw(n, w, k):
    """Port of upstream Bw(n, w, k)."""
    return 1.0 - k ** (1.0 / n) * np.cos((2 * w + 1) * np.pi / n)


def alpha(n, k):
    """Port of upstream alpha(n, k). Equals 1 + k for n == 1, else Aw(n, 0, k)."""
    if n == 1:
        return 1.0 + k
    return aw(n, 0, k)


def knorm_factor(n):
    """sec(pi/n)**n, relating normalized to absolute k. Equals 4.0 for n=4."""
    return (1.0 / np.cos(np.pi / n)) ** n


def biquad_coeffs(n, fs, fc, k):
    """Biquad coefficients for the even-order ladder cascade.

    Direct port of the "Coefficients" block of moog_ladder_linear.m, lines 125
    to 142. Returns (b0, a1, a2) of length n // 2, for sections whose numerator
    is b0 * [1, 2, 1] and denominator is [1, a1, a2].
    """
    d = np.tan(np.pi / fs * fc)
    half = n // 2
    a = np.array([aw(n, w, k) for w in range(half)])
    b = np.array([bw(n, w, k) for w in range(half)])
    d2 = d * d
    ad2 = (a * a) * d2
    a0bd = (a[0] * d) * b
    a02 = a[0] * a[0]
    e = ad2 + 2.0 * a0bd + a02
    b0 = d2 / e
    a1 = 2.0 * (ad2 - a02) / e
    a2 = 1.0 - 4.0 * a0bd / e
    return b0, a1, a2


def frequency_response(n, fs, fc, k, freqs):
    """Complex response of the cascade at freqs, given in Hz."""
    from scipy import signal

    b0, a1, a2 = biquad_coeffs(n, fs, fc, k)
    h = np.ones(len(freqs), dtype=complex)
    for w in range(n // 2):
        r = signal.freqz(
            b0[w] * np.array([1.0, 2.0, 1.0]),
            [1.0, a1[w], a2[w]],
            worN=freqs,
            fs=fs,
        )
        h = h * (r[1] if len(r) > 1 else r)
    return h


def magnitude_db(n, fs, fc, k, freqs):
    """20*log10 magnitude of the cascade at freqs, given in Hz."""
    return 20.0 * np.log10(np.abs(frequency_response(n, fs, fc, k, freqs)))