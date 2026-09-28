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


def analog_response(fc, fs, k, freqs):
    """Continuous-time H(s) that biquad_coeffs discretizes, evaluated at freqs.

    freqs are DIGITAL frequencies in Hz, the same convention as magnitude_db, and
    they are mapped onto the analog axis by the inverse bilinear warp. That warp
    is not optional: biquad_coeffs is a prewarped bilinear transform, so a digital
    frequency f corresponds to the analog frequency fs/pi * tan(pi*f/fs), and
    wc = 2*fs*tan(pi*fc/fs) is the corner the prewarp lands on digital fc. Reading
    freqs as analog radians per second instead is worth up to 23 dB of spurious
    disagreement in the upper stopband, because the two axes part company near
    Nyquist. The warp is only meaningful for 0 < f < fs/2, the same range
    magnitude_db is asked for.

    The prototype is the four-pole closed loop, four unity one-poles in cascade
    with feedback around all four:

        H(s) = 1 / [(1 + a0*s/wc)**4 + k],    a0 = aw(4, 0, k)

    n = 4 only: this specializes aw and bw to the N = 4 case, while its siblings
    take n. The loop's one-pole corner is wc/a0, not the prewarped wc itself. The
    upstream file corroborates this reading: its gaincomp == 3 branch multiplies by
    alpha(N, k)**N, and alpha(N, k) is aw(N, 0, k) = a0, which is exactly the
    factor by which the corner sits below the prewarped one.

    a0 equals 1 exactly at k=0 and k=4, and dips to 1/sqrt(2) near k=0.25. The
    natural short form of this prototype, 1/[(1 + s/wc)**4 + k], takes the corner
    to be wc itself and therefore coincides with the port only at those two
    endpoints. That is why a check written in the short form cannot pass for
    general k: it differs by up to 9.6 dB in the midband for 0 < k < 4. The
    factored two-section form is the derivation, obtained by matching the
    bilinear image of an analog section u**2 + P*u + Q, which is digital
    [1, a1, a2] proportional to [M**2 + P*M + Q, 2*(Q - M**2), M**2 - P*M + Q]
    with M = 1/d, against biquad_coeffs. That gives P_w = 2*b_w/a_0 and
    Q_w = (a_w/a_0)**2 for w = 0, 1, and a0**4 times the resulting product
    collapses back to (1 + a0*u)**4 + k. Concretely, with r = k**(1/4) and
    u = s/wc:

        a0sq = 1 + r**2 - sqrt(2)*r      a1sq = 1 + r**2 + sqrt(2)*r
        b0   = 1 - r/sqrt(2)             b1   = 1 + r/sqrt(2)      a0 = sqrt(a0sq)
        H(s) = 1/a0**4 / [(u**2 + 2*(b0/a0)*u + 1) * (u**2 + 2*(b1/a0)*u + (a1/a0)**2)]

    All of it is closed form in k alone: no call into aw, bw or biquad_coeffs, so a
    wrong cosine argument or a transposed A0BD/A02 index in the port shows up as a
    disagreement here instead of cancelling out. DC gain is 1/(1+k), which is what
    the upstream file's gaincomp == 2 branch compensates by multiplying by (1+k).

    Measured against the port over 10 Hz to 0.45*fs: agreement to 1.5e-13 dB at
    k=0, 1.4e-13 dB at k=2 and 2.7e-11 dB at k=3.99, and 9e-9 dB in the worst case
    of an fc sweep from 50 Hz to 12 kHz. The a0/a1 cross-indexed mutant of
    A0BD/A02 that the task brief warns about is rejected here by 30 dB at k=1 and
    71 dB at k=3.99, though not at k=0, where every aw and bw is 1 and the index
    cannot matter.
    """
    wc = 2.0 * fs * np.tan(np.pi * fc / fs)
    r = k**0.25
    root2 = np.sqrt(2.0)
    a0sq = 1.0 + r * r - root2 * r
    a1sq = 1.0 + r * r + root2 * r
    a0 = np.sqrt(a0sq)
    b0 = 1.0 - r / root2
    b1 = 1.0 + r / root2
    f = np.asarray(freqs, dtype=float)
    u = 2j * (fs * np.tan(np.pi * f / fs)) / wc
    den = (u * u + 2.0 * (b0 / a0) * u + 1.0) * (
        u * u + 2.0 * (b1 / a0) * u + a1sq / a0sq
    )
    return (1.0 / a0**4) / den