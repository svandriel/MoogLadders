#!/usr/bin/env python
"""Nonlinear oracle for the generalized Moog ladder.

Port of D'Angelo's moog_ladder_nonlinear.m (see
reference/upstream/PROVENANCE.md), from Part II of "Generalized Moog Ladder
Filter: Part II-Explicit Nonlinear Model through a Novel Delay-Free
Implementation Method", IEEE/ACM TASLP 22(12), 2014.

This is a MODEL of the analog circuit, not a measurement of hardware. Nothing
downstream may describe results from this module as hardware-referenced.

Always float64. The sample loop is a direct port and is not optimized; the
upstream Octave loop is equally sequential.

Parameter conventions and scope, as in the linear sibling's header:

- k is the ABSOLUTE global feedback gain, self-oscillation at k=4. Never the
  normalized gain, so the upstream knorm branch is not ported.
- fc is the leading-pole cutoff, NOT the -3 dB point, and the upstream's
  fccomp = true. The .m's `fc .*= alpha(k)` at lines 100 to 102 is deliberately
  NOT applied, exactly as in moog_ladder_linear, which is why this oracle and the
  linear reference agree in the linear limit.
- DC gain is -1/(1+k): inverted, and down by (1+k). Neither is compensated here,
  because the upstream gaincomp argument is not ported. A caller wanting a
  positive unity DC must multiply by -(1+k) itself. The inversion is upstream's
  sign convention, not a defect: the 2013 model in moog_ladder_old.m shares it,
  deriving the identical -1/(1+k) law at DC. Never "fix" it in isolation.
- fc and k are NOT clamped. The .m runs limit() at lines 113 and 114 and warns;
  here an out-of-range sweep value silently returns finite, meaningless output
  (fc above fs/2 turns tan(pi*fc/fs) negative and the filter runs backwards), so
  clamp in the caller rather than trusting a warning that will not come.
- Time-invariant scalar parameters only. Per-sample fc/k vectors are not
  supported and raise a broadcast error rather than working. The loop itself is
  generic in n, but only n=4 is validated, and the linear reference it is checked
  against is even-order only.
"""

import numpy as np

from reference.moog_ladder_linear import alpha

VT = 26e-3


def _binomial_row(n, orders):
    """Binomial coefficients C(n, j) for j in `orders`, as float array.

    Upstream uses Octave's bincoeff(n, 1:n). Built with the multiplicative
    recurrence so it works for numpy array input.
    """
    orders = np.asarray(orders, dtype=float)
    out = np.ones_like(orders)
    for step in range(1, int(orders.max()) + 1):
        out = np.where(orders >= step, out * (n - (step - 1)) / step, out)
    return out


def coefficients(n, fs, fc, k):
    """Coefficients from the "Coefficients" block, lines 125 to 147.

    Returns a dict with keys g, p0s, q0s, r1s, k0s, rg, qg, k0g. This evaluation
    never uses time-varying k, so the upstream per-sample vectorization
    collapses to scalars plus one length-n vector.
    """
    g = np.tan(np.pi / fs * fc) / alpha(n, k)
    vt2 = 2.0 * VT
    vt2i = 1.0 / vt2

    p0s = 1.0 / (1.0 + g)
    q0s = 1.0 - g
    r1s = -g
    k0s = vt2 * g * p0s

    gn = (1.0 - p0s) ** n
    kgn = k * gn
    p0g = 1.0 / (1.0 + kgn)
    bin_ = _binomial_row(n, np.arange(1, n + 1))
    orders = np.arange(1, n + 1, dtype=float)
    rg = -bin_ * kgn
    qg = rg - bin_ * ((g - 1.0) * p0s) ** orders
    k0g = -vt2i * p0g

    return {
        "g": g,
        "p0s": p0s,
        "q0s": q0s,
        "r1s": r1s,
        "k0s": k0s,
        "rg": rg,
        "qg": qg,
        "k0g": k0g,
    }


def process(x, fs, fc, k, n=4):
    """Run the Part II delay-free nonlinear model over x in float64.

    Direct port of the "Filter" block, lines 149 to 173. The inner variable is
    named stage_out rather than reusing y, because y is the output array and the
    two are easy to confuse.
    """
    x = np.asarray(x, dtype=np.float64)
    c = coefficients(n, fs, fc, k)

    y = np.zeros_like(x)
    si = np.zeros(n)
    sf = np.zeros(n)
    sg = np.zeros(n)

    k0s = c["k0s"]
    r1s = c["r1s"]
    q0s = c["q0s"]
    k0g = c["k0g"]
    rg = c["rg"]
    qg = c["qg"]
    vt2i = 1.0 / (2.0 * VT)

    for i in range(len(x)):
        yo = np.tanh(k0g * (x[i] + sg[0]))
        for j in range(n):
            yi = yo
            yd = k0s * (yi + sf[j])
            stage_out = yd + si[j]
            yo = np.tanh(vt2i * stage_out)
            si[j] = yd + stage_out
            sf[j] = r1s * yi - q0s * yo
        y[i] = stage_out
        yf = k * stage_out
        for j in range(n - 1):
            sg[j] = rg[j] * x[i] + qg[j] * yf + sg[j + 1]
        sg[n - 1] = rg[n - 1] * x[i] + qg[n - 1] * yf

    return y
