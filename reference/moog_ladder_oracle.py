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

k is the absolute global feedback gain in [0, 4], with self-oscillation at k=4.
fc is the leading-pole cutoff, not the -3 dB point, as in the linear reference.
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
