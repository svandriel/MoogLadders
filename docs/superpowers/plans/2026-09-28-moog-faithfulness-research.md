# Moog Ladder Faithfulness Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python evaluation suite that ranks the repo's 12 digital Moog ladder models
against an analytic transfer function and a nonlinear circuit oracle, then write the resulting
engineering report to `docs/moog-faithfulness-report.md`.

**Architecture:** Two reference implementations are ported from D'Angelo's published Octave code
and must pass validation gates before they are allowed to score anything. The linear axis
compares measured model response against the analytic continuous-time H(s). The nonlinear axis
compares model output against the Part II circuit oracle. Model outputs come from driving
`RunFilters` with float32 WAVs, added in Task 6, so distortion is not floored at the 16-bit
noise level.

**Tech Stack:** Existing C++14 library plus RtAudio-free build via `RTAUDIO_DUMMY=ON`. Python
with numpy, scipy, matplotlib, pytest. Reference papers: S. D'Angelo and V. Valimaki,
"Generalized Moog Ladder Filter", Part I and Part II, IEEE/ACM TASLP 22(12), December 2014.

## Global Constraints

- Sample rate is **44100 Hz** for every test unless a step says otherwise.
- Signals are generated in **float64**. Only the handoff to `RunFilters` is float32.
- The nonlinear oracle always runs in **float64**, in-process. Never through a subprocess or WAV.
- The linear score is computed **in band only**, up to **0.4*fs** (17640 Hz). Out-of-band
  response is reported separately and excluded from every score.
- Reference `k` is the **absolute** feedback gain in **[0, 4]**. Self-oscillation is at
  **k = 4**. Normalized `k` in [0,1] maps to absolute via `k *= sec(pi/n)**n`, which is
  **4.0** for n=4.
- Filter order is **n = 4** everywhere.
- Never claim hardware-referenced accuracy. The nonlinear axis is model-referenced only.
- `scripts/filter_verification.py` must not be modified. It stays a smoke test.
- Default 16-bit WAV output from `RunFilters` must stay byte-identical. `--float` is opt-in.
- The 12 model names, in `example/helpers.hpp` enum order, with exact spelling:
  Stilson, Simplified, Huovilainen, Improved, **Krajeski**, RKSimulation, Microtracker, MusicDSP,
  OberheimVariation, Hyperion, HyperionTanh, HyperionLegacy.
- Python style follows `scripts/filter_verification.py`: shebang, module docstring with usage,
  `argparse`, `pathlib.Path`, dataclasses, `typing`, lazy imports for matplotlib and scipy.
- Commit messages are lowercase, terse, no conventional-commit prefixes, matching existing
  history such as `bump to cpp14` and `fix include`.

## scipy 1.18 API Landmines

Every one of these was hit while preparing this plan. Each has a test that catches a regression.

- `signal.bilinear(b, a, fs)` returns `(beta, alpha)`, **not** `(z, p, k)`, and has no `prewarp`
  keyword in 1.18. Prewarp by scaling the analog frequency before the call.
- `signal.freqz(...)` and `signal.zpk2tf(...)` return a **single array** when `worN` is an array,
  and a `(w, h)` tuple` when `worN` is an integer. Always normalize with
  `r[1] if len(r) > 1 else r`.
- Prewarp relation: to land the analog corner on digital `fc`, use `wc = 2.0 * fs * tan(pi*fc/fs)`.
  Dividing by the tangent instead gives a corner three orders of magnitude too wide.

## Two Facts Established During Planning

These are measured, not assumed, and both have tests. Do not "simplify" either away.

1. **The reference's `fc` is not the -3 dB point of the realized cascade.** At k=0 the ratio
   is about 0.4348 at fc=1000, and it drifts slightly with fc: 0.4342 at fc=100, 0.4496 at
   fc=5000. Any code that assumes `fc` equals the -3 dB frequency is wrong.
2. **The K=4 identity holds exactly.** With `wc = 2*fs*tan(pi*fc/fs)` and fc=1000, the analog
   poles at k=4 all sit at 1001.6951 Hz, which is `wc/2pi`, the prewarped corner. This is the
   guard against a wrong `k` convention, where normalized and absolute get swapped and every
   resonance number inverts.

## File Structure

Created:

- `reference/upstream/moog_ladder_linear.m`, `moog_ladder_nonlinear.m`, `moog_ladder_old.m` -
  vendored upstream Octave, unmodified
- `reference/upstream/PROVENANCE.md` - source URLs, MIT license, errata status
- `reference/__init__.py` - empty
- `reference/moog_ladder_linear.py` - analytic H(s), aw/bw/alpha helpers, biquad coefficients
- `reference/moog_ladder_oracle.py` - Part II delay-free nonlinear model, float64
- `scripts/faithfulness_eval.py` - signals, metrics, scoring, figures, report data
- `tests/test_linear_reference.py` - validation gates for the linear reference
- `tests/test_oracle.py` - validation gates for the nonlinear oracle
- `tests/test_eval_metrics.py` - self-validation of the measurement code
- `tests/test_runfilters_float.py` - WAV format round-trip and the byte-identical default
- `docs/moog-faithfulness-report.md` - the deliverable
- `docs/moog-faithfulness/plots/` - six committed figures

Modified:

- `example/helpers.hpp` - add `WriteWavFileFloat`, leave `WriteWavFile` untouched
- `example/run-filters.cpp` - add `--float`, plumb through to the writer

---

### Task 1: Vendor the upstream Octave references

The reference code must be in the repo so the port is reviewable against its source, and so the
validation gates mean something. It is MIT licensed, so vendoring is fine with attribution.

**Files:**
- Create: `reference/upstream/moog_ladder_linear.m`
- Create: `reference/upstream/moog_ladder_nonlinear.m`
- Create: `reference/upstream/moog_ladder_old.m`
- Create: `reference/upstream/PROVENANCE.md`

**Interfaces:**
- Consumes: nothing
- Produces: the files Tasks 3 and 5 port from. No Python symbols.

- [x] **Step 1: Verify the vendored files are present and transfer-clean**

The files were downloaded during planning. Confirm they are intact and free of CRLF:

```bash
cd /home/sander/projects/github/ddiakopoulos/MoogLadders
wc -l reference/upstream/*.m
grep -l $'\r' reference/upstream/*.m || echo "no CRLF"
```

Expected: 230, 209 and 141 lines, then `no CRLF`.

- [x] **Step 2: Write PROVENANCE.md**

```markdown
# Upstream reference code

These Octave/MATLAB files are vendored verbatim from Stefano D'Angelo's public
site. They are the authority for the Python ports in `reference/`. Do not edit
them. When changing a port, diff against these.

| file | source |
| --- | --- |
| `moog_ladder_linear.m` | https://dangelo.audio/assets/code/moog_ladder_linear.m |
| `moog_ladder_nonlinear.m` | https://dangelo.audio/assets/code/moog_ladder_nonlinear.m |
| `moog_ladder_old.m` | https://dangelo.audio/assets/code/moog_ladder_old.m |

Retrieved 2026-09-28. MIT licensed, Copyright (C) 2014 Stefano D'Angelo. The
license text is reproduced in the header of each file.

Papers:

- Part I: S. D'Angelo and V. Valimaki, "Generalized Moog Ladder Filter:
  Part I-Linear Analysis and Parameterization", IEEE/ACM TASLP, 22(12),
  1825-1832, 2014. DOI 10.1109/TASLP.2014.2352495
- Part II: S. D'Angelo and V. Valimaki, "Generalized Moog Ladder Filter:
  Part II-Explicit Nonlinear Model through a Novel Delay-Free Implementation
  Method", IEEE/ACM TASLP, 22(12), 1873-1883, 2014.
  DOI 10.1109/TASLP.2014.2352556

## Errata status

Author-published errata:

- https://dangelo.audio/assets/doc/errata_gladder1.pdf
- https://dangelo.audio/assets/doc/errata_gladder2.pdf

STATUS: read. Both PDFs are single-page and machine-readable; they were read on
2026-09-29 by text extraction (`pypdf`). All three corrections were checked
against the ports and none applies; the per-correction record with page numbers
is in `reference/upstream/PROVENANCE.md`.
```

> **Updated by Task 16, after both PDFs were read.** The STATUS above replaces the
> "not yet read" text this plan originally specified, which no longer describes
> reality. The corrections, one line each:
>
> - Part I, p.1827: a prose sentence should name the leading-pole property `Q`
>   alongside `fc`. **Does not apply**, because it is a correction to the
>   published prose: the port implements coefficient equations (`biquad_coeffs`,
>   `alpha`, `knorm_factor` in `reference/moog_ladder_linear.py`), which the
>   sentence does not change, and Q is not a parameter the port computes.
> - Part I, p.1828: `fc_hat = fc/A0(k)` should read `fc_hat = fc/alpha(k)`.
>   **Does not apply**, because the ports already use alpha(k) everywhere:
>   `reference/upstream/moog_ladder_linear.m:96` (`fc .*= alpha(k)`),
>   `reference/moog_ladder_linear.py:40` and `reference/moog_ladder_oracle.py:65`
>   (`alpha(n, k)`). The erroneous `A0(k)` form appears nowhere in the ladder
>   ports; those `A0`/`A02` hits are the bilinear-transform intermediates in
>   upstream, and an unrelated allpass filter in the halfband generator.
> - Part II, p.1879: the caption of Fig. 6 should begin "Non-solid lines
>   represent...". **Does not apply**, because it is a figure-caption
>   correction, and `reference/moog_ladder_oracle.py` implements the delay-free
>   loop equations, which a caption does not change.
>
> No correction applies to a port, so no measured number in the report moves.
> Task 16 also replaced the report's erratum bullet with the "were read, none
> applies" statement the same finding requires.

- [x] **Step 3: Flag the errata gap in the design doc's terms**

Because the errata are unread, add one line to the report's threats-to-validity section when
Task 15 writes it. Do not skip the step and do not claim the errata were applied.

Done: the flag was written when Task 15 built the threats-to-validity section, phrased as
"the D'Angelo errata were not read". Task 16 read both PDFs and rewrote the bullet to the
"were read, none of the three corrections applies" statement the same section now carries.

- [x] **Step 4: Commit**

```bash
git add reference/upstream/
git commit -m "vendor dangelo octave references with provenance"
```

---

### Task 2: Set up the Python test harness

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `reference/__init__.py` (empty)
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing
- Produces: pytest available in `.venv`, `reference` importable as a package

- [x] **Step 1: Install pytest into the venv**

```bash
cd /home/sander/projects/github/ddiakopoulos/MoogLadders
.venv/bin/python -m pip install pytest
```

- [x] **Step 2: Write a smoke test that proves the package imports**

`tests/test_smoke.py`:

```python
import reference


def test_reference_package_imports():
    assert reference is not None
```

- [x] **Step 3: Run it**

Run: `.venv/bin/python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add tests/ reference/__init__.py
git commit -m "add pytest harness for reference validation"
```

---

### Task 3: Port the linear reference and pin its conventions

Port `moog_ladder_linear.m`. Only the even-order, time-invariant, float64 path is needed, since
this evaluation always uses scalar `fc` and `k` with n=4. Everything else in the Octave file is
dropped deliberately, and the docstring says so.

**Files:**
- Create: `reference/moog_ladder_linear.py`
- Test: `tests/test_linear_reference.py`

**Interfaces:**
- Consumes: `reference/upstream/moog_ladder_linear.m`
- Produces, all in `reference/moog_ladder_linear.py`:
  - `aw(n: int, w: int, k: float) -> float`
  - `bw(n: int, w: int, k: float) -> float`
  - `alpha(n: int, k: float) -> float`
  - `knorm_factor(n: int) -> float`
  - `biquad_coeffs(n, fs, fc, k) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` returning
    `(b0, a1, a2)`, each of length `n // 2`
  - `frequency_response(n, fs, fc, k, freqs) -> np.ndarray` complex response at `freqs` in Hz
  - `magnitude_db(n, fs, fc, k, freqs) -> np.ndarray` 20*log10 magnitude

- [x] **Step 1: Write the failing test for the helpers**

`tests/test_linear_reference.py`:

```python
import math

import numpy as np
import pytest

from reference import moog_ladder_linear as mll

N = 4
FS = 44100.0


def test_knorm_factor_is_four_for_n4():
    assert mll.knorm_factor(4) == pytest.approx(4.0, abs=1e-12)


def test_alpha_at_zero_k_is_one():
    assert mll.alpha(4, 0.0) == pytest.approx(1.0, abs=1e-12)


def test_alpha_at_k_four_is_one():
    # alpha(4,4) = sqrt(1 + 4^0.5 - 2*4^0.25*cos(pi/4)) = 1
    assert mll.alpha(4, 4.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("w", [0, 1])
def test_aw_matches_closed_form(w):
    expected = math.sqrt(
        1.0
        + 2.0 ** (2.0 / N)
        - 2.0 * 2.0 ** (1.0 / N) * math.cos((2 * w + 1) * math.pi / N)
    )
    assert mll.aw(N, w, 2.0) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("w", [0, 1])
def test_bw_matches_closed_form(w):
    expected = 1.0 - 2.0 ** (1.0 / N) * math.cos((2 * w + 1) * math.pi / N)
    assert mll.bw(N, w, 2.0) == pytest.approx(expected, rel=1e-12)
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_linear_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reference.moog_ladder_linear'`

- [x] **Step 3: Write the module with the helpers and the response functions**

`reference/moog_ladder_linear.py`:

```python
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
```

- [x] **Step 4: Run to verify the helper tests pass**

Run: `.venv/bin/python -m pytest tests/test_linear_reference.py -v`
Expected: PASS, 7 tests

- [x] **Step 5: Write the tests that pin the DC gain, monotonicity, and the fc/-3dB ratio**

Append to `tests/test_linear_reference.py`:

```python
def _mag_db(fc, k, freqs):
    return mll.magnitude_db(N, FS, fc, k, freqs)


@pytest.mark.parametrize("fc", [100.0, 1000.0, 5000.0])
def test_dc_gain_is_exactly_one(fc):
    # Measured during planning: section DC gain came out to 1.0000000000000024,
    # unity to float64 round-off. A wrong cosine argument in aw/bw breaks this.
    assert _mag_db(fc, 0.0, np.array([1e-3, 1.0, 10.0]))[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("fc", [100.0, 1000.0, 5000.0])
def test_response_is_monotonically_decreasing_at_zero_k(fc):
    freqs = np.logspace(1, np.log10(0.45 * FS), 2000)
    d = _mag_db(fc, 0.0, freqs)
    assert np.all(np.diff(d) <= 1e-9)


def test_fc_to_minus3db_ratio_is_pinned():
    """fc is the leading-pole cutoff, not the -3 dB point of the cascade.

    Measured during planning at k=0: the ratio is 0.4342 at fc=100, 0.4348 at
    fc=1000 and 0.4496 at fc=5000, so it is mildly fc-dependent. This test
    records the fc=1000 value so any future port change is caught. The
    evaluation code must never assume the ratio is 1.
    """
    freqs = np.logspace(1, np.log10(0.45 * FS), 200000)
    d = _mag_db(1000.0, 0.0, freqs)
    f3 = freqs[int(np.argmin(np.abs(d + 3.0)))]
    assert f3 / 1000.0 == pytest.approx(0.4348, abs=5e-4)
```

- [x] **Step 6: Run these three tests**

Run: `.venv/bin/python -m pytest tests/test_linear_reference.py -v`
Expected: all PASS. If `test_fc_to_minus3db_ratio_is_pinned` fails, do **not** relax the expected
value to make it green. Re-derive the coefficients by hand against the Octave source and find
the real discrepancy.

- [x] **Step 7: Write the K=4 identity test, the load-bearing gate**

Append to `tests/test_linear_reference.py`:

```python
def test_k_four_is_marginal_stability_with_poles_at_the_corner():
    """At absolute k=4 the 4-pole ladder is marginally stable.

    The analog transfer function is H(s) = wc**4 / ((s+wc)**4 + k*wc**4). At k=4
    the denominator (s+wc)**4 + 4*wc**4 has all four roots on the imaginary axis
    at s = +/- j*wc, so the pole frequency equals the corner frequency.

    Verified during planning: with fc=1000 and wc = 2*fs*tan(pi*fc/fs), the roots
    at k=4 all sit at 1001.6951 Hz against a corner of 1000 Hz.

    This is the guard against a wrong k convention. If normalized and absolute get
    swapped anywhere, this fails and every resonance number in the report is wrong.
    """
    fc = 1000.0
    wc = 2.0 * FS * np.tan(np.pi * fc / FS)
    roots = np.roots([1.0, 4.0 * wc, 6.0 * wc**2, 4.0 * wc**3, wc**4 * 5.0])
    pole_hz = np.sort(np.abs(roots.imag)) / (2.0 * np.pi)
    assert len(pole_hz) == 4
    assert np.allclose(pole_hz, wc / (2.0 * np.pi), rtol=1e-9)


def test_peak_gain_grows_toward_k_four():
    freqs = np.logspace(1, np.log10(0.45 * FS), 20000)
    peak3 = _mag_db(1000.0, 3.0, freqs).max()
    peak399 = _mag_db(1000.0, 3.99, freqs).max()
    assert peak399 > peak3
    assert peak399 > 40.0
```

- [x] **Step 8: Run all linear reference tests**

Run: `.venv/bin/python -m pytest tests/test_linear_reference.py -v`
Expected: all PASS

- [x] **Step 9: Commit**

```bash
git add reference/moog_ladder_linear.py tests/test_linear_reference.py
git commit -m "port linear ladder reference and pin its conventions"
```

---

### Task 4: Check the linear port against an independent derivation

The port in Task 3 could be wrong in a way that stays self-consistent, for example a transposed
cosine argument. This task checks it against a transfer function derived from the circuit rather
than from the paper's coefficient formulas.

The analog Moog ladder is four one-pole sections in cascade with the feedback wrapped around all
four, so `H(s) = wc**4 / ((s + wc)**4 + k*wc**4)`. This comes from the circuit, not from
`aw`/`bw`. If the port's cosine arguments were wrong, the two disagree.

**Files:**
- Modify: `reference/moog_ladder_linear.py` (add `analog_response`)
- Test: `tests/test_linear_reference.py`

**Interfaces:**
- Consumes: `magnitude_db` from Task 3
- Produces: `analog_response(fc, fs, k, freqs) -> np.ndarray`, complex continuous-time response
  using `wc = 2*fs*tan(pi*fc/fs)`

- [x] **Step 1: Write the failing test**

```python
def test_ported_coeffs_match_independent_analog_derivation():
    """The ported cascade must equal H(s) = wc^4 / ((s+wc)^4 + k*wc^4).

    The two agreed to about 1e-6 dB across the passband during planning. The
    comparison is restricted to where the ported response is above -120 dB,
    because differencing two numerically-zero stopbands in dB produces
    meaningless tens-of-dB errors. That mistake was made and caught during
    planning; do not reintroduce it.
    """
    fc = 1000.0
    freqs = np.logspace(1, np.log10(0.45 * FS), 4000)
    for k in (0.0, 1.0, 2.0, 3.0, 3.99):
        ported = mll.magnitude_db(N, FS, fc, k, freqs)
        analog = 20.0 * np.log10(np.abs(mll.analog_response(fc, FS, k, freqs)))
        band = ported > -120.0
        assert np.max(np.abs(ported[band] - analog[band])) < 1e-3
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_linear_reference.py::test_ported_coeffs_match_independent_analog_derivation -v`
Expected: FAIL with `AttributeError: module has no attribute 'analog_response'`

- [x] **Step 3: Add analog_response**

Append to `reference/moog_ladder_linear.py`:

```python
def analog_response(fc, fs, k, freqs):
    """Continuous-time H(s) = wc**4 / ((s + wc)**4 + k * wc**4), from the circuit.

    Four one-pole sections in cascade with feedback around all four. This is an
    independent check on biquad_coeffs, which comes from the paper's closed
    forms instead. wc is prewarped so the analog corner lands on digital fc.

    The relation is wc = 2*fs*tan(pi*fc/fs). Dividing by the tangent instead
    gives a corner three orders of magnitude too wide; that mistake was made and
    caught during planning.
    """
    wc = 2.0 * fs * np.tan(np.pi * fc / fs)
    s = 2j * np.pi * np.asarray(freqs, dtype=float)
    return wc**4 / ((s + wc) ** 4 + k * wc**4)
```

- [x] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_linear_reference.py -v`
Expected: all PASS

- [x] **Step 5: If it fails, do not relax the threshold**

A failure means the port is wrong. Recheck `reference/upstream/moog_ladder_linear.m` lines 125
to 142. In particular `A0BD` and `A02` both index `A(1,:)`, which in Octave's one-based notation
is the **first** element, that is `aw(n, 0, k)`. A port that used `a[1]` there would fail this
test and nothing else.

- [x] **Step 6: Commit**

```bash
git add reference/moog_ladder_linear.py tests/test_linear_reference.py
git commit -m "check linear port against independent circuit derivation"
```

---

### Task 5: Port the nonlinear oracle

Port `moog_ladder_nonlinear.m`. The filter loop is a sample-by-sample delay-free implementation
and ports directly to a Python loop. It is slow. Do not optimize it without a failing timing
test, and do not add a chunking scheme speculatively.

**Files:**
- Create: `reference/moog_ladder_oracle.py`
- Test: `tests/test_oracle.py`

**Interfaces:**
- Consumes: `alpha` from `reference.moog_ladder_linear`
- Produces:
  - `coefficients(n, fs, fc, k) -> Dict[str, Any]` with keys g, p0s, q0s, r1s, k0s, rg, qg, k0g
  - `process(x: np.ndarray, fs: float, fc: float, k: float, n: int = 4) -> np.ndarray`, float64,
    same length as `x`

- [x] **Step 1: Write the failing tests**

`tests/test_oracle.py`:

```python
import numpy as np
import pytest

from reference import moog_ladder_linear as mll
from reference import moog_ladder_oracle as oracle

FS = 44100.0
N = 4


def test_zero_input_gives_zero_output():
    y = oracle.process(np.zeros(1000), FS, 1000.0, 2.0, N)
    assert np.all(y == 0.0)


def test_output_is_finite_and_bounded_for_small_input():
    x = 0.01 * np.sin(2 * np.pi * 440.0 * np.arange(4410) / FS)
    y = oracle.process(x, FS, 1000.0, 2.0, N)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 1.0


def test_dc_gain_is_one_at_zero_k():
    """With k=0 a small DC input must pass through at unity.

    The tanh stages saturate above roughly 0.5 V internally, so 0.05 exercises the
    linear region. This pins the 1/g scaling in g = tan(pi*fc/fs)/alpha(n,k),
    which is the easiest coefficient in the file to mistranscribe.
    """
    x = np.full(20000, 0.05)
    y = oracle.process(x, FS, 200.0, 0.0, N)
    assert y[-1000:].mean() == pytest.approx(0.05, rel=1e-3)


def test_larger_k_gives_larger_peak_gain():
    x = 0.05 * np.sin(2 * np.pi * 1000.0 * np.arange(44100) / FS)
    low = oracle.process(x, FS, 1000.0, 1.0, N)
    high = oracle.process(x, FS, 1000.0, 3.0, N)
    assert np.max(np.abs(high)) > np.max(np.abs(low))
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reference.moog_ladder_oracle'`

- [x] **Step 3: Write the module**

`reference/moog_ladder_oracle.py`:

```python
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
```

- [x] **Step 4: Run to verify the four tests pass**

Run: `.venv/bin/python -m pytest tests/test_oracle.py -v`
Expected: all PASS. If `test_dc_gain_is_one_at_zero_k` fails, the `g` scaling or `k0s` is wrong.
Compare against `reference/upstream/moog_ladder_nonlinear.m` lines 127 to 147.

- [x] **Step 5: Write the linear-limit test, the oracle's main gate**

The oracle must converge to the linear reference as drive vanishes, because the tanh stages
become linear. Append to `tests/test_oracle.py`:

```python
def test_oracle_converges_to_linear_reference_for_tiny_signals():
    """As drive vanishes the tanh stages linearize and the oracle matches Part I.

    Compared in band only, to 0.4*fs, because the two are not expected to agree
    once the analog stopband folds into the digital one. Shapes are compared
    after normalizing each to its own DC value, since the absolute gain of an
    impulse scaled by 1e-5 is arbitrary.
    """
    n_fft = 1 << 14
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)
    imp = np.zeros(n_fft)
    imp[0] = 1e-5

    h_oracle = np.abs(np.fft.rfft(oracle.process(imp, FS, 1000.0, 2.0, N)))
    h_ref = np.abs(mll.frequency_response(N, FS, 1000.0, 2.0, freqs))

    band = (freqs >= 20.0) & (freqs <= 0.4 * FS)
    got = 20.0 * np.log10(h_oracle[band] / h_oracle[0])
    want = 20.0 * np.log10(h_ref[band] / h_ref[0])
    assert np.max(np.abs(got - want)) < 0.5
```

- [x] **Step 6: Run the linear-limit test**

Run: `.venv/bin/python -m pytest tests/test_oracle.py::test_oracle_converges_to_linear_reference_for_tiny_signals -v`
Expected: PASS. If it fails, the global feedback section (`rg`, `qg`, `k0g`) is the suspect: those
terms are what the delay-free method introduces and are the easiest to mistranscribe. Note the
per-stage signs carefully against the Octave, where `sf(n) = r1s(i)*yi - q0s(i)*yo` and `r1s` is
negative.

- [x] **Step 7: Commit**

```bash
git add reference/moog_ladder_oracle.py tests/test_oracle.py
git commit -m "port nonlinear ladder oracle from part two"
```

---

### Task 6: Add opt-in float32 WAV output to RunFilters

Without this, every distortion measurement is floored at the 16-bit noise level near 0.0015 %
THD, which is exactly where the existing suite's ranking went wrong.

**Files:**
- Modify: `example/helpers.hpp` (add `WriteWavFileFloat` after `WriteWavFile`, which ends near
  line 247)
- Modify: `example/run-filters.cpp` (option variable near line 195, help text near line 26,
  argument parsing near line 234, write call near line 339)
- Test: `tests/test_runfilters_float.py`

**Interfaces:**
- Consumes: nothing
- Produces: `inline bool WriteWavFileFloat(const char* filename, int sampleRate, int numChannels,
  const std::vector<float>& samples)`, and a `--float` flag on `RunFilters` that defaults off

- [x] **Step 1: Write the failing tests**

`tests/test_runfilters_float.py`:

```python
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
RUNFILTERS = REPO / "build" / "RunFilters"
needs_build = pytest.mark.skipif(
    not RUNFILTERS.exists(), reason="build/RunFilters not built"
)


def _write_input(path, samples, fs=44100):
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(fs)
    w.writeframes(pcm.tobytes())
    w.close()


def _read_fmt(path):
    """Return (audio_format_tag, channels, sample_rate, bits_per_sample)."""
    with open(path, "rb") as f:
        data = f.read()
    at = data.index(b"fmt ")
    tag, channels, rate, byterate, blockalign, bits = struct.unpack_from(
        "<HHIIHH", data, at + 4
    )
    return tag, channels, rate, bits


def _first_wav(outdir):
    files = sorted(Path(outdir).glob("*.wav"))
    assert files, "no output wav produced"
    return files[0]


def _read_samples(path, bits):
    w = wave.open(str(path), "rb")
    raw = w.readframes(w.getnframes())
    w.close()
    if bits == 32:
        return np.frombuffer(raw, "<f4").astype(np.float64)
    return np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0


def _run(src, outdir, extra=()):
    cmd = [
        str(RUNFILTERS), "-f", str(src), "-c", "1000", "-r", "0.0",
        "-s", "0", "-o", str(outdir),
    ]
    cmd.extend(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


@needs_build
def test_float_flag_writes_audioformat3_32bit(tmp_path):
    src = tmp_path / "in.wav"
    _write_input(src, 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(44100) / 44100.0))
    out = tmp_path / "f32"
    r = _run(src, out, ["--float"])
    assert r.returncode == 0, r.stderr
    tag, ch, rate, bits = _read_fmt(_first_wav(out))
    assert tag == 3
    assert bits == 32
    assert ch == 1
    assert rate == 44100


@needs_build
def test_default_output_is_still_pcm16(tmp_path):
    src = tmp_path / "in.wav"
    _write_input(src, 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(44100) / 44100.0))
    out = tmp_path / "pcm16"
    r = _run(src, out)
    assert r.returncode == 0, r.stderr
    tag, _, _, bits = _read_fmt(_first_wav(out))
    assert tag == 1
    assert bits == 16


@needs_build
@pytest.mark.parametrize("extra,expected_tag", [(["--float"], 3), ([], 1)])
def test_tone_amplitude_survives_float32_but_not_pcm16(tmp_path, extra, expected_tag):
    """A -60 dBFS tone survives float32 and is destroyed by 16-bit.

    This is the whole reason the flag exists. 0.001 amplitude is well below what
    a 16-bit file resolves at this length, so the integer path quantizes it away.
    """
    src = tmp_path / "in.wav"
    _write_input(src, 0.001 * np.sin(2 * np.pi * 1000.0 * np.arange(88200) / 44100.0))
    out = tmp_path / ("t" if extra else "p")
    r = _run(src, out, extra)
    assert r.returncode == 0, r.stderr
    got = _first_wav(out)
    tag, _, _, bits = _read_fmt(got)
    assert tag == expected_tag
    y = _read_samples(got, bits)
    rms = float(np.sqrt(np.mean(y[44100:] ** 2)))
    # A -60 dBFS sine has rms 0.001/sqrt(2).
    assert rms == pytest.approx(0.001 / np.sqrt(2.0), rel=0.05)
```

- [x] **Step 2: Build and run to verify the float tests fail**

```bash
cd /home/sander/projects/github/ddiakopoulos/MoogLadders
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRTAUDIO_DUMMY=ON
cmake --build build 2>&1 | tail -5
.venv/bin/python -m pytest tests/test_runfilters_float.py -v
```

Expected: the `--float` tests FAIL, because `RunFilters` rejects the unknown argument and returns
nonzero. The default PCM16 tests PASS.

- [x] **Step 3: Add WriteWavFileFloat to helpers.hpp**

Insert immediately after `WriteWavFile`. Leave that function byte-for-byte untouched.

```cpp
// Write WAV file (32-bit IEEE float)
inline bool WriteWavFileFloat(const char* filename, int sampleRate, int numChannels, const std::vector<float>& samples) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    auto write32 = [&](uint32_t v) { file.write((char*)&v, 4); };
    auto write16 = [&](uint16_t v) { file.write((char*)&v, 2); };

    uint32_t numSamples = static_cast<uint32_t>(samples.size());
    uint16_t bitsPerSample = 32;
    uint32_t dataSize = numSamples * (bitsPerSample / 8);
    uint32_t fileSize = 36 + dataSize;

    file.write("RIFF", 4);
    write32(fileSize);
    file.write("WAVE", 4);

    file.write("fmt ", 4);
    write32(16);
    write16(3); // audio format (IEEE float)
    write16(static_cast<uint16_t>(numChannels));
    write32(static_cast<uint32_t>(sampleRate));
    write32(static_cast<uint32_t>(sampleRate * numChannels * (bitsPerSample / 8)));
    write16(static_cast<uint16_t>(numChannels * (bitsPerSample / 8)));
    write16(bitsPerSample);

    file.write("data", 4);
    write32(dataSize);
    file.write((const char*)samples.data(), dataSize);

    return file.good();
}
```

`ReadWavFile` at `example/helpers.hpp:201` already handles `audioFormat == 3`, so there is no
reader change to make. Do not add one.

- [x] **Step 4: Add the flag to run-filters.cpp**

Near line 195, beside the other option variables:

```cpp
    bool floatOutput = false;
```

In the argument loop, beside the `--bench` branch near line 234:

```cpp
        else if (arg == "--float") {
            floatOutput = true;
        }
```

In the help text, after the `--bench` block near line 26:

```cpp
    std::cout << "  --float                 Write 32-bit IEEE float WAV output instead of\n";
    std::cout << "                          16-bit PCM. Use when measuring distortion, since\n";
    std::cout << "                          16-bit quantisation floors THD near 0.0015%.\n";
```

At the write site near line 339, branch on the flag and leave the surrounding messages alone:

```cpp
        bool wrote = floatOutput
            ? WriteWavFileFloat(outputFile.c_str(), sampleRate, numChannels, samples)
            : WriteWavFile(outputFile.c_str(), sampleRate, numChannels, samples);
        if (wrote) {
```

- [x] **Step 5: Rebuild and run to verify all four pass**

```bash
cmake --build build 2>&1 | tail -5
.venv/bin/python -m pytest tests/test_runfilters_float.py -v
```

Expected: 4 passed

- [x] **Step 6: Verify the default path is byte-identical, as a test not a manual check**

This is a hard constraint, so make it reproducible. Capture a reference from the pre-change
binary by stashing the C++ edits:

```bash
cd /home/sander/projects/github/ddiakopoulos/MoogLadders
git stash push example/helpers.hpp example/run-filters.cpp
cmake -B build_ref -DCMAKE_BUILD_TYPE=Release -DRTAUDIO_DUMMY=ON >/dev/null
cmake --build build_ref >/dev/null
git stash pop
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRTAUDIO_DUMMY=ON >/dev/null
cmake --build build >/dev/null
.venv/bin/python - <<'EOF'
import hashlib
import pathlib
import subprocess
import wave

import numpy as np

work = pathlib.Path("/tmp/bytecheck")
work.mkdir(exist_ok=True)
src = work / "in.wav"
if not src.exists():
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    w = wave.open(str(src), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(44100)
    w.writeframes(np.clip(x * 32767, -32768, 32767).astype("<i2").tobytes())
    w.close()

def hashes(binary, outdir):
    subprocess.run(
        [binary, "-f", str(src), "-c", "1000", "-r", "0.3", "-s", "0", "-o", str(outdir)],
        check=True, capture_output=True,
    )
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(pathlib.Path(outdir).glob("*.wav"))}

before = hashes("./build_ref/RunFilters", work / "ref")
after = hashes("./build/RunFilters", work / "new")
assert before == after, f"MISMATCH\nbefore={before}\nafter={after}"
print(f"byte-identical across {len(before)} files")
EOF
```

Expected: `byte-identical across 12 files`. If this fails, something other than the flag path
changed. Do not proceed until it matches.

- [x] **Step 7: Commit**

```bash
git add example/helpers.hpp example/run-filters.cpp tests/test_runfilters_float.py
git commit -m "add opt-in float32 wav output to runfilters"
```

---

### Task 7: Signal generation and model invocation

**Files:**
- Create: `scripts/faithfulness_eval.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces, in `scripts/faithfulness_eval.py`:
  - `SAMPLE_RATE = 44100`, `BAND_LIMIT_FRACTION = 0.4`, `ORDER = 4`
  - `FILTER_NAMES: List[str]`, the 12 names in enum order
  - `sine_sweep(n, fs, f0, f1, amplitude) -> np.ndarray`
  - `two_tone(n, fs, f1, f2, amplitude) -> np.ndarray`
  - `step(n, fs, amplitude) -> np.ndarray`
  - `steady_sine(n, fs, freq, amplitude) -> np.ndarray`
  - `write_wav(path, samples, fs=SAMPLE_RATE, int16=False) -> None`
  - `read_wav(path) -> Tuple[np.ndarray, int]`
  - `read_wav_float(path) -> np.ndarray`
  - `run_model(model, signal, cutoff, resonance, oversample, runfilters, workdir)
    -> Optional[np.ndarray]`, returning None when the model produced non-finite output
  - `RUNFILTERS_PATH: Path`, module global defaulted to `Path("build/RunFilters")`

- [x] **Step 1: Write the failing tests**

`tests/test_eval_metrics.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import faithfulness_eval as fe  # noqa: E402


def test_filter_names_match_the_cpp_enum_order():
    assert fe.FILTER_NAMES == [
        "Stilson",
        "Simplified",
        "Huovilainen",
        "Improved",
        "Krajeski",
        "RKSimulation",
        "Microtracker",
        "MusicDSP",
        "OberheimVariation",
        "Hyperion",
        "HyperionTanh",
        "HyperionLegacy",
    ]


def test_sine_sweep_frequency_rises_over_time():
    n = 44100
    x = fe.sine_sweep(n, 44100, 20.0, 20000.0, 0.5)
    assert x.shape == (n,)
    assert np.max(np.abs(x)) <= 0.5 + 1e-9
    crossings = np.where(np.diff(np.sign(x)))[0]
    assert len(crossings) > 10
    half = len(crossings) // 2
    assert np.diff(crossings[half:]).mean() > np.diff(crossings[:half]).mean() * 2.0


def test_step_is_zero_then_constant():
    x = fe.step(1000, 44100, 0.5)
    assert np.all(x[:100] == 0.0)
    assert np.all(x[100:] == 0.5)


def test_two_tone_stays_in_range():
    x = fe.two_tone(44100, 44100, 440.0, 554.0, 0.9)
    assert np.max(np.abs(x)) <= 0.9 + 1e-9


def test_float32_wav_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x = (0.5 * rng.standard_normal(4096)).astype(np.float32)
    p = tmp_path / "x.wav"
    fe.write_wav(p, x)
    y, fs = fe.read_wav(p)
    assert fs == 44100
    assert np.allclose(y.astype(np.float32), x, atol=1e-7)


def test_int16_wav_roundtrip(tmp_path):
    p = tmp_path / "y.wav"
    fe.write_wav(p, np.full(1024, 0.25, dtype=np.float32), int16=True)
    y, fs = fe.read_wav(p)
    assert fs == 44100
    assert np.allclose(y, 0.25, atol=1e-4)
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'faithfulness_eval'`

- [x] **Step 3: Write the module header, constants, generators and WAV IO**

`scripts/faithfulness_eval.py`, starting:

```python
#!/usr/bin/env python
"""Rank the repo's Moog ladder models by faithfulness to the analog filter.

Two independent axes:

- Linear: measured model response against the analytic continuous-time H(s) from
  D'Angelo and Valimaki Part I, ported in reference/moog_ladder_linear.py.
- Nonlinear: measured model output against the Part II circuit model, ported in
  reference/moog_ladder_oracle.py. This axis is model-referenced, NOT
  hardware-referenced. No public Moog I/O dataset was found, so it cannot be
  independently verified here.

The linear score is computed in band only, up to 0.4*fs. Out-of-band response is
reported but not scored, because a digital implementation must fold the analog
stopband back and that is not a modeling error.

Usage:
    python scripts/faithfulness_eval.py --runfilters build/RunFilters
    python scripts/faithfulness_eval.py --runfilters build/RunFilters --filters Stilson
    python scripts/faithfulness_eval.py --runfilters build/RunFilters --no-figures

Output:
    filter_validation/faithfulness/<run_id>/
        wav/       per-model RunFilters output
        metrics/   per-model JSON
        scores.json, ranking.md
    docs/moog-faithfulness/plots/   the six committed figures
"""

import argparse
import json
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 44100
BAND_LIMIT_FRACTION = 0.4
ORDER = 4

FILTER_NAMES = [
    "Stilson",
    "Simplified",
    "Huovilainen",
    "Improved",
    "Krajeski",
    "RKSimulation",
    "Microtracker",
    "MusicDSP",
    "OberheimVariation",
    "Hyperion",
    "HyperionTanh",
    "HyperionLegacy",
]

RUNFILTERS_PATH = Path("build/RunFilters")

_matplotlib_pyplot = None


def _get_pyplot():
    """Lazy import for matplotlib.pyplot, matching filter_verification.py."""
    global _matplotlib_pyplot
    if _matplotlib_pyplot is None:
        import matplotlib.pyplot as plt
        _matplotlib_pyplot = plt
    return _matplotlib_pyplot


def sine_sweep(n, fs, f0, f1, amplitude):
    """Logarithmic sine sweep from f0 to f1 over n samples."""
    t = np.arange(n) / float(fs)
    k = np.log(f1 / f0)
    phase = 2.0 * np.pi * f0 * (np.exp(k * t) - 1.0) / k
    return amplitude * np.sin(phase)


def two_tone(n, fs, f1, f2, amplitude):
    """Two equal-amplitude sines, scaled so the peak stays under amplitude."""
    t = np.arange(n) / float(fs)
    return (amplitude / 2.0) * (
        np.sin(2.0 * np.pi * f1 * t) + np.sin(2.0 * np.pi * f2 * t)
    )


def step(n, fs, amplitude):
    """Unit step delayed by 100 samples."""
    x = np.zeros(n)
    x[100:] = amplitude
    return x


def steady_sine(n, fs, freq, amplitude):
    t = np.arange(n) / float(fs)
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def write_wav(path, samples, fs=SAMPLE_RATE, int16=False):
    """Write a mono WAV, float32 by default or 16-bit PCM when int16 is set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float32)
    width = 2 if int16 else 4
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(width)
    w.setframerate(fs)
    if int16:
        pcm = np.clip(x * 32767.0, -32768.0, 32767.0).astype("<i2")
        w.writeframes(pcm.tobytes())
    else:
        w.writeframes(x.astype("<f4").tobytes())
    w.close()


def read_wav(path):
    """Read a mono WAV into float64. Handles 8, 16, 24 and 32 bit integer."""
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        width = w.getsampwidth()
        fs = w.getframerate()
        raw = w.readframes(n)
    if width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0, fs
    if width == 2:
        return np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0, fs
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        return v.astype(np.float64) / 8388608.0, fs
    if width == 4:
        return np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0, fs
    raise ValueError(f"unsupported sample width {width}")


def read_wav_float(path):
    """Read a mono float32 WAV, as written by `RunFilters --float`."""
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, "<f4").astype(np.float64)


def run_model(model, signal, cutoff, resonance, oversample, runfilters, workdir):
    """Drive one model with signal via RunFilters --float.

    RunFilters writes one WAV per model, named <FilterName>_c<cutoff>_r<resonance>
    (_os<n>x when oversampling). This function picks the file whose name starts
    with `model`. Returns the output as float64, or None if the run failed,
    returns None if the run failed, produced no file for this model, or produced
    any non-finite sample.
    """
    if cutoff <= 0.0 or cutoff >= 0.5 * SAMPLE_RATE:
        raise ValueError(f"cutoff {cutoff} must be in (0, fs/2)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = f"{model}_c{int(cutoff)}_r{resonance:.2f}"
    src = workdir / f"{tag}_in.wav"
    outdir = workdir / f"{tag}_out"
    write_wav(src, signal)

    result = subprocess.run(
        [
            str(runfilters),
            "-f", str(src),
            "-c", str(int(cutoff)),
            "-r", f"{resonance:.2f}",
            "-s", str(oversample),
            "-o", str(outdir),
            "--float",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    pattern = f"*_c{int(cutoff)}_r{resonance:.2f}"
    if oversample:
        pattern += f"_os{oversample}x"
    pattern += ".wav"
    candidates = list(outdir.glob(pattern))
    match = [p for p in candidates if p.stem.startswith(model + "_")]
    if not match:
        return None
    y = read_wav_float(match[0])
    if not np.all(np.isfinite(y)):
        return None
    return y
```

- [x] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: PASS, 6 tests

- [x] **Step 5: Commit**

```bash
git add scripts/faithfulness_eval.py tests/test_eval_metrics.py
git commit -m "add signal generation and model invocation for evaluation"
```

---

### Task 8: Self-validate the measurement code before trusting it

If the measurement code cannot recover a known number, no ranking is credible. This is the gate
from the design doc.

**Files:**
- Modify: `scripts/faithfulness_eval.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `write_wav`, `read_wav` from Task 7
- Produces, in `scripts/faithfulness_eval.py`:
  - `spectrum_db(x, fs) -> Tuple[np.ndarray, np.ndarray]`
  - `fundamental_freq(x, fs) -> float`
  - `goertzel_amplitude(x, fs, freq) -> float`
  - `harmonic_ratio_db(x, fs, order) -> float`
  - `thd_percent(x, fs, orders=10) -> float`
  - `noise_floor_db(x, fs) -> float`

- [x] **Step 1: Write the failing self-validation tests**

Append to `tests/test_eval_metrics.py`:

```python
def test_thd_recovers_a_known_second_harmonic():
    """Fundamental plus a -20 dB H2 must read exactly 10.0 % THD.

    THD is the ratio of summed harmonic power to fundamental power, so a single
    H2 at amplitude ratio 0.1 gives sqrt(0.01) = 10.0 %.
    """
    fs = 44100
    t = np.arange(88200) / fs
    x = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 2000 * t)
    assert fe.thd_percent(x, fs) == pytest.approx(10.0, rel=1e-3)


def test_thd_matches_hand_computed_harmonics():
    fs = 44100
    t = np.arange(88200) / fs
    x = (
        np.sin(2 * np.pi * 1000 * t)
        + 0.1 * np.sin(2 * np.pi * 2000 * t)
        + 0.05 * np.sin(2 * np.pi * 3000 * t)
    )
    expected = np.sqrt(0.1**2 + 0.05**2) * 100.0
    assert fe.thd_percent(x, fs) == pytest.approx(expected, rel=1e-3)


def test_thd_of_pure_sine_is_negligible():
    fs = 44100
    t = np.arange(88200) / fs
    assert fe.thd_percent(np.sin(2 * np.pi * 1000 * t), fs) < 0.01


def test_fundamental_freq_finds_the_tone():
    fs = 44100
    t = np.arange(88200) / fs
    assert fe.fundamental_freq(np.sin(2 * np.pi * 1234 * t), fs) == pytest.approx(
        1234.0, rel=1e-3
    )


def test_noise_floor_of_silence_is_as_low_as_possible():
    assert fe.noise_floor_db(np.zeros(16384), 44100) < -300.0


def test_thd_below_16bit_floor_survives_float32_but_not_pcm16(tmp_path):
    """The exact bug that broke the old suite: a -60 dBFS tone must still show THD.

    Round-tripping through 16-bit quantizes this away. Through float32 it
    survives. This is the direct guard on why --float exists.
    """
    fs = 44100
    t = np.arange(88200) / fs
    x = 0.001 * (
        np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 2000 * t)
    )
    p32 = tmp_path / "a32.wav"
    fe.write_wav(p32, x)
    back32, _ = fe.read_wav(p32)
    p16 = tmp_path / "a16.wav"
    fe.write_wav(p16, x, int16=True)
    back16, _ = fe.read_wav(p16)

    assert fe.thd_percent(back32, fs) == pytest.approx(10.0, rel=0.02)
    assert fe.thd_percent(back16, fs) < 5.0
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `AttributeError: module has no attribute 'thd_percent'`

- [x] **Step 3: Implement the measurement functions**

Append to `scripts/faithfulness_eval.py`:

```python
def spectrum_db(x, fs):
    """Return (freqs_hz, magnitude_db) from a Hann-windowed real FFT."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        raise ValueError("signal too short for spectral analysis")
    win = np.hanning(x.size)
    mag = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    return freqs, 20.0 * np.log10(np.maximum(mag, 1e-30))


def noise_floor_db(x, fs):
    """Median magnitude in dB of the Hann-windowed spectrum."""
    _, db = spectrum_db(x, fs)
    return float(np.median(db))


def fundamental_freq(x, fs):
    """Frequency of the largest spectral peak below 1 kHz, in Hz."""
    freqs, db = spectrum_db(x, fs)
    limit = int(np.searchsorted(freqs, 1000.0))
    if limit < 2:
        raise ValueError("no fundamental below 1 kHz")
    return float(freqs[1:limit][int(np.argmax(db[1:limit]))])


def goertzel_amplitude(x, fs, freq):
    """Amplitude of `freq` in x, by Goertzel, normalized so a unit sine reads 1.

    Projects at the exact requested frequency rather than reading the nearest FFT
    bin, so a signal that does not contain a whole number of periods does not
    leak into the reading.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    k = 2.0 * np.pi * freq / fs
    coeff = 2.0 * np.cos(k)
    s1 = 0.0
    s2 = 0.0
    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0
    real = s1 - s2 * np.cos(k)
    imag = s2 * np.sin(k)
    return float(2.0 * np.hypot(real, imag) / n)


def harmonic_ratio_db(x, fs, order):
    """Amplitude of harmonic `order` relative to the fundamental, in dB."""
    f0 = fundamental_freq(x, fs)
    fund = goertzel_amplitude(x, fs, f0)
    harm = goertzel_amplitude(x, fs, order * f0)
    return 20.0 * np.log10(max(harm / max(fund, 1e-30), 1e-30))


def thd_percent(x, fs, orders=10):
    """Total harmonic distortion in percent, harmonics 2 through `orders`."""
    f0 = fundamental_freq(x, fs)
    fund = goertzel_amplitude(x, fs, f0)
    if fund <= 0.0:
        return 0.0
    power = 0.0
    for order in range(2, orders + 1):
        amp = goertzel_amplitude(x, fs, order * f0)
        power += amp * amp
    return 100.0 * np.sqrt(power) / fund
```

The Goertzel loops are O(n) in Python per harmonic and are the slow part of the suite. If a test
times out, replace the inner loop with a `np.dot` against precomputed cosine and sine tables,
keeping the mathematics identical and leaving the tests unchanged so the swap is verified rather
than assumed.

- [x] **Step 4: Run to verify the self-validation tests pass**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: all PASS. Pay attention to
`test_thd_below_16bit_floor_survives_float32_but_not_pcm16`: the float32 branch must read about
10 % and the 16-bit branch under 5 %. If both read 10 %, the 16-bit write is not quantizing and
the test is not testing what it claims.
### Task 9: Reference accessor and resonance calibration

Each model's `SetResonance(r)` maps `r` in [0,1] to an internal feedback gain `k` differently.
Comparing a model at `r=0.5` against a reference at `k=2` would double-penalize a
mis-calibrated control: once for the wrong shape and once for the wrong tuning. So the linear
axis calibrates each control first, then compares shape at the calibrated point.

**Files:**
- Modify: `scripts/faithfulness_eval.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `mll.magnitude_db`, `run_model` from Task 7
- Produces, in `scripts/faithfulness_eval.py`:
  - `reference_magnitude_db(fc: float, k: float, freqs: np.ndarray) -> np.ndarray`
  - `measured_magnitude_db(y: np.ndarray, freqs: np.ndarray) -> np.ndarray`
  - `best_r_for_target(responses: List[np.ndarray], resonances: List[float], reference_db) -> Tuple[float, float, int]`
  - `calibrate_resonance(model, runfilters, workdir, target_k=2.0, fc=1000.0, n=N) -> Tuple[Optional[float], Dict[str, Any]]`

- [x] **Step 1: Write the failing tests**

Append to `tests/test_eval_metrics.py`:

```python
def test_reference_magnitude_is_zero_db_at_dc_for_zero_k():
    d = fe.reference_magnitude_db(1000.0, 0.0, np.array([1e-3, 1.0, 10.0]))
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert d[2] == pytest.approx(0.0, abs=1e-3)


def test_reference_matches_reference_module():
    from reference import moog_ladder_linear as mll
    freqs = np.logspace(1, np.log10(0.4 * 44100), 500)
    assert np.allclose(
        fe.reference_magnitude_db(2000.0, 2.0, freqs),
        mll.magnitude_db(4, 44100.0, 2000.0, 2.0, freqs),
    )


def test_reference_peak_gain_rises_with_k():
    freqs = np.logspace(1, np.log10(0.4 * 44100), 4000)
    peaks = [fe.reference_magnitude_db(1000.0, k, freqs).max() for k in (0.0, 1.0, 2.0, 3.0)]
    assert all(b > a for a, b in zip(peaks, peaks[1:]))


def test_best_r_for_target_finds_injected_optimum():
    target = np.zeros(100)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    curves = [target + 6.0, target, target + 3.0, target + 9.0, target + 12.0]
    r, err, idx = fe.best_r_for_target(curves, grid, target)
    assert idx == 1
    assert r == 0.5
    assert err == pytest.approx(0.0, abs=1e-9)


def test_measured_magnitude_interpolates_onto_log_freqs():
    n = 8192
    freqs = np.logspace(1, np.log10(0.4 * 44100), 200)
    y = np.zeros(n)
    y[0] = 1.0
    d = fe.measured_magnitude_db(y, freqs)
    assert d.shape == freqs.shape
    assert np.all(np.isfinite(d))
    assert d[0] == pytest.approx(-60.0, abs=1.0)
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `AttributeError: module has no attribute 'reference_magnitude_db'`

- [x] **Step 3: Implement the accessors**

Append to `scripts/faithfulness_eval.py`:

```python
def reference_magnitude_db(fc, k, freqs):
    """Analytic reference magnitude in dB at freqs, in Hz.

    fc is the leading-pole cutoff, NOT the -3 dB point of the cascade. See
    reference/moog_ladder_linear.py and
    tests/test_linear_reference.py::test_fc_to_minus3db_ratio_is_pinned.
    """
    from reference import moog_ladder_linear as mll
    return mll.magnitude_db(ORDER, SAMPLE_RATE, fc, k, np.asarray(freqs, dtype=float))


def measured_magnitude_db(y, freqs):
    """Magnitude in dB of an impulse response, interpolated onto freqs in Hz.

    The impulse response is not windowed. The existing suite windowed it with a
    Hann window whose first sample is zero, and since the first sample is the
    entire excitation, the measurement read the window, not the filter.
    """
    freqs = np.asarray(freqs, dtype=float)
    h = np.abs(np.fft.rfft(np.asarray(y, dtype=np.float64)))
    f = np.fft.rfftfreq(len(y), 1.0 / SAMPLE_RATE)
    return np.interp(freqs, f, 20.0 * np.log10(np.maximum(h, 1e-30)))


def best_r_for_target(responses, resonances, reference_db):
    """Pick the resonance whose measured curve best matches the reference.

    responses: list of magnitude-dB arrays, one per entry of resonances.
    Returns (best_r, rms_error_db, index).
    """
    best = None
    for i, (r, curve) in enumerate(zip(responses, resonances)):
        err = float(np.sqrt(np.mean((np.asarray(curve) - reference_db) ** 2)))
        if best is None or err < best[1]:
            best = (float(r), err, i)
    return best
```

- [x] **Step 4: Run to verify the accessor tests pass**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: PASS

- [x] **Step 5: Implement calibrate_resonance, which drives the binary**

Append to `scripts/faithfulness_eval.py`:

```python
def calibrate_resonance(model, runfilters, workdir, target_k=2.0, fc=1000.0, n=32768):
    """Find the user-facing resonance that best matches the reference at target_k.

    Sweeps r in [0,1] in 0.05 steps, drives the model with a unit impulse at
    fc=subject fc, and keeps the r minimizing RMS dB error against the
    reference. Returns (best_r, info). best_r is None when the model produced
    no finite output at any resonance.
    """
    freqs = np.logspace(np.log10(20.0), np.log10(BAND_LIMIT_FRACTION * SAMPLE_RATE), 400)
    ref = reference_magnitude_db(fc, target_k, freqs)
    impulse = np.zeros(n)
    impulse[0] = 1.0

    resonances = [i / 20.0 for i in range(21)]
    curves, used = [], []
    for r in resonances:
        y = run_model(model, impulse, fc, r, 0, runfilters, workdir)
        if y is None:
            continue
        curves.append(measured_magnitude_db(y, freqs))
        used.append(r)

    if not curves:
        return None, {"reason": "no finite output at any resonance"}

    best_r, err, _ = best_r_for_target(curves, used, ref)
    return best_r, {"rms_error_db": err, "n_points": len(used), "target_k": target_k}
```

- [x] **Step 6: Commit**

```bash
git add scripts/faithfulness_eval.py tests/test_eval_metrics.py
git commit -m "add resonance calibration against the linear reference"
```

---

### Task 10: Shape metrics and the linear score

**Files:**
- Modify: `scripts/faithfulness_eval.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `reference_magnitude_db` from Task 9, `normalize_error` defined below
- Produces, in `scripts/faithfulness_eval.py`:
  - `_cross_freq(freqs, db, target_db) -> float`
  - `shape_metrics(measured_db, reference_db, freqs) -> Dict[str, float]`
  - `LINEAR_WEIGHTS: Dict[str, Tuple[float, float, float]]`
  - `normalize_error(value, best, worst) -> float`
  - `score_linear(metrics) -> float`

- [x] **Step 1: Write the failing tests**

Append to `tests/test_eval_metrics.py`:

```python
def _flat_reference():
    freqs = np.logspace(1, np.log10(0.4 * 44100), 2000)
    return freqs, np.zeros_like(freqs)


def test_shape_metrics_perfect_match_is_all_zero():
    freqs, ref = _flat_reference()
    m = fe.shape_metrics(ref.copy(), ref, freqs)
    assert m["magnitude_rms_db_error"] == pytest.approx(0.0, abs=1e-9)
    assert m["cutoff_3db_error_cents"] == pytest.approx(0.0, abs=1e-6)
    assert m["passband_gain_error_db"] == pytest.approx(0.0, abs=1e-9)
    assert m["stopband_slope_error_db_per_oct"] == pytest.approx(0.0, abs=1e-6)
    assert m["peak_gain_error_db"] == pytest.approx(0.0, abs=1e-9)
    assert m["peak_freq_error_cents"] == pytest.approx(0.0, abs=1e-6)


def test_shape_metrics_detect_1db_passband_lift():
    freqs, ref = _flat_reference()
    m = fe.shape_metrics(ref + 1.0, ref, freqs)
    assert m["passband_gain_error_db"] == pytest.approx(1.0, abs=0.05)
    assert m["magnitude_rms_db_error"] > 0.0


def test_shape_metrics_detect_2pole_disguised_as_4pole():
    freqs = np.logspace(2, np.log10(0.4 * 44100), 3000)
    ref = -24.0 * np.log2(freqs / 1000.0)
    meas = -12.0 * np.log2(freqs / 1000.0)
    m = fe.shape_metrics(meas, ref, freqs)
    assert m["stopband_slope_error_db_per_oct"] == pytest.approx(12.0, rel=0.1)


def test_linear_score_bounded_and_orders_correctly():
    perfect = {
        "magnitude_rms_db_error": 0.0, "cutoff_3db_error_cents": 0.0,
        "passband_gain_error_db": 0.0, "stopband_slope_error_db_per_oct": 0.0,
        "peak_gain_error_db": 0.0, "peak_freq_error_cents": 0.0,
    }
    bad = {
        "magnitude_rms_db_error": 6.0, "cutoff_3db_error_cents": 400.0,
        "passband_gain_error_db": 3.0, "stopband_slope_error_db_per_oct": 12.0,
        "peak_gain_error_db": 4.0, "peak_freq_error_cents": 200.0,
    }
    assert fe.score_linear(perfect) == pytest.approx(100.0, abs=1e-6)
    assert 0.0 <= fe.score_linear(bad) <= 100.0
    assert fe.score_linear(perfect) > fe.score_linear(bad)


def test_linear_score_handles_nonfinite_metric():
    metrics = {
        "magnitude_rms_db_error": float("nan"), "cutoff_3db_error_cents": 50.0,
        "passband_gain_error_db": 1.0, "stopband_slope_error_db_per_oct": 4.0,
        "peak_gain_error_db": 2.0, "peak_freq_error_cents": 100.0,
    }
    assert 0.0 <= fe.score_linear(metrics) <= 100.0
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `AttributeError: module has no attribute 'shape_metrics'`

- [x] **Step 3: Implement shape_metrics**

Append to `scripts/faithfulness_eval.py`:

```python
def _cross_freq(freqs, db, target_db):
    """Frequency where a monotonic curve crosses target_db, linearly interpolated."""
    diff = np.asarray(db) - target_db
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    d0, d1 = diff[i], diff[i + 1]
    frac = 0.0 if d1 == d0 else float(d0) / (d0 - d1)
    return float(freqs[i] + frac * (freqs[i + 1] - freqs[i]))


def shape_metrics(measured_db, reference_db, freqs):
    """Compare one magnitude curve against the reference.

    Returns magnitude_rms_db_error, cutoff_3db_error_hz, cutoff_3db_error_cents,
    passband_gain_error_db, stopband_slope_db_per_oct,
    stopband_slope_error_db_per_oct, peak_gain_error_db, peak_freq_error_cents.
    """
    measured_db = np.asarray(measured_db, dtype=float)
    reference_db = np.asarray(reference_db, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    rms = float(np.sqrt(np.mean((measured_db - reference_db) ** 2)))

    f3m = _cross_freq(freqs, measured_db, -3.0)
    f3r = _cross_freq(freqs, reference_db, -3.0)
    cents = (
        1200.0 * np.log2(f3m / f3r)
        if (np.isfinite(f3m) and np.isfinite(f3r) and f3m > 0 and f3r > 0)
        else float("nan")
    )

    pb = freqs <= f3r * 0.5
    passband = (
        float(np.median(measured_db[pb]) - np.median(reference_db[pb])) if pb.any()
        else float("nan")
    )

    sb = freqs >= max(f3r, 1.0) * 4.0
    if sb.sum() >= 3:
        logf = np.log2(freqs[sb])
        slope_m = float(np.polyfit(logf, measured_db[sb], 1)[0])
        slope_r = float(np.polyfit(logf, reference_db[sb], 1)[0])
    else:
        slope_m = slope_r = float("nan")

    peak_m = int(np.argmax(measured_db))
    peak_r = int(np.argmax(reference_db))
    peak_freq_cents = float(1200.0 * np.log2(freqs[peak_m] / freqs[peak_r]))

    return {
        "magnitude_rms_db_error": rms,
        "cutoff_3db_error_hz": float(f3m - f3r),
        "cutoff_3db_error_cents": float(cents),
        "passband_gain_error_db": passband,
        "stopband_slope_db_per_oct": float(slope_m),
        "stopband_slope_error_db_per_oct": float(abs(slope_m - slope_r)),
        "peak_gain_error_db": float(measured_db[peak_m] - reference_db[peak_r]),
        "peak_freq_error_cents": peak_freq_cents,
    }
```

- [x] **Step 4: Implement the weights and score_linear**

Append to `scripts/faithfulness_eval.py`:

```python
# metric name -> (best, worst, weight). Best=0 error maps to 1.0 in normalize_error.
LINEAR_WEIGHTS = {
    "magnitude_rms_db_error": (0.0, 6.0, 0.30),
    "cutoff_3db_error_cents": (0.0, 400.0, 0.20),
    "passband_gain_error_db": (0.0, 3.0, 0.15),
    "stopband_slope_error_db_per_oct": (0.0, 12.0, 0.15),
    "peak_gain_error_db": (0.0, 6.0, 0.12),
    "peak_freq_error_cents": (0.0, 400.0, 0.08),
}


def normalize_error(value, best, worst):
    """Map a raw error metric to 0..1 where best maps to 1.0 and worst to 0.0.

    Non-finite values score 0.0 (a failed measurement is a failure, not a pass).
    """
    if not np.isfinite(value):
        return 0.0
    if value <= best:
        return 1.0
    if value >= worst:
        return 0.0
    return 1.0 - (value - best) / (worst - best)


def score_linear(metrics, weights=LINEAR_WEIGHTS):
    """Weighted 0..100 linear score. 100 means indistinguishable from the reference."""
    total = 0.0
    wsum = 0.0
    for key, (best, worst, weight) in weights.items():
        if key not in metrics:
            continue
        total += weight * normalize_error(metrics[key], best, worst)
        wsum += weight
    return 100.0 * total / wsum if wsum > 0.0 else 0.0
```

- [x] **Step 5: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: all PASS. The 2-pole test is what justifies `stopband_slope_error_db_per_oct`.

- [x] **Step 6: Commit**

```bash
git add scripts/faithfulness_eval.py tests/test_eval_metrics.py
git commit -m "add shape metrics and linear scoring"
```

---

### Task 11: Nonlinear metrics against the oracle

**Files:**
- Modify: `scripts/faithfulness_eval.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `fundamental_freq`, `goertzel_amplitude` from Task 8
- Produces, in `scripts/faithfulness_eval.py`:
  - `align_signals(model, oracle_y, fs, max_lag=256) -> Tuple[np.ndarray, np.ndarray]`
  - `time_domain_nrmse(a, b) -> float`
  - `spectral_distance_db(a, b, fs, f_min=20.0, f_max=None) -> float`
  - `harmonic_profile_db(x, fs, orders=10) -> np.ndarray`
  - `nonlinear_metrics(model, oracle_y, fs) -> Dict[str, float]`
  - `NONLINEAR_WEIGHTS`, `score_nonlinear(metrics) -> float`

- [x] **Step 1: Write the failing tests**

Append to `tests/test_eval_metrics.py`:

```python
def test_align_signals_removes_constant_delay():
    fs = 44100
    t = np.arange(44100) / fs
    ref = np.sin(2 * np.pi * 1000 * t)
    delayed = np.concatenate([np.zeros(64), ref[:-64]])
    a, b = fe.align_signals(delayed, ref, fs)
    assert fe.time_domain_nrmse(a, b) < 1e-9


def test_align_signals_removes_constant_gain_error():
    fs = 44100
    t = np.arange(44100) / fs
    ref = np.sin(2 * np.pi * 1000 * t)
    a, b = fe.align_signals(ref * 2.5, ref, fs)
    assert fe.time_domain_nrmse(a, b) < 1e-9


def test_align_signals_preserves_real_differences():
    """Alignment must not be able to erase a genuine spectral difference."""
    fs = 44100
    t = np.arange(44100) / fs
    ref = np.sin(2 * np.pi * 1000 * t)
    other = ref + 0.2 * np.sin(2 * np.pi * 3000 * t)
    a, b = fe.align_signals(other, ref, fs)
    assert fe.time_domain_nrmse(a, b) > 0.01


def test_spectral_distance_identical_is_zero():
    fs = 44100
    t = np.arange(44100) / fs
    x = np.sin(2 * np.pi * 1000 * t)
    assert fe.spectral_distance_db(x, x, fs) == pytest.approx(0.0, abs=1e-6)


def test_spectral_distance_grows_with_added_harmonic():
    fs = 44100
    t = np.arange(44100) / fs
    a = np.sin(2 * np.pi * 1000 * t)
    small = a + 0.1 * np.sin(2 * np.pi * 2000 * t)
    large = a + 0.5 * np.sin(2 * np.pi * 2000 * t)
    ds, dl = fe.spectral_distance_db(a, small, fs), fe.spectral_distance_db(a, large, fs)
    assert 0.0 < ds < dl


def test_harmonic_profile_is_relative_to_fundamental():
    fs = 44100
    t = np.arange(88200) / fs
    x = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)
    prof = fe.harmonic_profile_db(x, fs)
    assert prof[0] == pytest.approx(0.0, abs=0.5)
    assert prof[2] == pytest.approx(-20.0, abs=0.5)


def test_nonlinear_score_bounded_and_orders_correctly():
    near = {"spectral_distance_db": 0.5, "time_domain_nrmse": 0.01,
            "thd_delta_db": 0.5, "harmonic_profile_corr": 0.99}
    far = {"spectral_distance_db": 12.0, "time_domain_nrmse": 0.9,
           "thd_delta_db": 18.0, "harmonic_profile_corr": 0.2}
    assert fe.score_nonlinear(near) > fe.score_nonlinear(far)
    assert 0.0 <= fe.score_nonlinear(far) <= 100.0
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `AttributeError: module has no attribute 'align_signals'`

- [x] **Step 3: Implement alignment and distances**

Append to `scripts/faithfulness_eval.py`:

```python
def align_signals(model, oracle_y, fs, max_lag=256):
    """Remove DC offset, bulk delay, and a single constant gain error from model.

    Cross-correlates the two signals, shifts by the integer lag maximizing
    correlation, then rescales so the least-squares gain is 1. Returns both
    signals trimmed to a common length.
    """
    model = np.asarray(model, dtype=np.float64)
    oracle_y = np.asarray(oracle_y, dtype=np.float64)
    model = model - model.mean()
    oracle_y = oracle_y - oracle_y.mean()

    n = min(model.size, oracle_y.size)
    model = model[:n]
    oracle_y = oracle_y[:n]

    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(
        np.fft.rfft(model, size) * np.conj(np.fft.rfft(oracle_y, size)), size
    )
    window = np.concatenate((corr[-(max_lag - 1):], corr[: max_lag + 1]))
    lag = int(np.argmax(window)) - max_lag

    if lag > 0:
        model = model[lag:]
        oracle_y = oracle_y[: model.size]
    elif lag < 0:
        oracle_y = oracle_y[-lag:]
        model = model[: oracle_y.size]

    denom = float(np.dot(oracle_y, oracle_y))
    if denom > 0.0:
        model = model / (float(np.dot(model, oracle_y)) / denom)
    return model, oracle_y


def time_domain_nrmse(a, b):
    """Root mean square error normalized by the reference RMS."""
    n = min(len(a), len(b))
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    rms = float(np.sqrt(np.mean(b * b)))
    return float(np.sqrt(np.mean((a - b) ** 2)) / rms) if rms > 0.0 else float("inf")


def spectral_distance_db(a, b, fs, f_min=20.0, f_max=None):
    """RMS log-magnitude difference between two signals, dB, band-limited."""
    if f_max is None:
        f_max = BAND_LIMIT_FRACTION * fs
    n = min(len(a), len(b))
    win = np.hanning(n)
    fa = np.abs(np.fft.rfft(np.asarray(a[:n], dtype=float) * win))
    fb = np.abs(np.fft.rfft(np.asarray(b[:n], dtype=float) * win))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= f_min) & (f <= f_max)
    floor = max(float(np.max(fb[band])) * 1e-6, 1e-30)
    da = 20.0 * np.log10(np.maximum(fa[band], floor))
    db_ = 20.0 * np.log10(np.maximum(fb[band], floor))
    return float(np.sqrt(np.mean((da - db_) ** 2)))


def harmonic_profile_db(x, fs, orders=10):
    """Amplitude of harmonics 1..orders relative to the fundamental, in dB."""
    f0 = fundamental_freq(x, fs)
    return np.array([
        20.0 * np.log10(max(goertzel_amplitude(x, fs, h * f0) /
                            max(goertzel_amplitude(x, fs, f0), 1e-30), 1e-30))
        for h in range(1, orders + 1)
    ])
```

- [x] **Step 4: Implement nonlinear_metrics and score_nonlinear**

Append to `scripts/faithfulness_eval.py`:

```python
def nonlinear_metrics(model, oracle_y, fs):
    """Model-vs-oracle metrics on one aligned test case."""
    a, b = align_signals(model, oracle_y, fs)
    thd_m, thd_o = thd_percent(a, fs), thd_percent(b, fs)
    prof_m, prof_o = harmonic_profile_db(a, fs), harmonic_profile_db(b, fs)
    corr = float(np.corrcoef(prof_m, prof_o)[0, 1]) if len(prof_m) > 1 else 0.0
    return {
        "spectral_distance_db": spectral_distance_db(a, b, fs),
        "time_domain_nrmse": time_domain_nrmse(a, b),
        "thd_delta_db": abs(20.0 * np.log10(max(thd_m, 1e-9) / max(thd_o, 1e-9))),
        "harmonic_profile_corr": corr if np.isfinite(corr) else 0.0,
        "thd_model_percent": thd_m,
        "thd_oracle_percent": thd_o,
        "harmonic_profile_db_model": prof_m.tolist(),
        "harmonic_profile_db_oracle": prof_o.tolist(),
    }


# metric name -> (best, worst, weight). For harmonic_profile_corr best=1.0.
NONLINEAR_WEIGHTS = {
    "spectral_distance_db": (0.0, 12.0, 0.35),
    "time_domain_nrmse": (0.0, 0.9, 0.25),
    "thd_delta_db": (0.0, 18.0, 0.25),
    "harmonic_profile_corr": (0.0, 1.0, 0.15),
}


def score_nonlinear(metrics):
    """Weighted 0..100 nonlinear score against the oracle."""
    total = 0.0
    wsum = 0.0
    for key, (best, worst, weight) in NONLINEAR_WEIGHTS.items():
        if key not in metrics:
            continue
        if key == "harmonic_profile_corr":
            part = normalize_error(metrics[key], worst, best)
        else:
            part = normalize_error(metrics[key], best, worst)
        total += weight * part
        wsum += weight
    return 100.0 * total / wsum if wsum > 0.0 else 0.0
```

- [x] **Step 5: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: all PASS

- [x] **Step 6: Commit**

```bash
git add scripts/faithfulness_eval.py tests/test_eval_metrics.py
git commit -m "add nonlinear metrics against the part two oracle"
```

---

### Task 12: Output collectors, robustness gate, and the driver

**Files:**
- Modify: `scripts/faithfulness_eval.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `run_model`, `score_linear`, `calibrate_resonance`,
  `nonlinear_metrics`, `score_nonlinear` from Tasks 7, 9, 10, 11
- Produces, in `scripts/faithfulness_eval.py`:
  - config tuples: `LEVELS_DBFS`, `SWEEP_N`, `IMPULSE_N`,
    `DEFAULT_CUTOFFS`, `DEFAULT_OVERSAMPLES`, `SELFOSC_RESONANCES`,
    `MODEL_NAMES`, `RUNFILTERS_EXE`, `COMBINED_WEIGHT_LINEAR`
  - `collect_linear(model, runfilters, workdir) -> Dict[str, Any]`
  - `collect_nonlinear(model, runfilters, workdir, oracle, r) -> Dict[str, Any]`
  - `collect_selfoscillation(model, runfilters, workdir) -> Dict[str, Any]`
  - `main(argv=None) -> int`

- [x] **Step 1: Write the failing tests**

Append to `tests/test_eval_metrics.py`:

```python
def test_run_model_rejects_fc_beyond_half_sample_rate(tmp_path):
    with pytest.raises(ValueError):
        fe.run_model("Dummy", np.zeros(64), 30000.0, 0.0, 0, "ignored-binary", str(tmp_path))
```
```

- For the robustness gate, add a fixture in `tests/test_eval_metrics.py` at module level:

```python
def _stub_runfilters(tmp_path):
    """Create a fake runfilters binary that writes one valid mono WAV per call.

    This keeps orchestrator tests from depending on the real C++ binary being
    built. It echoes -c/-r/-s back into the filename exactly like RunFilters
    (from example/run-filters.cpp), so `run_model` finds and reads a zero sample.
    """
    py = (
        "import sys, wave\n"
        "from pathlib import Path\n"
        "r = list(sys.argv)\n"
        "def val(flag):\n"
        "    return r[r.index(flag) + 1] if flag in r else ''\n"
        "outdir = Path(val('-o')); fc = val('-c'); reso = val('-r'); os_ = val('-s')\n"
        "outdir.mkdir(exist_ok=True)\n"
        "name = f'Dummy_c{int(float(fc))}_r{float(reso):.2f}'\n"
        "if int(os_): name += f'_os{int(os_)}x'\n"
        "wf = wave.open(str(outdir / (name + '.wav')), 'wb')\n"
        "wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100)\n"
        "wf.writeframes(b'\\x00' * 8192); wf.close()\n"
    )
    script = tmp_path / "runfilters_stub.py"
    script.write_text(py)
    return str(script)
```

This fixture will be replaced at Task 14 by `str(runfilters_binary)` once the real
binary exists. For now it lets the collector tests run without a build.

Add a test validating the collected per-model JSON schema:

```python
def test_collect_linear_returns_key_schema(tmp_path):
    result = fe.collect_linear("Dummy", _stub_runfilters(tmp_path), str(tmp_path))
    for key in ("requested", "measured", "linear_score", "score_parts"):
        assert key in result, key


def test_collect_linear_excludes_unstable_resonance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fe,
        "calibrate_resonance",
        lambda *a, **k: (None, {"reason": "no finite output at any resonance"}),
    )
    result = fe.collect_linear("Dummy", _stub_runfilters(tmp_path), str(tmp_path))
    assert result["linear_score"] is None
    assert "reason" in result
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `AttributeError: module has no attribute 'collect_linear'`

- [x] **Step 3: Implement the collectors**

Append to `scripts/faithfulness_eval.py`:

```python
LEVELS_DBFS = (-24.0, -18.0, -12.0, -6.0, -3.0)
SWEEP_N = 131072
IMPULSE_N = 32768
DEFAULT_CUTOFFS = (100.0, 1000.0, 5000.0)
DEFAULT_OVERSAMPLES = (0, 4)
SELFOSC_RESONANCES = (0.5, 0.9, 1.0)
COMBINED_WEIGHT_LINEAR = 0.5

# Order must match FilterModelNames in example/helpers.hpp; rankings sort by enum.
MODEL_NAMES = (
    "Stilson", "Simplified", "Huovilainen", "Improved", "Krajeski",
    "RKSimulation", "Microtracker", "MusicDSP", "OberheimVariation",
    "Hyperion", "HyperionTanh", "HyperionLegacy",
)

RUNFILTERS_EXE = os.environ.get(
    "RUNFILTERS_EXE",
    str(Path("build") / ("RunFilters" + (".exe" if os.name == "nt" else ""))),
)
```

```python
def collect_linear(model, runfilters, workdir):
    """Per-model linear set: calibrate resonance, sweep cutoffs, score shape."""
    r, cal = calibrate_resonance(model, runfilters, workdir)
    out = {"requested": {}, "measured": {}, "linear_score": None, "score_parts": None}
    if r is None:
        out["reason"] = cal.get("reason")
        return out

    freqs = np.logspace(np.log10(20.0), np.log10(BAND_LIMIT_FRACTION * SAMPLE_RATE), 800)
    out["requested"]["resonance"] = r
    out["requested"]["cutoffs_hz"] = list(DEFAULT_CUTOFFS)
    out["requested"]["oversamples"] = list(DEFAULT_OVERSAMPLES)

    metric_sets = []
    for fc in DEFAULT_CUTOFFS:
        ref = reference_magnitude_db(fc, 2.0, freqs)
        for os in DEFAULT_OVERSAMPLES:
            y = run_model(model, np.zeros(IMPULSE_N), fc, r, os, runfilters, workdir)
            if y is None:
                out["measured"][f"fc{fc}_os{os}"] = {"error": "no finite output"}
                continue
            m = shape_metrics(measured_magnitude_db(y, freqs), ref, freqs)
            m["cutoff_hz"] = fc
            m["oversample"] = os
            out["measured"][f"fc{fc}_os{os}"] = m
            metric_sets.append(m)

    if metric_sets:
        agg = {
            key: float(np.nanmean([m[key] for m in metric_sets]))
            for key in (
                "magnitude_rms_db_error",
                "cutoff_3db_error_cents",
                "passband_gain_error_db",
                "stopband_slope_error_db_per_oct",
                "peak_gain_error_db",
                "peak_freq_error_cents",
            )
        }
        out["score_parts"] = agg
        out["linear_score"] = score_linear(agg)
    return out
```

- [x] **Step 4: Implement collect_nonlinear and collect_selfoscillation

```python
def collect_nonlinear(model, runfilters, workdir, oracle, r):
    """Per-model nonlinear set: model at calibrated resonance r vs oracle at k=2.

    r is the user-facing resonance calibrated in collect_linear to match the
    reference at absolute k=2.0, so this runs model and oracle at the same
    absolute operating point. The signal is a hard-switched square-ish tone
    (via np.sign) so the comparison stresses the tanh stages.
    """
    results = {}
    k_abs = 2.0
    for fc in DEFAULT_CUTOFFS:
        for level in LEVELS_DBFS:
            amp = 10.0 ** (level / 20.0)
            t = np.arange(SWEEP_N) / SAMPLE_RATE
            x = amp * np.sign(np.sin(2 * np.pi * 440.0 * t))
            yo = oracle.process(x, SAMPLE_RATE, fc, k_abs, ORDER)
            for os in DEFAULT_OVERSAMPLES:
                ym = run_model(model, x, fc, r, os, runfilters, workdir)
                key = f"fc{fc}_lvl{level}_os{os}"
                if ym is None:
                    results[key] = {"error": "no finite output"}
                    continue
                results[key] = nonlinear_metrics(ym, yo, SAMPLE_RATE)
    cases = [v for v in results.values() if "error" not in v]
    total = float(np.mean([score_nonlinear(v) for v in cases])) if cases else None
    return {"cases": results, "nonlinear_score": total}


def _rms_db(x):
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-30))


def collect_selfoscillation(model, runfilters, workdir):
    """Drive the model at high user resonance and record whether its tail rings.

    The reference self-oscillates at absolute k=4. We do NOT claim any model's
    r maps to k=4; we document, per r in the top of each model's own range,
    whether the output tail stays above -60 dBFS long after the burst ends.
    """
    burst = 12000
    out = {}
    for r in SELFOSC_RESONANCES:
        t = np.arange(SWEEP_N) / SAMPLE_RATE
        x = np.where(t < burst / SAMPLE_RATE, 0.001 * np.sin(2 * np.pi * 500.0 * t), 0.0)
        y = run_model(model, x, DEFAULT_CUTOFFS[1], r, 0, runfilters, workdir)
        if y is None:
            out[str(r)] = {"ringing": True, "tail_rms_db": None, "note": "model diverged"}
            continue
        tail = y[burst:]
        db = _rms_db(tail)
        out[str(r)] = {"ringing": bool(db > -60.0), "tail_rms_db": db}
    return out
```

- [x] **Step 5: Implement main**

```python
def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_oracle():
    from reference import moog_ladder_oracle
    return moog_ladder_oracle


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    ap = argparse.ArgumentParser(description="Evaluate model faithfulness")
    ap.add_argument("--models", default="All", help="comma-separated model names, or All")
    ap.add_argument("--out-dir", default=str(Path("filter_validation/faithfulness") / _timestamp()))
    ap.add_argument("--ref-only", action="store_true", help="only rebuild the oracle / references")
    ap.add_argument("--no-run", action="store_true", help="skip external binary runs")
    args = ap.parse_args(argv)

    ld = {
        "args": vars(args),
        "generated_at": _timestamp(),
        "git_head": _git_head(),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                      "scipy": scipy.__version__},
    }
    run_dir = Path(args.out_dir)
    metrics_dir = run_dir / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    names = MODEL_NAMES if args.models == "All" else [s.strip() for s in args.models.split(",")]
    oracle = get_oracle()
    for name in names:
        record = {
            "model": name,
            "status": "ok",
            "linear": None,
            "nonlinear": None,
            "selfosc": None,
        }
        try:
            linear = collect_linear(name, RUNFILTERS_EXE, str(run_dir))
            record["linear"] = linear
            r = linear["requested"].get("resonance")
            if r is not None:
                record["nonlinear"] = collect_nonlinear(name, RUNFILTERS_EXE, str(run_dir), oracle, r)
            record["selfosc"] = collect_selfoscillation(name, RUNFILTERS_EXE, str(run_dir))
        except Exception as exc:
            record["status"] = f"error: {exc.__class__.__name__}: {exc}"
        with open(metrics_dir / f"{name}.json", "w") as fh:
            json.dump({**ld, **record}, fh, indent=2)
    return 0
```

- [x] **Step 6: Wire the CLI entry point**

Append to `scripts/faithfulness_eval.py`:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 7: Commit**

```bash
git add scripts/faithfulness_eval.py tests/test_eval_metrics.py
git commit -m "add orchestration and per-model collectors"
```

---

### Task 13: Ranking table and the six figures

The design doc fixes exactly six figures (F1–F6). Captions must each include the
operating point `(fc, K, fs, level)`; the figure paths are git-committed, so tests
must assert their existence.

**Files:**
- Modify: `scripts/faithfulness_eval.py`
- New: `tests/test_report.py`
- Output: `docs/moog-faithfulness/plots/F1_{...}.png` ... `F6_{...}.png`

- [x] **Step 1: Write the failing tests**

```python
def test_write_ranking_table_leads_with_both_axes():
    rows = [
        {"model": "A", "linear": 90.0, "nonlinear": 30.0, "combined": 60.0},
        {"model": "B", "linear": 20.0, "nonlinear": 95.0, "combined": 57.5},
    ]
    text = fe.write_ranking_table(rows)
    header = text.splitlines()[0]
    assert header == "| Model | Linear | Nonlinear | Combined |"
    assert text.index("Linear") < text.index("Combined")
    assert text.index("Nonlinear") < text.index("Combined")


def test_generate_figures_writes_six_pngs(tmp_path):
    out = fe.generate_figures({}, tmp_path)
    names = sorted(p.name for p in out)
    assert len(names) == 6, names
    assert all(p.suffix == ".png" for p in out)
    assert all(p.stat().st_size > 0 for p in out)


def test_figure_caption_states_operating_point():
    cap = fe.fig_caption("F1", fc=1000, K=2, f_s=44100, level=-12)
    assert "1000" in cap and "44100" in cap and "level" in cap
```

- [x] **Step 2: Run to verify it fails**

Expected: FAIL with `ImportError` when pytest imports `scripts/faithfulness_eval` if
it has no exports expected above. Add imports as needed.

- [x] **Step 3: Implement write_ranking_table

```python
def write_ranking_table(rows):
    """Render a markdown table. Both axes must appear before the combined score."""
    lines = ["| Model | Linear | Nonlinear | Combined |",
             "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['linear']:.2f} | {r['nonlinear']:.2f} | {r['combined']:.2f} |")
    return "\n".join(lines)
```

- [x] **Step 4: Implement the figure generator

```python
def fig_caption(tag, **op):
    """Build a one-line caption that pins the operating point."""
    base = f"F{tag} — {op.pop('title', tag)}"
    op_text = ", ".join(f"{k}={v}" for k, v in op.items())
    return f"{base} at ({op_text})"
```

The six figures (drawable from the record schema; supersedes the earlier
magnitude-curve list because the collectors store metrics, not curves):

F1: analytic reference magnitude vs. k at fc=1000 (D'Angelo Part I closed form).
F2: model-vs-reference cutoff error in cents vs. linear score, one point per model. Errors
   from the recorded `fc1000.0_os0` case; skip flagged/error models and non-finite values.
F3: harmonic spectrum at -6 dBFS, oracle line plus one line per model. Bins come from
   `harmonic_profile_db_model` / `_oracle` (10 entries) of the `fc1000.0_lvl-6.0_os0` case.
F4: THD vs. input level, one line per model plus the oracle line. `thd_model_percent` /
   `thd_oracle_percent` across the 5 levels at `fc1000.0_os0`.
F5: spectral distance vs. fc at one level, one line per model. `spectral_distance_db` of
   the `lvl-6.0_os0` cases at the three cutoffs.
F6: linear vs. nonlinear score bars per model, sorted by combined score.

```python
def _case(rec, key):
    part = rec.get("nonlinear") or {}
    return (part.get("cases") or {}).get(key)


def _ok(rec):
    return rec.get("status") == "ok"


def generate_figures(records, out_dir):
    """Write exactly the six fixed figures as PNGs. Returns the Path list.

    records: {model_name: per-model JSON dict from main's collectors}.
    out_dir: where the PNGs land (normally docs/moog-faithfulness/plots).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name, fc, k, level):
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        written.append(out_dir / name)
        return out_dir / name

    # F1: reference magnitude vs k (analytic, D'Angelo Part I closed form).
    fig, ax = plt.subplots()
    freqs = np.logspace(1, np.log10(BAND_LIMIT_FRACTION * SAMPLE_RATE), 2000)
    for k in (0.0, 1.0, 2.0, 3.0, 3.9):
        ax.plot(freqs, reference_magnitude_db(1000.0, k, freqs), label=f"k={k}")
    ax.set_xscale("log"); ax.set_ylabel("dB"); ax.set_xlabel("Hz"); ax.legend()
    save(fig, "F1_reference_magnitude_vs_k.png", 1000, "various", 0)

    # F2: cutoff error vs linear score, one point per ok model.
    fig, ax = plt.subplots()
    for name, rec in records.items():
        if not _ok(rec):
            continue
        score = (rec.get("linear") or {}).get("linear_score")
        err = ((rec.get("linear") or {}).get("measured") or {}).get(
            "fc1000.0_os0", {}).get("cutoff_3db_error_cents")
        if score is not None and err is not None and np.isfinite(err):
            ax.scatter(score, err, label=name)
    ax.set_xlabel("linear score"); ax.set_ylabel("cutoff error (cents)")
    ax.set_title(fig_caption("F2", fc=1000, K="cal", f_s=SAMPLE_RATE, level=-6))
    if records: ax.legend(fontsize="small")
    save(fig, "F2_cutoff_error_vs_score.png", 1000, "cal", -6)

    # F3: harmonic spectrum at -6 dBFS, oracle plus models.
    fig, ax = plt.subplots()
    # The oracle profile is identical in every model's case; draw it once, from
    # the first ok model that carries the case.
    for name, rec in records.items():
        if not _ok(rec):
            continue
        c = _case(rec, "fc1000.0_lvl-6.0_os0")
        if not c:
            continue
        if c.get("harmonic_profile_db_oracle"):
            n = len(c["harmonic_profile_db_oracle"])
            ax.plot(range(1, n + 1), c["harmonic_profile_db_oracle"],
                    color="k", linestyle="--", label="oracle")
        break
    for name, rec in records.items():
        if not _ok(rec):
            continue
        c = _case(rec, "fc1000.0_lvl-6.0_os0")
        if not c:
            continue
        prof = c.get("harmonic_profile_db_model")
        if prof:
            n = len(prof)
            ax.plot(range(1, n + 1), prof, label=name)
    ax.set_xlabel("harmonic order"); ax.set_ylabel("dBFS")
    ax.set_title(fig_caption("F3", fc=1000, K=2, f_s=SAMPLE_RATE, level=-6))
    if records: ax.legend(fontsize="small")
    save(fig, "F3_harmonic_spectrum.png", 1000, 2, -6)

    # F4: THD vs level, model lines plus oracle line.
    fig, ax = plt.subplots()
    model_thd = {name: [] for name in records}
    oracle_by_level = {}
    for name, rec in records.items():
        if not _ok(rec):
            continue
        for level in LEVELS_DBFS:
            c = _case(rec, f"fc1000.0_lvl{level}_os0")
            if not c:
                continue
            t = c.get("thd_model_percent")
            if t is not None:
                model_thd[name].append((level, t))
            oracle_by_level[level] = c.get("thd_oracle_percent")
    for name, pts in model_thd.items():
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name)
    o = sorted((l, v) for l, v in oracle_by_level.items() if v is not None)
    if o:
        ax.plot([p[0] for p in o], [p[1] for p in o],
                color="k", linestyle="--", label="oracle")
    ax.set_xlabel("level (dBFS)"); ax.set_ylabel("THD (%)")
    ax.set_title(fig_caption("F4", fc=1000, K=2, f_s=SAMPLE_RATE, level="5 levels"))
    if records: ax.legend(fontsize="small")
    save(fig, "F4_thd_vs_level.png", 1000, 2, "levels")

    # F5: spectral distance vs fc at one level, one line per model.
    fig, ax = plt.subplots()
    for name, rec in records.items():
        if not _ok(rec):
            continue
        pts = []
        for fc in DEFAULT_CUTOFFS:
            c = _case(rec, f"fc{fc}_lvl-6.0_os0")
            d = c.get("spectral_distance_db") if c else None
            if d is not None:
                pts.append((fc, d))
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=name)
    ax.set_xscale("log"); ax.set_xlabel("fc (Hz)"); ax.set_ylabel("spectral distance (dB)")
    ax.set_title(fig_caption("F5", fc="3 cutoffs", K="cal", f_s=SAMPLE_RATE, level=-6))
    if records: ax.legend(fontsize="small")
    save(fig, "F5_spectral_distance_vs_fc.png", "all", "cal", -6)

    # F6: linear vs nonlinear score bars, sorted by combined.
    fig, ax = plt.subplots()
    bars = []
    for name, rec in records.items():
        if not _ok(rec):
            continue
        lin = (rec.get("linear") or {}).get("linear_score")
        nlin = (rec.get("nonlinear") or {}).get("nonlinear_score")
        if lin is None or nlin is None:
            continue
        bars.append((name, lin, nlin,
                     COMBINED_WEIGHT_LINEAR * lin + (1 - COMBINED_WEIGHT_LINEAR) * nlin))
    bars.sort(key=lambda b: b[3], reverse=True)
    names = [b[0] for b in bars]
    x = np.arange(len(bars))
    if bars:
        ax.bar(x - 0.2, [b[1] for b in bars], 0.4, label="linear")
        ax.bar(x + 0.2, [b[2] for b in bars], 0.4, label="nonlinear")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize="small")
    ax.set_ylabel("score (0-100)"); ax.set_ylim(0, 100); ax.legend()
    ax.set_title(fig_caption("F6", fc="all", K="cal/2", f_s=SAMPLE_RATE, level="mixed"))
    save(fig, "F6_score_bars.png", "all", "cal/2", "mixed")

    assert len(written) == 6, f"expected 6 figures, got {len(written)}"
    return written
```

- [x] **Step 5: Wire figures into main

Add a `--write-figs PATH` option to main; after collecting, build `records =
{name: record_dict}` (the same dicts main already writes) and call
`generate_figures(records, PATH)`.

- [x] **Step 6: Commit

```bash
git add scripts/faithfulness_eval.py tests/test_report.py docs/moog-faithfulness/plots
git commit -m "add ranking table and fixed six-figure set"
```

Note: the PNGs are committed, so `docs/moog-faithfulness/plots/` must be tracked.
At Task 13 the committed figures are rendered from available records (possibly
empty fixture data); Task 14 regenerates them from the real end-to-end run.

---

### Task 14: End-to-end run and report generation

**Files:**
- Modify: `scripts/faithfulness_eval.py`, `example/run-filters.cpp`, `example/helpers.hpp`
- New: `scripts/faithfulness_report.py`

- [x] **Step 1: Build the float32 path first

Done in Tasks 6/7 (`--float` landed in commit `afa8389`; the PCM16 regression
tests live in `tests/test_runfilters_float.py`). Nothing left to build.

- [x] **Step 2: Run the full sweep

```bash
mkdir -p filter_validation/faithfulness
python scripts/faithfulness_eval.py --models All --out-dir filter_validation/faithfulness/run1
```

Check: per-model JSON exists, per-model `status == "ok"` (or `flagged`, which a
look at `flagged` keys explains), and no model reports the legacy suite's
`-240.0` dB self-oscillation sentinel (the faithful harness uses a `-600` dB
floor instead, so any `-240.0` in a record is a bug).

Note (Task 14 dispatch, reconciled against the real run): the earlier wording
claimed the combined order is "Stilson/Huovilainen/Improved not reversed". The
measured order from `filter_validation/figs_all/metrics` is HyperionTanh,
Hyperion, Huovilainen, MusicDSP, OberheimVariation, Krajeski, Improved,
Microtracker, HyperionLegacy, RKSimulation, Simplified, Stilson. Stilson's
linear axis measures 0.30 (its `fc100.0_os0` magnitude sits on the `-600` dB
floor, consistent with the legacy-suite artifact `dc_gain=0.0000`), so it ranks
last, not first. The report states the measured order; it does not assert a
predicted order.

Done (`filter_validation/faithfulness/run1/metrics/*.json`, reproduces
`figs_all` byte-for-byte; all 12 `status == "ok"`).

- [x] **Step 3: Implement the report writer

`scripts/faithfulness_report.py` reads the JSON directory and emits
`docs/moog-faithfulness-report.md`. It renders:

1. Purpose and scope.
2. HOW this evaluation is calibrated (fc is leading-pole, not -3 dB, so the
   legacy suite's rejection is expected).
3. Top-line side-by-side ranking table (from Task 13).
4. Sorted linear ranking.
5. Sorted nonlinear ranking.
6. Per-model verdicts: 1–3 sentences each, a single `reason` from the report.
7. Six figures with operating-point captions.
8. Reproduction, environment (py/versions), commit hashes, and confirmation that
   `scripts/filter_verification.py` is untouched.
9. Threats to validity, incl. known: no hardware data, no validated D'Angelo
   errata, comparison port may be coincidentally matched, filter libraries may
   be mislabeled, legacy test did not measure the -3 dB point.

- [x] **Step 4: Assertion gate before writing

In the report script, before writing the final markdown, assert:

- each rankable model has `status == "ok"`; a model with `status == "flagged"`
  is excluded from the ranking and its flag reason is stated instead,
- the ranking table is non-empty,
- at least one model rings (self-oscillation `ringing: True` exists, so the
  self-oscillation risk is stated; HyperionLegacy does not ring in the faithful
  run, and the legacy suite's step data shows it settles only in the last
  millisecond of the 743 ms window at r=0.50 and r=0.90 — the report says both
  facts),
- `docs/moog-faithfulness/plots/*.png` exists (all six).

- [x] **Step 5: Generate and open the report

```bash
python scripts/faithfulness_report.py filter_validation/faithfulness/run1
```

- [x] **Step 6: Commit

```bash
git add scripts/faithfulness_report.py docs/moog-faithfulness-report.md && git commit -m "add end to end report generation"
```

---

### Task 15: Address mid-flight surprises, including figure counting

Keep a running `### Mid-flight` section in the plan.

- [x] **Step 1: Figure blessing

Count exactly six committed figures. If a seventh is genuinely useful it must be
approved, the design doc updated, and this remaining question removed.

- [x] **Step 2: Confirmed artifacts for the report

Confirm in the report (values corrected during Task 14 review against
`filter_validation/2026-09-28_201435/metrics/step.json`; the step suite swept
fc=1000 only):

- Stilson `dc_gain=0.0000` at the step operating point (not "all three
  cutoffs"), Improved `dc_gain=-0.99997` (inversion, shrinking across
  resonance; not exactly -1.0000),
- HyperionLegacy step settling 742.27 ms (r=0.50) / 742.77 ms (r=0.90), last
  millisecond of the 743.04 ms window; it settles promptly (27 ms) at r=0.00.
  MusicDSP alone (r=0.90, 743.04 ms = record length) never settles,
- THD ordering at -6 dBFS floor: Stilson 0.0000 %, Huovilainen 0.0023 %.

- [x] **Step 3: Robustness grep

Grep the final report for absolute non-operating-point claims: "kill", "match",
"identical", "hardware", "best-sounding". Any cleansed.

- [x] **Step 4: Commit, add the design-doc Completion section, and tag

Commit the report, record the run's outcome under the design doc's
`### Completion` section, and cut the `moog-faithfulness-v1` tag. The ledger
at `.superpowers/sdd/.../progress.md` is the controller's to write and is never
touched from here, so no completed dates or run metadata are copied into any
doc by hand.

### Mid-flight

- **Figure count: six, and no seventh needed.** `git ls-files
  docs/moog-faithfulness/plots/` returns exactly six committed PNGs, F1..F6,
  and each is referenced exactly once by `docs/moog-faithfulness-report.md`. The
  design doc's figure table (six rows, F1..F6) matches that set 1:1, so the
  counting question closes against the existing table rather than by adding a
  plot to it.
- **Artifact set: three confirmed, one added.** The three step facts were
  checked against `filter_validation/2026-09-28_201435/metrics/step.json` and
  left as they stand (Stilson `dc_gain = 0.0000`, Improved `-0.99997`,
  HyperionLegacy settling at 742.27 ms / 742.77 ms with MusicDSP reporting the
  record length, i.e. never settling). The THD ordering was genuinely missing
  and is now on Stilson's card, carrying the legacy sweep's operating point
  (fc=5000, r=0.00, -6 dBFS) and contrasted, in numbers, against section 4.3's
  own fc=1000 figure for the same model, so it cannot read as contradicting it.
  Each artifact entry is now a (fact, what-this-sweep-cannot-say) pair, because
  a step measurement and a THD measurement are confirmed by different evidence,
  and the contrast figure is read out of the record rather than typed in beside
  it. Each step card also names the operating point that suite measured it at
  (fc=1000 only, at resonances 0.0/0.5/0.9), so the step facts are scoped the
  way the THD fact is.
- **Robustness grep: clean.** `kill`, `match`, `identical`, `hardware` and
  `best-sounding` over the final report return the same seven pre-existing
  lines, all of them scoped or counterfactual uses, and nothing from the new
  text. No edits were needed.

---

### Task 16: Final verification

- [x] **Step 1: Clean tree

```bash
git status --short
```

Expect no untracked sources except `filter_validation/` (gitignored) and any
ignored venv.

Done: `git status --short` is empty at `23994fe`; `git status --porcelain --ignored`
lists only `.pytest_cache/`, `.superpowers/`, `.venv/`, `build/`,
`filter_validation/`, and the three `__pycache__/` directories.

- [x] **Step 2: Full tests

```bash
python -m pytest tests -q
```

Done: 169 passed, 6 warnings in 19.72s.

- [x] **Step 3: Rebuild from scratch path

If a fresh clone is available, re-run `cmake --build` and the full sweep to
confirm the harness is reproducible end-to-end.

Done (2026-09-29): cloned the local repo to `/tmp/opencode/moog-fresh` at
`23994fe` and built from clean sources (`cmake -S . -B build` plus
`cmake --build build -j`, 7.9s wall, `build/RunFilters` produced). The full
sweep into `filter_validation/faithfulness/rebuild` took 11m57s and wrote 12
model JSONs, every one `status == "ok"`. Compared leaf by leaf against
`filter_validation/faithfulness/run1/metrics/*.json`: the maximum absolute
difference over every numeric field is 0.0, and the only differing fields are
the four provenance stamps (`args.out_dir`, `args.write_figs`, `generated_at`,
`git_head`). Regenerating the report from the rebuild run reproduced the
committed report byte for byte apart from its `git_head` line.

- [x] **Step 4: Close the plan

Remove the `STATUS: not yet read` errata note only after reading both errata PDFs;
otherwise leave it and make the follow-up a tracked item.

Done (2026-09-29): both PDFs were read by text extraction (`pypdf`). All three corrections were
checked against the ports and none applies, so the STATUS line was replaced with the read
record in `reference/upstream/PROVENANCE.md` and in this plan's "## Errata status" block, and
the report's erratum bullet now says the errata were read. No measured number moved.

---

## Completion checklist (global)

- [x] All Tasks 1–16 marked done.
- [x] `python -m pytest tests -q` passes.
- [x] Report committed: `docs/moog-faithfulness-report.md`.
- [x] Six figures committed: `docs/moog-faithfulness/plots/`.
- [x] `scripts/filter_verification.py` untouched.
- [x] Every claim in the report is tied to a measured artifact.
