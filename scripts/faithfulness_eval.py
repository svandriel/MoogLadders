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
    python scripts/faithfulness_eval.py
    python scripts/faithfulness_eval.py --models Stilson,Improved
    python scripts/faithfulness_eval.py --out-dir filter_validation/faithfulness/run1

    The binary is build/RunFilters unless RUNFILTERS_EXE names another one. It
    must already be built: the driver does not build it.

Output:
    <out-dir>/
        <Model>_c<fc>_r<res>_in.wav     the driven input
        <Model>_c<fc>_r<res>_out/       RunFilters' own per-model output
        metrics/<Model>.json            the record for that model
"""

import argparse
import json
import os
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy

# The `reference` package lives at the repo root, but running this file as a
# script puts scripts/ on sys.path instead, so `from reference import ...`
# raises ModuleNotFoundError exactly when the CLI is used and never when it is
# imported from the tests. Same bootstrap tests/test_eval_metrics.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 44100
BAND_LIMIT_FRACTION = 0.4
ORDER = 4

# Length of the raised-cosine fade at the end of a sine_sweep, in samples.
SWEEP_FADE_N = 1024

# RunFilters' own domain, from the -s check in example/run-filters.cpp:230.
SUPPORTED_OVERSAMPLES = (0, 2, 4, 8)

FILTER_NAMES: List[str] = [
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
    """Logarithmic sine sweep from f0 to f1 over n samples, faded out at the end.

    The last SWEEP_FADE_N samples are scaled by a raised cosine running 1 -> 0, so
    the signal does not stop on a step from whatever phase it happened to reach.
    That step is a broadband transient, and the windowed FFT and THD measured
    downstream of this signal would read it as filter content. The envelope is
    positive, so the zero crossings the sweep tests count are unaffected.
    """
    t = np.arange(n) / float(fs)
    k = np.log(f1 / f0)
    phase = 2.0 * np.pi * f0 * (np.exp(k * t) - 1.0) / k
    x = amplitude * np.sin(phase)
    fade = min(SWEEP_FADE_N, n)
    if fade > 1:
        ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade) / (fade - 1)))
        x[n - fade:] *= ramp
    return x


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
    """Write a mono WAV, 32-bit IEEE float by default or 16-bit PCM when int16.

    The header is built here rather than with the wave module, because wave
    always writes a PCM format tag: float bytes behind a PCM tag get decoded as
    int32 by RunFilters' ReadWavFile and by read_wav below, which rescales the
    signal by 2**-31 instead of returning it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float64)
    if int16:
        raw = np.clip(x * 32767.0, -32768.0, 32767.0).astype("<i2")
        tag, width = 1, 2
    else:
        raw = x.astype("<f4")
        tag, width = 3, 4
    data_size = raw.nbytes
    header = b"".join([
        b"RIFF", struct.pack("<I", 36 + data_size), b"WAVE",
        b"fmt ", struct.pack("<I", 16),
        struct.pack("<HHIIHH", tag, 1, fs, fs * width, width, width * 8),
        b"data", struct.pack("<I", data_size),
    ])
    path.write_bytes(header + raw.tobytes())


def _read_riff(path):
    """Return ((format_tag, channels, fs, bits), data_bytes) from a RIFF/WAVE file.

    Walks the chunk list rather than searching for chunk ids, so a file carrying
    a LIST or fact chunk before fmt is still read correctly.
    """
    blob = Path(path).read_bytes()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    fmt = None
    data = None
    at = 12
    while at + 8 <= len(blob):
        chunk_id = blob[at:at + 4]
        (size,) = struct.unpack_from("<I", blob, at + 4)
        body = blob[at + 8:at + 8 + size]
        if chunk_id == b"fmt ":
            tag, channels, fs, _byte_rate, _align, bits = struct.unpack_from(
                "<HHIIHH", body, 0
            )
            fmt = (tag, channels, fs, bits)
        elif chunk_id == b"data":
            if len(body) != size:
                raise ValueError(
                    f"{path} declares a {size} byte data chunk but holds {len(body)}"
                )
            data = body
            break
        at += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path} has no fmt or data chunk")
    return fmt, data


def read_wav(path):
    """Read a mono WAV into float64. Handles 8/16/24/32 bit integer and 32 bit float."""
    (tag, channels, fs, bits), raw = _read_riff(path)
    if channels != 1:
        raise ValueError(f"{path} has {channels} channels, expected mono")
    if tag == 3:
        if bits != 32:
            raise ValueError(f"unsupported float sample width {bits}")
        return np.frombuffer(raw, "<f4").astype(np.float64), fs
    width = bits // 8
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
    raise ValueError(f"unsupported sample width {bits}")


def read_wav_float(path):
    """Read a mono float32 WAV, as written by `RunFilters --float`.

    Raises rather than reinterpreting bytes, because reading 16-bit PCM as float32
    yields plausible-looking nonsense instead of an error.
    """
    (tag, channels, _fs, bits), raw = _read_riff(path)
    if tag != 3 or bits != 32:
        raise ValueError(f"{path} is audio format {tag} at {bits} bits, not float32")
    if channels != 1:
        raise ValueError(f"{path} has {channels} channels, expected mono")
    return np.frombuffer(raw, "<f4").astype(np.float64)


def run_model(model, signal, cutoff, resonance, oversample, runfilters, workdir, *, timeout=60):
    """Drive one model with signal via RunFilters --float.

    RunFilters writes one WAV per model, named <FilterName>_c<cutoff>_r<resonance>
    (_os<n>x when oversampling) into the output directory, and always processes
    every model, so the file whose name starts with `model` is the only thing that
    identifies this model's result. The name has to be the file-name spelling, one
    of FILTER_NAMES: RunFilters has no single-model mode, so a name that matches no
    file is a failed run rather than a different model.

    Resonance is 0..1 because that is the domain RunFilters accepts and exits 1
    outside of, and nothing deeper justifies it: the models map r to physical
    feedback differently, Hyperion scaling by 4 and OberheimVariation expecting r
    in 1..10 so that it runs negative on this domain. Task 12's resonance axis
    should expose that per-model spread rather than assume it away.

    `timeout` bounds the invocation in seconds and a run that overruns it is
    treated as failed, like a nonzero exit. It is keyword-only so a test can use a
    deadline short enough to keep the suite quick.

    Returns the output as float64, or None if the run failed, produced no file for
    this model, or produced any non-finite sample.
    """
    if not np.isfinite(cutoff) or cutoff <= 0.0 or cutoff >= 0.5 * SAMPLE_RATE:
        raise ValueError(f"cutoff {cutoff} must be in (0, fs/2)")
    # A NaN fails this test already: every comparison against it is False.
    if not 0.0 <= resonance <= 1.0:
        raise ValueError(f"resonance {resonance} must be in [0.0, 1.0]")
    if oversample not in SUPPORTED_OVERSAMPLES:
        raise ValueError(f"oversample {oversample} must be one of {SUPPORTED_OVERSAMPLES}")
    # One rounding, used for the tag, the argument and the glob, because
    # BuildOutputFilename names the cutoff with %.0f. Flooring it here would look
    # for a directory the binary never created at 1000.6 Hz. Python and C both
    # round halves to even, on a double and a float respectively.
    fc = f"{cutoff:.0f}"
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = f"{model}_c{fc}_r{resonance:.2f}"
    src = workdir / f"{tag}_in.wav"
    outdir = workdir / f"{tag}_out"
    # RunFilters does not create its output directory, so the harness does.
    outdir.mkdir(parents=True, exist_ok=True)
    write_wav(src, signal)

    try:
        result = subprocess.run(
            [
                str(runfilters),
                "-f", str(src),
                "-c", fc,
                "-r", f"{resonance:.2f}",
                "-s", str(oversample),
                "-o", str(outdir),
                "--float",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None

    pattern = f"*_c{fc}_r{resonance:.2f}"
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


def spectrum_db(x, fs):
    """Return (freqs_hz, magnitude_db) from a Hann-windowed real FFT.

    Normalized so a full-scale sine at a bin centre reads 0 dB: the magnitude is
    divided by half the window's coherent gain, sum(win)/2, which is exactly what
    a unit sine projects onto its own bin. Left unnormalized the reading is a
    function of the record length, a unit sine peaking at N/4, so a floor measured
    on a long record could not be compared with one measured on a short one. Only
    the on-bin case is exact: a tone between bins reads up to 1.4 dB low, which is
    the Hann mainlobe's scalloping loss and not the scale. The 1e-30 floor is
    applied after normalizing, so digital silence reads -600 dB at any length.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        raise ValueError("signal too short for spectral analysis")
    win = np.hanning(x.size)
    mag = np.abs(np.fft.rfft(x * win)) / (0.5 * float(win.sum()))
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    return freqs, 20.0 * np.log10(np.maximum(mag, 1e-30))


def noise_floor_db(x, fs):
    """Median magnitude in dB of the Hann-windowed spectrum.

    The median rather than the minimum, so one occupied bin or one dip between
    bins cannot stand in for the floor. On the spectrum_db scale this reads the
    noise level of the record: for white noise of standard deviation sigma each
    real bin is a complex Gaussian of mean square sigma**2*sum(win**2), so its
    magnitude is Rayleigh and the median of the record is
    sqrt(ln 2)*sigma*sqrt(sum(win**2))/(sum(win)/2). The record length cancels,
    which is the point of the normalization in spectrum_db.
    """
    _, db = spectrum_db(x, fs)
    return float(np.median(db))


def fundamental_freq(x, fs):
    """Frequency of the largest spectral peak in the band, in Hz, ignoring DC.

    The search covers the whole band rather than stopping below a fixed ceiling.
    A sub-1 kHz ceiling cannot find the tones this harness measures, which sit
    at and above 1 kHz: with the tone excluded, the largest remaining bin is the
    low-frequency skirt the Hann window leaves behind, and every harmonic is then
    read against a fundamental the signal does not contain.

    The peak is refined by parabolic interpolation over the three bins around the
    maximum, so the returned frequency is the interpolated peak and not the centre
    of the bin holding it. That is what makes the harmonic measurements mean
    anything: bin centres are m*fs/N, and projecting a harmonic at h*f0 is then
    h*delta bins off the true h*f0, an error the unwindowed Goertzel attenuates
    by sinc(h*delta). Both readings are biased together but not equally, since the
    fundamental sees sinc(delta) and the H2 sees sinc(2*delta), so an uncorrected
    bin centre turns a true 10 % THD into anything from 0.03 % to 10 %. On the
    dB values the fit is measured accurate to better than 0.02 bins over the whole
    offset range, which is what keeps a true 10 % inside 0.1 and a true -20 dB H2
    inside 0.03 dB. Interpolating in dB rather than in linear magnitude is what
    buys that: the linear-magnitude fit is three times worse, and the dB fit is
    also invariant to the overall scale of the spectrum, so the normalization in
    spectrum_db cannot move the answer.

    Falls back to the plain bin centre when the peak is not interpolable: at the
    first or last bin available, or on a flat peak whose curvature is too small to
    fit. The fit is then no better than the bin, and claiming otherwise would be
    worse than reporting the bin. DC is excluded because a DC offset is not a
    fundamental; a silent record has no peak, and reports the first bin above DC,
    which thd_percent then rejects as zero.
    """
    x = np.asarray(x, dtype=np.float64)
    freqs, db = spectrum_db(x, fs)
    peak = int(np.argmax(db[1:])) + 1
    delta = 0.0
    if 2 <= peak <= db.size - 2:
        alpha, beta, gamma = db[peak - 1], db[peak], db[peak + 1]
        curvature = alpha - 2.0 * beta + gamma
        if abs(curvature) > 1e-9:
            delta = 0.5 * (alpha - gamma) / curvature
    return float(freqs[peak] + delta * fs / x.size)


def goertzel_amplitude(x, fs, freq):
    """Amplitude of `freq` in x, by Goertzel, normalized so a unit sine reads 1.

    Projects at the exact requested frequency rather than reading the nearest FFT
    bin, so a signal that does not contain a whole number of periods does not
    leak into the reading.

    Unwindowed, and the projection is exact only when `freq` is where the energy
    actually is: an offset of delta cycles reads sinc(delta) of the amplitude, and
    delta a whole cycle reads nothing at all. That is why the callers pass an
    interpolated frequency rather than a bin centre.

    Unwindowing also sets a floor on what can be resolved. The projection of a
    tone a long way off responds as 1/(pi*df*T), so the fundamental leaks into
    every harmonic's reading at about 1/(pi*f0*T). Against a -20 dB harmonic that
    is 0.24 % at 440 Hz over 131072 samples, and 0.84 % at 202 Hz over 65536, and
    it is the limiting error once the frequency is interpolated.

    `freq` is not folded into the band. A frequency at or above Nyquist is a
    caller error here, because a Goertzel at such a frequency does not read the
    requested frequency, it reads whatever content happens to alias to it.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        raise ValueError("empty signal has no amplitude at any frequency")
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


def _harmonic_frequencies(f0, order, fs):
    """The frequency of harmonic `order` of f0, or None if it is at or above Nyquist.

    A harmonic at or above fs/2 cannot be represented in the record, and
    projecting there would read the frequency that content aliases to, so the
    callers treat the harmonic as absent rather than measure something else.
    """
    freq = order * f0
    return freq if freq < 0.5 * fs else None


def harmonic_ratio_db(x, fs, order):
    """Amplitude of harmonic `order` relative to the fundamental, in dB.

    A harmonic at or above Nyquist is reported as absent, on the same -600 dB
    floor spectrum_db gives digital silence, because the alternative is a reading
    of whatever aliased into the requested frequency.
    """
    f0 = fundamental_freq(x, fs)
    fund = goertzel_amplitude(x, fs, f0)
    freq = _harmonic_frequencies(f0, order, fs)
    harm = 0.0 if freq is None else goertzel_amplitude(x, fs, freq)
    return 20.0 * np.log10(max(harm / max(fund, 1e-30), 1e-30))


def thd_percent(x, fs, orders=10):
    """Total harmonic distortion in percent, harmonics 2 through `orders`.

    The fundamental is the interpolated peak, not the bin holding it, so the
    harmonics are projected where they actually are: every harmonic sits within
    0.02 bins of the frequency asked for, measured over a full sweep of fractional
    bin offsets, which is what keeps a tone placed deliberately between bins
    inside 1 % of a true 10 % and its H2 inside 0.1 dB of -20 dB. Before the peak
    was interpolated the same signals read anything from 0.03 % to 10 %.

    What limits the reading after that is the unwindowed leakage floor in
    goertzel_amplitude: about 0.24 % of a -20 dB harmonic at 440 Hz over 131072
    samples, rising to 0.84 % at 202 Hz over 65536, where the fundamental is only
    300 cycles from the H2 and leaks into its bin. A shorter record or a lower
    fundamental has a proportionally worse floor, so the tolerances above are
    stated for those record lengths and not for all of them.

    Harmonics at or above Nyquist are skipped rather than folded back into the
    band, so the total is over the harmonics the record can represent. A record
    with no measurable fundamental reads 0.0.
    """
    f0 = fundamental_freq(x, fs)
    fund = goertzel_amplitude(x, fs, f0)
    if fund <= 0.0:
        return 0.0
    power = 0.0
    for order in range(2, orders + 1):
        freq = _harmonic_frequencies(f0, order, fs)
        if freq is None:
            continue
        amp = goertzel_amplitude(x, fs, freq)
        power += amp * amp
    return 100.0 * np.sqrt(power) / fund


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
    for i, (r, curve) in enumerate(zip(resonances, responses)):
        err = float(np.sqrt(np.mean((np.asarray(curve) - reference_db) ** 2)))
        if best is None or err < best[1]:
            best = (float(r), err, i)
    return best


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


def _cross_freq(freqs, db, target_db):
    """Frequency where the curve falls through target_db, linearly interpolated.

    The *falling* crossing, and the last one, not the first crossing of any
    kind. A resonant ladder rises through the target on its way up the skirt and
    falls through it on its way down past the cutoff, so at the operating point
    this harness scores (k=2, fc=1000) the -3 dB level is crossed twice: at
    835 Hz on the way up, and at 1055 Hz on the way down. Only the second is the
    cutoff, and the first is not merely a worse answer -- it sits on the
    resonant skirt, where the curve is still climbing, so every window derived
    from it (passband, stopband) is anchored to the wrong place in the band.

    The rising crossing is skipped rather than measured and discarded because
    whether a curve has one depends on where the band starts: the same response
    read over a band that already sits above the target has no rising crossing
    to skip, and a search that had already committed to the first one would
    report a different cutoff for the same filter.

    Index 0 is not a crossing. diff[0] >= 0 and diff[1] < 0 is how a curve that
    *starts* exactly on the target looks to a sign test, and there is no
    approach from above to interpolate, so a curve that lands on the target at
    the first bin reports no crossing at all rather than a fabricated one.

    A curve that never falls through the target reports NaN.
    """
    diff = np.asarray(db) - target_db
    falling = np.where((diff[:-1] >= 0.0) & (diff[1:] < 0.0))[0]
    falling = falling[falling >= 1]
    if len(falling) == 0:
        return float("nan")
    i = int(falling[-1])
    d0, d1 = diff[i], diff[i + 1]
    frac = 0.0 if d1 == d0 else float(d0) / (d0 - d1)
    return float(freqs[i] + frac * (freqs[i + 1] - freqs[i]))


def shape_metrics(measured_db, reference_db, freqs):
    """Compare one magnitude curve against the reference.

    Returns magnitude_rms_db_error, cutoff_3db_error_hz, cutoff_3db_error_cents,
    passband_gain_error_db, stopband_slope_db_per_oct,
    stopband_slope_error_db_per_oct, peak_gain_error_db, peak_freq_error_cents.

    The peak is measured *above the passband*, as the largest value of
    `curve - curve[0]`, not as the band argmax. The two differ wherever the
    response is not already peaked at DC, which is to say for every resonant
    setting: the reference at k=2 has a DC gain of -9.54 dB and a peak of
    -1.74 dB, so the band argmax is the peak only by luck of the numbers, while
    a model with no resonance at all reports a flat curve whose argmax is its DC
    gain. Scored on the argmax such a model reads 1.74 dB of peak error where
    the honest answer is the full 7.80 dB hump it is missing, and it collects
    most of the peak credit it did not earn. Referencing the hump to the first
    bin measures the hump itself, so a curve that has no hump is charged for
    the reference's, in either direction.

    Errors that cannot be measured are NaN rather than zero: a curve that never
    crosses -3 dB has no cutoff to report, and reporting 0 there would let a
    model that failed to roll off score as a perfect match on three axes.
    score_linear scores a NaN as 0.0, so a missing measurement is a failure.
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

    hump_m = measured_db - measured_db[0]
    hump_r = reference_db - reference_db[0]
    peak_m = int(np.argmax(hump_m))
    peak_r = int(np.argmax(hump_r))
    peak_freq_cents = float(1200.0 * np.log2(freqs[peak_m] / freqs[peak_r]))

    return {
        "magnitude_rms_db_error": rms,
        "cutoff_3db_error_hz": float(f3m - f3r),
        "cutoff_3db_error_cents": float(cents),
        "passband_gain_error_db": passband,
        "stopband_slope_db_per_oct": float(slope_m),
        "stopband_slope_error_db_per_oct": float(abs(slope_m - slope_r)),
        "peak_gain_error_db": float(hump_m[peak_m] - hump_r[peak_r]),
        "peak_freq_error_cents": peak_freq_cents,
    }


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

    The value is taken in magnitude. Four of the six scored metrics are signed
    -- cutoff_3db_error_cents, passband_gain_error_db, peak_gain_error_db and
    peak_freq_error_cents can each come out either way, because a model can run
    its cutoff low or its peak high and both are equally wrong. Clamping only
    the upper side, as this did, therefore scored every "too low" model as
    perfect: `value <= best` is true for every negative number, so a cutoff
    2000 cents LOW kept the full 0.20 of the cutoff weight, and a peak 6 dB
    LOW kept 0.12, on top of the passband's 0.15 and the peak frequency's 0.08.
    That is 0.55 of the weight handed out for being wrong in one direction
    only.

    Taking the magnitude here rather than in shape_metrics is deliberate: the
    metric values themselves stay signed in the metrics dict, because a report
    that says "the cutoff was 200 cents LOW" is worth more than one that says
    "200", and only the score is direction-blind.

    Non-finite values score 0.0 (a failed measurement is a failure, not a pass).
    """
    if not np.isfinite(value):
        return 0.0
    value = abs(float(value))
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


def align_signals(model, oracle_y, fs, max_lag=256):
    """Remove DC offset, bulk delay, and a single constant gain error from model.

    Cross-correlates the two signals, shifts by the integer lag maximizing
    correlation, then rescales so the least-squares gain is 1. Returns both
    signals trimmed to a common length.

    The DC removal is done twice on purpose. De-meaning first keeps the
    correlation honest; de-meaning again after the trim is what makes the
    returned pair comparable, because the trim leaves each signal centred on
    its own window rather than the common one. A delay that drops a partial
    cycle off the front leaves a residual offset of a few times 1e-4 that no
    gain correction can remove, and that is larger than the tolerance a delay
    alignment is supposed to meet.
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
    lag = int(np.argmax(window)) - (max_lag - 1)

    if lag > 0:
        model = model[lag:]
        oracle_y = oracle_y[: model.size]
    elif lag < 0:
        oracle_y = oracle_y[-lag:]
        model = model[: oracle_y.size]

    model = model - model.mean()
    oracle_y = oracle_y - oracle_y.mean()

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
    """Amplitude of harmonics 1..orders relative to the fundamental, in dB.

    A harmonic at or above Nyquist is reported as absent, on the same -600 dB
    floor harmonic_ratio_db gives, for the same reason: a Goertzel at such a
    frequency does not read that harmonic, it reads whatever content aliases to
    it. The vector stays `orders` long either way so callers can index it by
    harmonic number.
    """
    f0 = fundamental_freq(x, fs)
    fund = goertzel_amplitude(x, fs, f0)
    profile = []
    for order in range(1, orders + 1):
        freq = _harmonic_frequencies(f0, order, fs)
        harm = 0.0 if freq is None else goertzel_amplitude(x, fs, freq)
        profile.append(20.0 * np.log10(max(harm / max(fund, 1e-30), 1e-30)))
    return np.array(profile)


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


# metric name -> (best, worst, weight). For harmonic_profile_corr best=1.0,
# scored as the error (1 - corr) so higher correlation scores higher.
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
            part = normalize_error(1.0 - metrics[key], 0.0, 1.0)
        else:
            part = normalize_error(metrics[key], best, worst)
        total += weight * part
        wsum += weight
    return 100.0 * total / wsum if wsum > 0.0 else 0.0


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
    impulse = np.zeros(IMPULSE_N)
    impulse[0] = 1.0
    for fc_ in DEFAULT_CUTOFFS:
        ref = reference_magnitude_db(fc_, 2.0, freqs)
        for os in DEFAULT_OVERSAMPLES:
            y = run_model(model, impulse, fc_, r, os, runfilters, workdir)
            if y is None:
                out["measured"][f"fc{fc_}_os{os}"] = {"error": "no finite output"}
                continue
            m = shape_metrics(measured_magnitude_db(y, freqs), ref, freqs)
            m["cutoff_hz"] = fc_
            m["oversample"] = os
            out["measured"][f"fc{fc_}_os{os}"] = m
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


if __name__ == "__main__":
    raise SystemExit(main())
