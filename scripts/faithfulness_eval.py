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
import struct
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
