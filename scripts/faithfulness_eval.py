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
