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


def _write_input_float(path, samples, fs=44100):
    """Write a mono 32-bit IEEE float WAV (audio format 3), like WriteWavFileFloat."""
    pcm = samples.astype("<f4").tobytes()
    data_size = len(pcm)
    header = b"".join([
        b"RIFF", struct.pack("<I", 36 + data_size), b"WAVE",
        b"fmt ", struct.pack("<I", 16),
        struct.pack("<HHIIHH", 3, 1, fs, fs * 4, 4, 32),
        b"data", struct.pack("<I", data_size),
    ])
    with open(path, "wb") as f:
        f.write(header)
        f.write(pcm)


def _read_fmt(path):
    """Return (audio_format_tag, channels, sample_rate, byte_rate, block_align, bits)."""
    with open(path, "rb") as f:
        data = f.read()
    at = data.index(b"fmt ")
    # "fmt " id, then the 4-byte chunk size, then the 16-byte fmt payload.
    tag, channels, rate, byterate, blockalign, bits = struct.unpack_from(
        "<HHIIHH", data, at + 8
    )
    return tag, channels, rate, byterate, blockalign, bits


def _first_wav(outdir):
    files = sorted(Path(outdir).glob("*.wav"))
    assert files, "no output wav produced"
    return files[0]


def _read_samples(path, bits):
    # Python's wave module rejects format 3, so the chunks are read by hand.
    with open(path, "rb") as f:
        data = f.read()
    raw = data[data.index(b"data") + 8:]
    if bits == 32:
        return np.frombuffer(raw, "<f4").astype(np.float64)
    return np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0


def _run(src, outdir, extra=()):
    # RunFilters does not create its output directory, so the harness does.
    Path(outdir).mkdir(parents=True, exist_ok=True)
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
    tag, ch, rate, byterate, blockalign, bits = _read_fmt(_first_wav(out))
    assert tag == 3
    assert bits == 32
    assert ch == 1
    assert rate == 44100
    assert byterate == 44100 * 4
    assert blockalign == 4


@needs_build
def test_default_output_is_still_pcm16(tmp_path):
    src = tmp_path / "in.wav"
    _write_input(src, 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(44100) / 44100.0))
    out = tmp_path / "pcm16"
    r = _run(src, out)
    assert r.returncode == 0, r.stderr
    tag, _, _, _, _, bits = _read_fmt(_first_wav(out))
    assert tag == 1
    assert bits == 16


@needs_build
@pytest.mark.parametrize("extra,expected_tag", [(["--float"], 3), ([], 1)])
def test_sub_lsb_tone_survives_float32_but_not_pcm16(tmp_path, extra, expected_tag):
    """A tone below one 16-bit LSB survives float32 and is destroyed by 16-bit.

    This is the whole reason the flag exists. A -100 dBFS tone is well below the
    1/32768 resolution of a 16-bit file, so the integer writer quantises the entire
    signal to digital silence while the float writer keeps it. The input has to be
    float32 for the tone to exist at all, which ReadWavFile already supports.
    """
    src = tmp_path / "in.wav"
    amp = 1e-5  # -100 dBFS, a third of one 16-bit LSB
    tone = amp * np.sin(2 * np.pi * 100.0 * np.arange(88200) / 44100.0)
    _write_input_float(src, tone)
    out = tmp_path / ("t" if extra else "p")
    r = _run(src, out, extra)
    assert r.returncode == 0, r.stderr
    got = _first_wav(out)
    tag, _, _, _, _, bits = _read_fmt(got)
    assert tag == expected_tag
    y = _read_samples(got, bits)
    rms = float(np.sqrt(np.mean(y[44100:] ** 2)))
    # 100 Hz is a decade below the 1000 Hz cutoff, so the tone has to come out at
    # roughly its input level; the exact gain is per-model (0.88x to 1.57x).
    in_rms = float(np.sqrt(np.mean(tone[44100:] ** 2)))
    if expected_tag == 3:
        assert 0.5 * in_rms < rms < 2.0 * in_rms
    else:
        assert rms == 0.0
