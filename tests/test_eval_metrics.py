import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import faithfulness_eval as fe  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNFILTERS = REPO / "build" / "RunFilters"
needs_build = pytest.mark.skipif(
    not RUNFILTERS.exists(), reason="build/RunFilters not built"
)


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
    assert len(np.where(np.diff(np.sign(x)))[0]) > 10


def test_sine_sweep_is_logarithmic():
    """The frequency rises, and it rises on a log scale, not a linear one.

    Crossing density is 2*f, so the density in two equal-length windows a fixed
    fraction of the sweep apart tells the two apart: exponential for a log sweep,
    linear for a straight one. Windows are equal in time rather than in count
    because a log sweep front-loads its crossings, so counting them in halves
    compares 0.9 s of sweep against 0.1 s of it.
    """
    n, fs, f0, f1 = 44100, 44100, 100.0, 5000.0
    x = fe.sine_sweep(n, fs, f0, f1, 0.5)
    crossings = np.where(np.diff(np.sign(x)))[0]
    quarter = n // 4
    early = crossings[crossings < quarter]
    late = crossings[crossings >= 3 * quarter]
    # Window centres are 0.125 s and 0.875 s apart, i.e. 0.75 of the sweep.
    expected = (f1 / f0) ** 0.75
    ratio = len(late) / len(early)
    assert 0.5 * expected < ratio < 2.0 * expected, (ratio, expected)


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


def test_write_wav_labels_float32_as_audio_format_three(tmp_path):
    """RunFilters' ReadWavFile dispatches on the format tag, not the byte width.

    Float bytes under a PCM tag are decoded as int32, which rescales the input by
    2**-31, so the tag has to be 3 for the filters to see the amplitude we wrote.
    """
    p = tmp_path / "f32.wav"
    fe.write_wav(p, np.full(64, 0.25, dtype=np.float32))
    blob = p.read_bytes()
    tag, channels, rate, _byte_rate, _align, bits = struct.unpack_from(
        "<HHIIHH", blob, blob.index(b"fmt ") + 8
    )
    assert (tag, channels, rate, bits) == (3, 1, 44100, 32)
    assert np.frombuffer(blob[blob.index(b"data") + 8:], "<f4")[0] == pytest.approx(0.25)


def test_read_wav_float_rejects_a_pcm16_file(tmp_path):
    """A format mismatch must be loud, not silently reinterpreted as float32."""
    p = tmp_path / "pcm16.wav"
    fe.write_wav(p, np.zeros(16, dtype=np.float32), int16=True)
    with pytest.raises(ValueError):
        fe.read_wav_float(p)


@needs_build
def test_run_model_reads_the_model_named_output_file(tmp_path):
    n = 4096
    signal = fe.steady_sine(n, 44100, 440.0, 0.5)
    y = fe.run_model("Krajeski", signal, 1000.0, 0.5, 0, str(RUNFILTERS), str(tmp_path))
    assert y is not None
    assert y.dtype == np.float64
    assert y.shape == (n,)
    written = tmp_path / "Krajeski_c1000_r0.50_out" / "Krajeski_c1000_r0.50.wav"
    assert written.exists()
    assert np.array_equal(y, fe.read_wav_float(written))


@needs_build
def test_run_model_does_not_confuse_hyperion_with_hyperion_legacy(tmp_path):
    """The model name is a prefix of another model's name, so matching must be exact."""
    signal = fe.steady_sine(4096, 44100, 440.0, 0.5)
    y = fe.run_model("Hyperion", signal, 1000.0, 0.5, 0, str(RUNFILTERS), str(tmp_path))
    outdir = tmp_path / "Hyperion_c1000_r0.50_out"
    assert np.array_equal(y, fe.read_wav_float(outdir / "Hyperion_c1000_r0.50.wav"))
    legacy = fe.read_wav_float(outdir / "HyperionLegacy_c1000_r0.50.wav")
    assert not np.allclose(y, legacy)


@needs_build
def test_run_model_matches_the_cpp_filter_name_spelling_exactly(tmp_path):
    """A name that is not the file-name spelling is a failed run, not a wrong model."""
    signal = fe.steady_sine(2048, 44100, 440.0, 0.5)
    assert (
        fe.run_model("krajeski", signal, 1000.0, 0.5, 0, str(RUNFILTERS), str(tmp_path))
        is None
    )


@needs_build
def test_run_model_writes_a_float32_input_wav(tmp_path):
    signal = fe.steady_sine(2048, 44100, 440.0, 0.5)
    fe.run_model("Stilson", signal, 1000.0, 0.5, 0, str(RUNFILTERS), str(tmp_path))
    src = tmp_path / "Stilson_c1000_r0.50_in.wav"
    blob = src.read_bytes()
    tag, _channels, rate, _byte_rate, _align, bits = struct.unpack_from(
        "<HHIIHH", blob, blob.index(b"fmt ") + 8
    )
    assert (tag, rate, bits) == (3, 44100, 32)


def test_run_model_rejects_a_cutoff_outside_the_open_band(tmp_path):
    for cutoff in (0.0, -100.0, fe.SAMPLE_RATE / 2, fe.SAMPLE_RATE / 2 + 1.0):
        with pytest.raises(ValueError):
            fe.run_model("Stilson", np.zeros(64), cutoff, 0.0, 0, "ignored-binary", str(tmp_path))


@needs_build
def test_run_model_returns_none_for_non_finite_output(tmp_path):
    """A diverging model is a per-model failure, not a failure of the whole run.

    MusicDSP goes non-finite on a wildly overdriven input while the other eleven
    models stay finite, so the guard has to be applied to the selected model's
    file and not to the run.
    """
    drive = 1e3 * fe.steady_sine(4096, 44100, 440.0, 1.0)
    assert fe.run_model("MusicDSP", drive, 1000.0, 0.5, 0, str(RUNFILTERS), str(tmp_path)) is None
    assert fe.run_model("Stilson", drive, 1000.0, 0.5, 0, str(RUNFILTERS), str(tmp_path)) is not None


@needs_build
def test_run_model_returns_none_when_runfilters_fails(tmp_path):
    """A nonzero exit voids the run, even when a file of the right name is present.

    RunFilters rejects resonance 2.0 with exit code 1 and writes nothing of its
    own, so the only file RunFilters_c1000_r2.00.wav in the directory is the one
    put there beforehand. A failed run must not be able to report stale data.
    """
    outdir = tmp_path / "Stilson_c1000_r2.00_out"
    outdir.mkdir(parents=True)
    fe.write_wav(outdir / "Stilson_c1000_r2.00.wav", np.zeros(32, dtype=np.float32))
    assert (
        fe.run_model("Stilson", np.zeros(64), 1000.0, 2.0, 0, str(RUNFILTERS), str(tmp_path))
        is None
    )


def test_run_model_raises_when_the_binary_is_missing(tmp_path):
    """A missing binary is a setup error, not a filter failure.

    Returning None here would zero out every model in the sweep and look like a
    result, so the error is left to propagate.
    """
    with pytest.raises(OSError):
        fe.run_model("Stilson", np.zeros(64), 1000.0, 0.0, 0, str(tmp_path / "nope"), str(tmp_path))
