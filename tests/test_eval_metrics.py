import inspect
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


# The body a well-behaved RunFilters would run: name the output the way
# BuildOutputFilename does (example/run-filters.cpp:55) and write a float32 WAV,
# because run_model always passes --float.
_STUB_WRITES_OUTPUT = (
    "a = sys.argv\n"
    "v = lambda f: a[a.index(f) + 1]\n"
    # %.0f, the way BuildOutputFilename names the cutoff, not int().
    "tag = f'Stilson_c{float(v('-c')):.0f}_r{float(v('-r')):.2f}'\n"
    "if int(v('-s')):\n"
    "    tag += f'_os{int(v('-s'))}x'\n"
    "d = Path(v('-o'))\n"
    "d.mkdir(parents=True, exist_ok=True)\n"
    "raw = b'\\x00' * 2048\n"
    "d.joinpath(tag + '.wav').write_bytes(b''.join([\n"
    "    b'RIFF', struct.pack('<I', 36 + len(raw)), b'WAVE',\n"
    "    b'fmt ', struct.pack('<I', 16),\n"
    "    struct.pack('<HHIIHH', 3, 1, 44100, 176400, 4, 32),\n"
    "    b'data', struct.pack('<I', len(raw)), raw]))\n"
)


def _stub_runfilters(tmp_path, sleep=0.0, exit_code=0):
    """A stand-in for RunFilters that writes the one file run_model goes looking for.

    `sleep` makes the stub overrun a deadline and `exit_code` makes it fail the way
    a rejected argument would, so the failure paths can be tested without a build.
    """
    body = f"time.sleep({sleep})\n"
    body += f"sys.exit({exit_code})\n" if exit_code else _STUB_WRITES_OUTPUT
    script = tmp_path / f"runfilters_stub_{sleep}_{exit_code}.py"
    script.write_text(
        f"#!{sys.executable}\n"
        "import struct, sys, time\n"
        "from pathlib import Path\n"
        f"{body}"
    )
    script.chmod(0o755)
    return str(script)


def _unfaded_sweep(n, fs, f0, f1, amplitude):
    """sine_sweep without the tail fade, for tests that compare against it."""
    t = np.arange(n) / float(fs)
    k = np.log(f1 / f0)
    return amplitude * np.sin(2.0 * np.pi * f0 * (np.exp(k * t) - 1.0) / k)


def test_sine_sweep_fades_to_zero_at_the_end():
    """A raised cosine over the last 1024 samples, so the sweep does not just stop.

    Ending mid-cycle at -0.16 would leave a step discontinuity at the end of the
    file, which is a broadband transient for the windowed FFT and THD tasks that
    consume this signal.
    """
    n, fs, f0, f1 = 44100, 44100, 20.0, 20000.0
    x = fe.sine_sweep(n, fs, f0, f1, 0.5)
    unfaded = _unfaded_sweep(n, fs, f0, f1, 0.5)

    fade = 1024
    assert np.array_equal(x[:n - fade], unfaded[:n - fade])
    assert x[-1] == 0.0
    # The ratio recovers the envelope wherever the carrier is away from a zero
    # crossing, where the quotient is meaningless.
    tail = slice(n - fade, n)
    env = x[tail] / unfaded[tail]
    loud = np.abs(unfaded[tail]) > 0.05
    raised_cosine = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade) / (fade - 1)))
    assert np.allclose(env[loud], raised_cosine[loud], atol=1e-9)


def test_sine_sweep_fade_does_not_change_the_zero_crossings():
    """A positive envelope cannot move a zero crossing, so the two agree exactly.

    The fade does change the final sample, from -0.16 to 0, but the sign change
    still lands between the same pair of samples, so even the index array matches
    and not just its length.
    """
    n, fs = 44100, 44100
    faded = fe.sine_sweep(n, fs, 20.0, 20000.0, 0.5)
    unfaded = _unfaded_sweep(n, fs, 20.0, 20000.0, 0.5)
    crossings = np.where(np.diff(np.sign(faded)))[0]
    assert np.array_equal(crossings, np.where(np.diff(np.sign(unfaded)))[0])
    assert len(crossings) > 10
    assert np.max(np.abs(faded)) <= 0.5 + 1e-9


def test_read_wav_rejects_a_data_chunk_shorter_than_it_declares(tmp_path):
    """A truncated file must not be read as though the missing tail were silence."""
    p = tmp_path / "short.wav"
    fe.write_wav(p, np.zeros(64, dtype=np.float32))
    p.write_bytes(p.read_bytes()[:-16])
    with pytest.raises(ValueError):
        fe.read_wav(p)
    with pytest.raises(ValueError):
        fe.read_wav_float(p)


def test_run_model_resolves_a_cutoff_that_is_not_a_whole_number(tmp_path):
    """RunFilters names output with %.0f, so the wrapper has to round, not floor.

    At 1000.6 Hz the binary writes Stilson_c1001_..., and a wrapper that floored
    would search a directory the binary never created.
    """
    stub = _stub_runfilters(tmp_path)
    for cutoff, named in ((1000.6, "1001"), (1001.5, "1002"), (999.5, "1000")):
        # A workdir per case: flooring 1001.5 gives 1001, which is the name the
        # 1000.6 case already used.
        workdir = tmp_path / f"c{named}"
        tag = f"Stilson_c{named}_r0.50"
        y = fe.run_model("Stilson", np.zeros(64), cutoff, 0.5, 0, stub, str(workdir))
        assert y is not None, cutoff
        assert (workdir / f"{tag}_out" / f"{tag}.wav").exists(), cutoff
        assert not (workdir / f"Stilson_c{int(cutoff)}_r0.50_out").exists(), cutoff


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


def test_run_model_returns_none_when_runfilters_times_out(tmp_path):
    """A run that overruns its deadline is void, like one that exits nonzero.

    The same stub completes inside a generous deadline and yields its file, so the
    None below can only come from the timeout.
    """
    fast = _stub_runfilters(tmp_path)
    slow = _stub_runfilters(tmp_path, sleep=2.0)
    signal = fe.steady_sine(2048, 44100, 440.0, 0.5)

    assert fe.run_model("Stilson", signal, 1000.0, 0.5, 0, slow, str(tmp_path), timeout=0.5) is None
    assert fe.run_model("Stilson", signal, 1000.0, 0.5, 0, fast, str(tmp_path), timeout=30) is not None


def test_run_model_accepts_every_supported_oversample_factor(tmp_path):
    """The _os<n>x suffix is half the output-naming contract and was uncovered."""
    stub = _stub_runfilters(tmp_path)
    for oversample in fe.SUPPORTED_OVERSAMPLES:
        y = fe.run_model("Stilson", np.zeros(64), 1000.0, 0.5, oversample, stub, str(tmp_path))
        assert y is not None, oversample
        assert y.shape == (512,), oversample


def test_run_model_rejects_an_unsupported_oversample_factor(tmp_path):
    """RunFilters accepts 0, 2, 4 or 8, so anything else is a caller error."""
    for oversample in (1, 3, 5, 6, 7, 16, -1):
        with pytest.raises(ValueError):
            fe.run_model("Stilson", np.zeros(64), 1000.0, 0.5, oversample, "ignored-binary", str(tmp_path))


def test_run_model_rejects_resonance_outside_the_unit_interval(tmp_path):
    """Resonance is 0..1 here, which is absolute feedback k = 4r in [0, 4]."""
    stub = _stub_runfilters(tmp_path)
    signal = np.zeros(64)
    for resonance in (0.0, 0.5, 1.0):
        assert fe.run_model("Stilson", signal, 1000.0, resonance, 0, stub, str(tmp_path)) is not None
    for resonance in (-0.1, 1.0001, 2.0, 5.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            fe.run_model("Stilson", signal, 1000.0, resonance, 0, "ignored-binary", str(tmp_path))


def test_run_model_default_timeout_is_sixty_seconds():
    """The deadline is a decision, not a detail, so the default is pinned."""
    assert inspect.signature(fe.run_model).parameters["timeout"].default == 60


def test_run_model_rejects_a_cutoff_outside_the_open_band(tmp_path):
    """A NaN is caught here rather than by int() further down, which says nothing useful."""
    for cutoff in (0.0, -100.0, fe.SAMPLE_RATE / 2, fe.SAMPLE_RATE / 2 + 1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="must be in"):
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


def test_run_model_returns_none_when_runfilters_fails(tmp_path):
    """A nonzero exit voids the run, even when a file of the right name is present.

    RunFilters exits 1 when it cannot finish, and leaves whatever it had already
    written. The only file with the expected name here is the one put there
    beforehand, so a failed run must not be able to report stale data.
    """
    stub = _stub_runfilters(tmp_path, exit_code=1)
    outdir = tmp_path / "Stilson_c1000_r0.50_out"
    outdir.mkdir(parents=True)
    fe.write_wav(outdir / "Stilson_c1000_r0.50.wav", np.zeros(32, dtype=np.float32))
    assert (
        fe.run_model("Stilson", np.zeros(64), 1000.0, 0.5, 0, stub, str(tmp_path)) is None
    )


def test_run_model_raises_when_the_binary_is_missing(tmp_path):
    """A missing binary is a setup error, not a filter failure.

    Returning None here would zero out every model in the sweep and look like a
    result, so the error is left to propagate.
    """
    with pytest.raises(OSError):
        fe.run_model("Stilson", np.zeros(64), 1000.0, 0.0, 0, str(tmp_path / "nope"), str(tmp_path))


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


def test_harmonic_ratio_db_reads_a_known_h2_and_rejects_the_absent_orders():
    """A -20 dB H2 must read -20 dB, and an absent H3 must stay far below it.

    thd_percent sums harmonics against each other; this is the per-harmonic
    level a ranking shows, so it is the number that has to be right. The absent
    orders are the other half of it: a projection that leaked would report a
    level for a harmonic the signal does not contain.
    """
    fs = 44100
    t = np.arange(88200) / fs
    x = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 2000 * t)
    assert fe.harmonic_ratio_db(x, fs, 2) == pytest.approx(-20.0, rel=1e-6)
    assert fe.harmonic_ratio_db(x, fs, 3) < -100.0


def test_fundamental_freq_finds_the_tone():
    fs = 44100
    t = np.arange(88200) / fs
    assert fe.fundamental_freq(np.sin(2 * np.pi * 1234 * t), fs) == pytest.approx(
        1234.0, rel=1e-3
    )


def test_noise_floor_of_silence_is_as_low_as_possible():
    assert fe.noise_floor_db(np.zeros(16384), 44100) < -300.0


def test_thd_below_16bit_floor_survives_float32_but_not_pcm16(tmp_path):
    """The exact bug that broke the old suite: a -100 dBFS tone must still show THD.

    Round-tripping through 16-bit quantizes this away. Through float32 it
    survives. This is the direct guard on why --float exists.

    The level has to be sub-LSB in 16-bit for that contrast to exist at all. At
    the -60 dBFS of 0.001 the fundamental is 32.8 LSB and the H2 is 3.3 LSB, and
    16-bit resolves the harmonic: it reads 10.31 %, not less than the 10.0 % that
    is really there. The floor only bites below 0.001/32768 per harmonic, so this
    drives 1e-5, where the whole record truncates to zero and THD reads 0.0.
    """
    fs = 44100
    t = np.arange(88200) / fs
    x = 1e-5 * (
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
