import inspect
import json
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
# because run_model always passes --float. @MODEL@ is the model the file is
# named for, because RunFilters writes every model and run_model picks its own
# out of the directory by that name.
_STUB_WRITES_OUTPUT = (
    "a = sys.argv\n"
    "v = lambda f: a[a.index(f) + 1]\n"
    # %.0f, the way BuildOutputFilename names the cutoff, not int().
    "tag = f'@MODEL@_c{float(v('-c')):.0f}_r{float(v('-r')):.2f}'\n"
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


_STUB_FAILS_AT = (
    # RunFilters exits 1 and writes nothing when it cannot finish a model, which
    # is the condition the robustness gate exists to catch. sys.argv is read
    # directly: the body below this block is what binds the name `a`, and a
    # check placed before it would fail on every call for the wrong reason.
    "_ARGS = sys.argv\n"
    "if (float(_ARGS[_ARGS.index('-c') + 1]),\n"
    "        int(_ARGS[_ARGS.index('-s') + 1])) in _FAIL_FOR:\n"
    "    sys.exit(1)\n"
)


def _stub_runfilters(tmp_path, sleep=0.0, exit_code=0, model="Stilson", fail_for=()):
    """A stand-in for RunFilters that writes the one file run_model goes looking for.

    `sleep` makes the stub overrun a deadline and `exit_code` makes it fail the way
    a rejected argument would, so the failure paths can be tested without a build.
    `model` names the file the stub writes, which is the only thing that tells
    run_model which of the per-model files in the directory is its own.
    `fail_for` is a tuple of (cutoff, oversample) operating points at which the
    stub exits nonzero and writes nothing, so a single case of the grid can be
    failed while the rest of it succeeds.

    The output is float32 and not 16-bit PCM because run_model reads with
    read_wav_float, which rejects PCM by design
    (test_read_wav_float_rejects_a_pcm16_file): a stub built on the `wave` module
    would hand run_model an audio-format-1 file and raise instead of returning a
    signal.
    """
    body = f"time.sleep({sleep})\n"
    if exit_code:
        body += f"sys.exit({exit_code})\n"
    else:
        if fail_for:
            body += f"_FAIL_FOR = {tuple(fail_for)!r}\n" + _STUB_FAILS_AT
        body += _STUB_WRITES_OUTPUT.replace("@MODEL@", model)
    script = tmp_path / f"runfilters_stub_{model}_{sleep}_{exit_code}_{len(fail_for)}.py"
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


def test_noise_floor_of_silence_reads_the_amplitude_epsilon():
    """Pinned to -600 dB because that is 20*log10(1e-30), the floor on the scale.

    It is the same at any record length, which is what makes it a scale statement
    rather than a length artifact: the epsilon is applied after spectrum_db
    normalizes, so nothing about the window's gain is left in it.
    """
    for n in (16384, 88200, 131072):
        assert fe.noise_floor_db(np.zeros(n), 44100) == pytest.approx(-600.0, abs=1e-9)


# 1308.5 puts the tone exactly half a bin from the nearest bin centre, which is
# where the uncorrected reading collapsed to 0.03 %: probing the H2 there put it a
# whole cycle away from the true H2, on a null of the projection.
_OFF_BIN_CASES = [
    (440.0, 131072),
    (1000.37, 88200),
    (1308.5 * 44100 / 131072, 131072),
]


@pytest.mark.parametrize("f_true,n", _OFF_BIN_CASES)
def test_thd_recovers_a_tone_placed_between_bins(f_true, n):
    """A tone off the FFT grid must still read a true 10 % THD and a -20 dB H2.

    Every other tone in this file is on the grid, because a whole number of
    periods over the record length puts it on a bin centre, so the suite was
    structurally unable to see this: the uncorrected code read 7.03 % here.
    """
    fs = 44100
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f_true * t) + 0.1 * np.sin(2 * np.pi * 2 * f_true * t)
    assert fe.thd_percent(x, fs) == pytest.approx(10.0, rel=1e-2)
    assert fe.harmonic_ratio_db(x, fs, 2) == pytest.approx(-20.0, abs=0.1)


@pytest.mark.parametrize("f_true,n", _OFF_BIN_CASES)
def test_fundamental_freq_interpolates_the_peak_between_bins(f_true, n):
    """Within 0.05 bins, which is 3x the worst case measured over a full sweep.

    Half a bin, the loosest bound that could be asserted, is what the bin centre
    already satisfies, so it would not have caught anything.
    """
    fs = 44100
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f_true * t) + 0.1 * np.sin(2 * np.pi * 2 * f_true * t)
    assert abs(fe.fundamental_freq(x, fs) - f_true) <= 0.05 * fs / n


def test_spectrum_db_normalizes_a_full_scale_sine_to_zero_db():
    """A bin-centred full-scale sine reads 0 dB at every record length.

    Unnormalized it peaked at N/4, so the same tone read 22 dB higher on a
    131072-sample record than on a 4096-sample one and no two records could be
    compared. The three lengths span 32x, so this is the normalization itself and
    not a coincidence at one size.
    """
    fs = 44100
    for n in (4096, 16384, 131072):
        f0 = 41 * fs / 4096  # a bin centre at all three lengths
        t = np.arange(n) / fs
        _freqs, db = fe.spectrum_db(np.sin(2 * np.pi * f0 * t), fs)
        assert db.max() == pytest.approx(0.0, abs=1e-6), n


def test_noise_floor_db_reads_the_injected_white_noise_level():
    """A known sigma must read back within 1.5 dB, at two different lengths.

    The expectation is the analytic one for the recipe: a real DFT bin of white
    noise is complex Gaussian with mean square sigma**2*sum(win**2), so its
    magnitude is Rayleigh and the median of the record is
    sqrt(ln 2)*sigma*sqrt(sum(win**2))/(sum(win)/2). The length cancels, which is
    the claim worth testing: the two lengths are 3 dB apart under the old scale
    and 0.09 dB apart here.
    """
    fs, sigma = 44100, 1e-3
    for n in (44100, 88200):
        rng = np.random.default_rng(20260929)
        t = np.arange(n) / fs
        # A tone the floor must not mistake for the floor itself.
        x = np.sin(2 * np.pi * 997 * t) + sigma * rng.standard_normal(n)
        win = np.hanning(n)
        expected = 20 * np.log10(
            np.sqrt(np.log(2)) * sigma * np.sqrt((win ** 2).sum()) / (0.5 * win.sum())
        )
        assert fe.noise_floor_db(x, fs) == pytest.approx(expected, abs=1.5), n


def test_thd_percent_ignores_harmonics_that_would_fold_back_into_the_band():
    """A harmonic above Nyquist does not exist, so it is not summed.

    A Goertzel at 30 kHz on a 44100 Hz record does not read 30 kHz, it reads the
    content aliased to 14.1 kHz, and the 14.1 kHz tone here is read as though it
    were a third harmonic: 10.0 % of real H2 reads as 14.14 % of distortion.
    """
    fs, n = 44100, 44100
    t = np.arange(n) / fs
    # 1000, 2000 and 14100 Hz are all bin centres at n=44100, so this is not a
    # peak-interpolation case: the 14100 Hz tone is simply unrelated content.
    x = (
        np.sin(2 * np.pi * 10000 * t)
        + 0.1 * np.sin(2 * np.pi * 20000 * t)
        + 0.1 * np.sin(2 * np.pi * 14100 * t)
    )
    assert fe.fundamental_freq(x, fs) == pytest.approx(10000.0, rel=1e-6)
    assert fe.thd_percent(x, fs) == pytest.approx(10.0, rel=1e-2)
    assert fe.harmonic_ratio_db(x, fs, 2) == pytest.approx(-20.0, abs=0.01)


def test_harmonic_ratio_db_calls_an_above_nyquist_harmonic_absent():
    """Reported as absent, not as whatever aliased into the requested frequency."""
    fs, n = 44100, 44100
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 10000 * t) + 0.1 * np.sin(2 * np.pi * 20000 * t)
    # Order 3 is 30 kHz and order 5 is 50 kHz, both past fs/2 = 22050 Hz.
    assert fe.harmonic_ratio_db(x, fs, 3) == pytest.approx(-600.0, abs=1e-9)
    assert fe.harmonic_ratio_db(x, fs, 5) == pytest.approx(-600.0, abs=1e-9)
    assert fe.harmonic_ratio_db(x, fs, 2) == pytest.approx(-20.0, abs=0.01)


def test_goertzel_amplitude_rejects_an_empty_signal():
    """An empty record divides by n, so it has to be a caller error, not -nan."""
    with pytest.raises(ValueError):
        fe.goertzel_amplitude(np.zeros(0), 44100, 1000.0)


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


def test_reference_magnitude_is_zero_db_at_dc_for_zero_k():
    d = fe.reference_magnitude_db(1000.0, 0.0, np.array([1e-3, 1.0, 10.0]))
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    # 10 Hz is 0.01*fc, not DC. The cascade is 1/((1+x**2)**2) with x = f/fc, so
    # the low-frequency departure is -17.37*x**2 dB to leading order: -0.0017 dB
    # here, and exactly what the reference returns. Pinning it to 1e-3 would be
    # pinning a 4-pole rolloff to 0, which is not a filter property.
    assert d[2] == pytest.approx(0.0, abs=1e-2)


def test_reference_matches_reference_module():
    from reference import moog_ladder_linear as mll
    freqs = np.logspace(1, np.log10(0.4 * 44100), 500)
    assert np.allclose(
        fe.reference_magnitude_db(2000.0, 2.0, freqs),
        mll.magnitude_db(4, 44100.0, 2000.0, 2.0, freqs),
    )


def test_reference_peak_gain_rises_with_k():
    freqs = np.logspace(1, np.log10(0.4 * 44100), 4000)
    # magnitude_db is uncompensated, so its DC gain is 1/(1+k) and the passband
    # itself falls by 6 dB per doubling of k. The raw band maximum is therefore
    # not monotonic: k=0 is a flat 0 dB passband at -0.0017 dB, which sits above
    # the -3.72 dB resonant peak of k=1. The resonance is the peak above the
    # passband, and it rises strictly: 0.0, 2.30, 7.80, 15.54 dB.
    humps = []
    for k in (0.0, 1.0, 2.0, 3.0):
        curve = fe.reference_magnitude_db(1000.0, k, freqs)
        humps.append((curve - curve[0]).max())
    assert all(b > a for a, b in zip(humps, humps[1:]))


def test_best_r_for_target_finds_injected_optimum():
    target = np.zeros(100)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    curves = [target + 6.0, target, target + 3.0, target + 9.0, target + 12.0]
    r, err, idx = fe.best_r_for_target(curves, grid, target)
    assert idx == 1
    assert r == grid[1]
    assert err == pytest.approx(0.0, abs=1e-9)


def test_measured_magnitude_interpolates_onto_log_freqs():
    n = 8192
    freqs = np.logspace(1, np.log10(0.4 * 44100), 200)
    y = np.zeros(n)
    y[0] = 1.0
    d = fe.measured_magnitude_db(y, freqs)
    assert d.shape == freqs.shape
    assert np.all(np.isfinite(d))
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def _real_reference():
    freqs = np.logspace(1, np.log10(0.4 * 44100), 2000)
    return freqs, fe.reference_magnitude_db(1000.0, 0.0, freqs)


def test_shape_metrics_perfect_match_is_all_zero():
    freqs, ref = _real_reference()
    m = fe.shape_metrics(ref.copy(), ref, freqs)
    assert m["magnitude_rms_db_error"] == pytest.approx(0.0, abs=1e-9)
    assert m["cutoff_3db_error_cents"] == pytest.approx(0.0, abs=1e-6)
    assert m["passband_gain_error_db"] == pytest.approx(0.0, abs=1e-9)
    assert m["stopband_slope_error_db_per_oct"] == pytest.approx(0.0, abs=1e-6)
    assert m["peak_gain_error_db"] == pytest.approx(0.0, abs=1e-9)
    assert m["peak_freq_error_cents"] == pytest.approx(0.0, abs=1e-6)


def test_shape_metrics_detect_1db_passband_lift():
    freqs, ref = _real_reference()
    m = fe.shape_metrics(ref + 1.0, ref, freqs)
    assert m["passband_gain_error_db"] == pytest.approx(1.0, abs=0.05)
    assert m["magnitude_rms_db_error"] > 0.0
    # Pins the scale of the RMS: a sum of squares instead of a mean of squares
    # would read 2000 and a missing sqrt would read 44.7.
    assert m["magnitude_rms_db_error"] == pytest.approx(1.0, abs=0.01)


def test_cross_freq_selects_the_falling_cutoff_at_the_pipeline_operating_point():
    # Task 12 scores the pipeline at k=2, and a resonant ladder crosses -3 dB
    # twice: once rising up the skirt, because the DC gain is 1/(1+k) = -9.54 dB
    # and the peak only reaches -1.74 dB, and once falling past the real cutoff.
    # Only the falling one is the cutoff.
    freqs = np.logspace(1, np.log10(0.4 * 44100), 2000)
    ref = fe.reference_magnitude_db(1000.0, 2.0, freqs)
    assert ref[0] < -3.0 < ref.max()
    assert fe._cross_freq(freqs, ref, -3.0) == pytest.approx(1055.5, abs=10.0)
    m = fe.shape_metrics(ref.copy(), ref, freqs)
    assert m["cutoff_3db_error_cents"] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(m["passband_gain_error_db"])


def test_cross_freq_rejects_a_curve_starting_exactly_on_the_target():
    # Landing on the target at the first bin is not a crossing: there is no
    # approach from one side, so the only honest answer is that the curve was
    # already there.
    freqs = np.logspace(1, 3, 200)
    db = -3.0 - np.log2(freqs / freqs[0])
    assert db[0] == pytest.approx(-3.0)
    assert np.all(np.diff(db) < 0.0)
    assert np.isnan(fe._cross_freq(freqs, db, -3.0))


def test_shape_metrics_passband_stays_finite_at_a_low_cutoff_with_resonance():
    # The passband window is half the cutoff, so a cutoff read low enough puts
    # the window below the measured band and the median has nothing to read.
    freqs = np.logspace(np.log10(50.0), np.log10(0.4 * 44100), 2000)
    ref = fe.reference_magnitude_db(100.0, 2.0, freqs)
    m = fe.shape_metrics(ref.copy(), ref, freqs)
    assert np.isfinite(m["passband_gain_error_db"])
    assert m["passband_gain_error_db"] == pytest.approx(0.0, abs=1e-9)


def test_shape_metrics_charges_a_flat_model_for_the_missing_resonant_peak():
    # A flat model's band maximum is its DC gain, so a plain argmax reads the
    # missing peak as a near-perfect one. The peak has to be measured above the
    # passband, the way the reference-peak test does it.
    freqs = np.logspace(1, np.log10(0.4 * 44100), 2000)
    ref = fe.reference_magnitude_db(1000.0, 2.0, freqs)
    hump = float((ref - ref[0]).max())
    assert hump > 6.0
    m = fe.shape_metrics(np.zeros_like(ref), ref, freqs)
    assert m["peak_gain_error_db"] == pytest.approx(-hump, abs=0.05)
    best, worst, _ = fe.LINEAR_WEIGHTS["peak_gain_error_db"]
    assert fe.normalize_error(m["peak_gain_error_db"], best, worst) == 0.0


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
    # A failed measurement is a failure, not a pass: a NaN on the heaviest axis
    # has to score strictly below a clean 0.0 there, or the bound above is the
    # only thing being tested.
    clean = dict(metrics, magnitude_rms_db_error=0.0)
    assert fe.score_linear(metrics) < fe.score_linear(clean)


def test_linear_score_treats_signed_errors_symmetrically():
    def with_cents(value):
        return {
            "magnitude_rms_db_error": 0.0, "cutoff_3db_error_cents": value,
            "passband_gain_error_db": 0.0, "stopband_slope_error_db_per_oct": 0.0,
            "peak_gain_error_db": 0.0, "peak_freq_error_cents": 0.0,
        }

    # A model 400 cents LOW is exactly as wrong as one 400 cents HIGH, and a
    # one-sided clamp scored both as perfect.
    assert fe.score_linear(with_cents(-400.0)) == pytest.approx(
        fe.score_linear(with_cents(400.0))
    )
    assert fe.score_linear(with_cents(-50.0)) > fe.score_linear(with_cents(-400.0))


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
    """Alignment must not be able to erase a genuine spectral difference.

    The value is pinned, not merely floored, because align_signals has exactly two
    free parameters -- an integer lag and a constant gain -- and neither can
    dissolve a tone that the reference does not contain:

      * The 1 kHz and 3 kHz tones run a whole number of cycles over the record, so
        they are orthogonal. The least-squares gain is
        dot(ref + 0.2*sin3k, ref) / dot(ref, ref), and orthogonality makes the
        sin3k term vanish, leaving g exactly 1.0 (measured: 1.0 bit-exact).
      * The residual is therefore exactly the 0.2*sin(3kHz) term, RMS
        0.2/sqrt(2) = 0.141421, over a reference RMS of 1/sqrt(2) = 0.707107, so
        the NRMSE is exactly 0.2 (measured: 0.19999999999999987).

    A worst-case alignment that drifted the lag by one sample would raise this, not
    lower it. The old `> 0.01` guard therefore let an alignment erase 95% of a real
    difference and still pass; rel=0.05 allows at most 5% erasure.
    """
    fs = 44100
    t = np.arange(44100) / fs
    ref = np.sin(2 * np.pi * 1000 * t)
    other = ref + 0.2 * np.sin(2 * np.pi * 3000 * t)
    a, b = fe.align_signals(other, ref, fs)
    assert fe.time_domain_nrmse(a, b) == pytest.approx(0.2, rel=0.05)


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


def test_harmonic_profile_floors_harmonics_above_nyquist():
    """A harmonic past Nyquist is absent, not an aliased reading.

    At fs=44100 a 6 kHz fundamental reaches Nyquist at the 4th harmonic
    (24000 > 22050), and 6 kHz sits on an exact bin over 88200 samples, as do its
    harmonics, so the first three read their true relative amplitudes and the rest
    must report the -600 dB floor. Without the guard a Goertzel at 24000 Hz reads
    whatever content aliases there -- 24000 - 44100 = -20100 Hz -- which is a
    number about a harmonic, not about silence.
    """
    fs = 44100
    t = np.arange(88200) / fs
    x = (np.sin(2 * np.pi * 6000 * t)
         + 0.1 * np.sin(2 * np.pi * 12000 * t)
         + 0.5 * np.sin(2 * np.pi * 18000 * t))
    prof = fe.harmonic_profile_db(x, fs, orders=10)
    assert len(prof) == 10
    assert prof[0] == pytest.approx(0.0, abs=0.01)
    assert prof[1] == pytest.approx(-20.0, abs=0.01)
    assert prof[2] == pytest.approx(20.0 * np.log10(0.5), abs=0.01)
    assert np.all(prof[3:] == -600.0)


NONLINEAR_METRIC_KEYS = {
    "spectral_distance_db", "time_domain_nrmse", "thd_delta_db",
    "harmonic_profile_corr", "thd_model_percent", "thd_oracle_percent",
    "harmonic_profile_db_model", "harmonic_profile_db_oracle",
}


def test_nonlinear_metrics_returns_a_json_serializable_contract():
    """nonlinear_metrics is what Task 12 consumes, so its shape is a contract.

    The two profiles are lists of plain floats and the two THD readings are
    floats, so the whole dict has to survive json.dumps -- including with
    allow_nan=False, which is what actually rejects a NaN or an infinity rather
    than emitting the `NaN` literal that is not valid JSON.
    """
    fs = 44100
    t = np.arange(88200) / fs
    oracle = np.sin(2 * np.pi * 1000 * t)
    model = oracle + 0.1 * np.sin(2 * np.pi * 2000 * t)
    metrics = fe.nonlinear_metrics(model, oracle, fs)

    assert set(metrics) == NONLINEAR_METRIC_KEYS
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics
    assert np.isfinite(metrics["thd_delta_db"])
    assert -1.0 < metrics["harmonic_profile_corr"] < 1.0
    assert len(metrics["harmonic_profile_db_model"]) == 10
    assert len(metrics["harmonic_profile_db_oracle"]) == 10


def test_nonlinear_score_bounded_and_orders_correctly():
    near = {"spectral_distance_db": 0.5, "time_domain_nrmse": 0.01,
            "thd_delta_db": 0.5, "harmonic_profile_corr": 0.99}
    far = {"spectral_distance_db": 12.0, "time_domain_nrmse": 0.9,
           "thd_delta_db": 18.0, "harmonic_profile_corr": 0.2}
    assert fe.score_nonlinear(near) > fe.score_nonlinear(far)
    assert 0.0 <= fe.score_nonlinear(far) <= 100.0


def test_nonlinear_score_corr_axis_discriminates():
    """The corr axis must not grant full credit to every correlation.

    (Controller-verified brief defect: scoring corr via a swapped
    normalize_error(corr, 1.0, 0.0) hits the `value <= best` clamp and
    returns 1.0 for ANY correlation <= 1.0, deadweighting the 0.15 axis.)
    """
    near = {"spectral_distance_db": 2.0, "time_domain_nrmse": 0.1,
            "thd_delta_db": 2.0, "harmonic_profile_corr": 0.99}
    far = {"spectral_distance_db": 2.0, "time_domain_nrmse": 0.1,
           "thd_delta_db": 2.0, "harmonic_profile_corr": 0.2}
    assert fe.score_nonlinear(near) > fe.score_nonlinear(far)
    # The brief wrote this case as dict(far, harmonic_profile_corr=-0.5) and
    # asserted a total of exactly 0, which needs the other three axes at their
    # worst values. far's 2.0/0.1/2.0 are mid-range against 12.0/0.9/18.0, so
    # that total is 73.6, not 0. Pinning the worst values here is what actually
    # isolates the corr axis: an anticorrelation must contribute exactly nothing.
    worst = {"spectral_distance_db": 12.0, "time_domain_nrmse": 0.9,
             "thd_delta_db": 18.0}
    negative = dict(worst, harmonic_profile_corr=-0.5)
    assert fe.score_nonlinear(negative) == 0.0
    # ...and a perfect correlation on an otherwise-worst model contributes
    # exactly its 0.15 weight share, no more.
    assert fe.score_nonlinear(dict(worst, harmonic_profile_corr=1.0)) == 15.0


def test_run_model_rejects_fc_beyond_half_sample_rate(tmp_path):
    """The collector's own grid must stay inside the open band, or every sweep raises."""
    with pytest.raises(ValueError):
        fe.run_model(
            "Dummy", np.zeros(64), 30000.0, 0.0, 0, "ignored-binary", str(tmp_path)
        )


def test_collect_linear_returns_key_schema(tmp_path):
    result = fe.collect_linear("Dummy", _stub_runfilters(tmp_path, model="Dummy"), str(tmp_path))
    for key in ("requested", "measured", "linear_score", "score_parts"):
        assert key in result, key
    # The grid is part of the schema: every cutoff crossed with every oversample
    # factor, or a model is scored on a subset of what it was asked for.
    assert set(result["measured"]) == {
        f"fc{fc}_os{os}"
        for fc in fe.DEFAULT_CUTOFFS
        for os in fe.DEFAULT_OVERSAMPLES
    }
    assert set(result["score_parts"]) == set(fe.LINEAR_WEIGHTS) | {"n_cases"}
    # The stub emits digital silence, so nothing about it is measurable: the
    # score is 0.0 rather than None, which is what shows the aggregation ran
    # over every case instead of the collector bailing out.
    assert result["linear_score"] == 0.0
    assert result["flagged"] == []
    assert result["flag_reason"] == ""


def test_collect_linear_reports_the_case_count_behind_every_metric(tmp_path):
    """A mean over however many values survived says nothing about how many that was.

    At fc=5000 the reference's -3 dB point is near 5.28 kHz, so its stopband
    window starts at 4x that, about 21.1 kHz, while the measurement grid stops
    at 0.4*fs = 17.64 kHz. The window is empty, the slope is NaN, and the
    nanmean quietly averages the other two cutoffs. Counting is what makes that
    visible: 4 of 6 cases here, not 6.
    """
    result = fe.collect_linear("Dummy", _stub_runfilters(tmp_path, model="Dummy"), str(tmp_path))
    counts = result["score_parts"]["n_cases"]
    assert counts["stopband_slope_error_db_per_oct"] == 4
    assert counts["magnitude_rms_db_error"] == 6
    # Silence never falls through -3 dB, so that axis has no cases at all and the
    # aggregate over it is NaN rather than a number that reads like a measurement.
    assert counts["cutoff_3db_error_cents"] == 0
    assert np.isnan(result["score_parts"]["cutoff_3db_error_cents"])


def test_collect_linear_flags_the_cases_the_model_failed(tmp_path, monkeypatch):
    """The design doc's robustness gate: a failed run is named, not averaged away.

    Calibration is stubbed so the failure lands on the shape sweep and only on
    the oversampled operating point; RunFilters is not involved, because the
    collector sees the same None either way.
    """
    def fake_run_model(model, signal, cutoff, resonance, oversample, *a, **k):
        return None if oversample == 4 else np.zeros(2048)

    monkeypatch.setattr(fe, "calibrate_resonance", lambda *a, **k: (0.5, {}))
    monkeypatch.setattr(fe, "run_model", fake_run_model)
    result = fe.collect_linear("Dummy", "unused", str(tmp_path))

    assert result["flagged"] == [f"fc{fc}_os4" for fc in fe.DEFAULT_CUTOFFS]
    assert "fc1000.0_os4" in result["flag_reason"]
    assert "3 of 6" in result["flag_reason"]
    # The failed cases are still in the record, and still say why.
    assert result["measured"]["fc1000.0_os4"] == {"error": "no finite output"}
    assert result["measured"]["fc1000.0_os0"] != {"error": "no finite output"}
    # The surviving half is still scored, as a diagnostic. The flag, not the
    # score, is what keeps the model out of the ranking.
    assert result["linear_score"] is not None
    assert result["score_parts"]["n_cases"]["magnitude_rms_db_error"] == 3


def test_collect_linear_excludes_unstable_resonance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fe,
        "calibrate_resonance",
        lambda *a, **k: (None, {"reason": "no finite output at any resonance"}),
    )
    result = fe.collect_linear("Dummy", _stub_runfilters(tmp_path, model="Dummy"), str(tmp_path))
    assert result["linear_score"] is None
    assert "reason" in result
    # Failing every one of the 21 calibration resonances is non-finite output on
    # every condition, which is the gate's condition and not an excuse to sit
    # at status ok with no score and no explanation.
    assert result["flagged"] == ["calibrate_resonance"]
    assert "calibrate_resonance" in result["flag_reason"]


def test_collect_linear_drives_the_shape_sweep_with_an_impulse(tmp_path, monkeypatch):
    """Silence measures silence: a flat -600 dB curve scores nothing about shape.

    The sweep's excitation is pinned here because nothing else about a run
    distinguishes it from a drive of zeros, and a zero drive is the failure mode
    this catches: every fc/os would report a perfect flat response in the
    passband and score its way into the ranking.
    """
    drives = []

    def fake_run_model(model, signal, cutoff, resonance, oversample, *a, **k):
        drives.append(np.asarray(signal, dtype=np.float64))
        return np.zeros(2048)

    # Calibration is the other half of the collector's driving, and is measured
    # by its own test; stubbing it keeps this to the shape sweep.
    monkeypatch.setattr(fe, "calibrate_resonance", lambda *a, **k: (0.5, {}))
    monkeypatch.setattr(fe, "run_model", fake_run_model)
    fe.collect_linear("Dummy", "unused", str(tmp_path))

    assert len(drives) == len(fe.DEFAULT_CUTOFFS) * len(fe.DEFAULT_OVERSAMPLES)
    for signal in drives:
        assert signal.shape == (fe.IMPULSE_N,)
        assert np.count_nonzero(signal) == 1
        assert signal[0] == 1.0


def test_collect_nonlinear_pairs_every_case_with_the_oracle_at_one_operating_point(
    tmp_path, monkeypatch
):
    """Model at the calibrated r, oracle at the absolute k that calibration matched.

    The pairing is the whole claim of this axis: a model measured at one
    operating point against an oracle measured at another measures the
    difference between them, not the model's error. SWEEP_N is dropped so the
    case grid is affordable here; the record length is a property of
    nonlinear_metrics, which has its own tests.

    The case keys are built from the loop variables, so they read correctly
    whatever the loop body was handed. That is why the arguments are recorded
    and compared against the key rather than the key being read off the loop.
    """
    monkeypatch.setattr(fe, "SWEEP_N", 4096)
    model_calls = []
    oracle_calls = []

    def fake_run_model(model, signal, cutoff, resonance, oversample, *a, **k):
        model_calls.append(
            (cutoff, resonance, oversample, float(np.max(np.abs(signal))))
        )
        return np.asarray(signal, dtype=np.float64)

    class Oracle:
        def process(self, x, fs, fc, k, n):
            oracle_calls.append((fc, k, n, float(np.max(np.abs(x)))))
            return 0.5 * np.asarray(x, dtype=np.float64)

    monkeypatch.setattr(fe, "run_model", fake_run_model)
    out = fe.collect_nonlinear("Dummy", "unused", str(tmp_path), Oracle(), 0.4)

    expected = {
        f"fc{fc}_lvl{lvl}_os{os}"
        for fc in fe.DEFAULT_CUTOFFS
        for lvl in fe.LEVELS_DBFS
        for os in fe.DEFAULT_OVERSAMPLES
    }
    assert set(out["cases"]) == expected
    assert len(expected) == 30
    assert not any("error" in v for v in out["cases"].values())
    assert out["flagged"] == []
    assert 0.0 <= out["nonlinear_score"] <= 100.0

    assert len(model_calls) == 30
    assert {c[1] for c in model_calls} == {0.4}
    # One oracle record per (cutoff, level), reused across the oversample
    # factors: oversampling is a property of the model under test, not of the
    # reference it is being compared with.
    assert len(oracle_calls) == len(fe.DEFAULT_CUTOFFS) * len(fe.LEVELS_DBFS)
    assert {(c[1], c[2]) for c in oracle_calls} == {(2.0, fe.ORDER)}

    # Every (cutoff, oversample) cell was driven once per level, which is what
    # makes the key's fc/os claim mean anything.
    for fc in fe.DEFAULT_CUTOFFS:
        for oversample in fe.DEFAULT_OVERSAMPLES:
            assert sum(
                1 for c in model_calls if c[0] == fc and c[2] == oversample
            ) == len(fe.LEVELS_DBFS)

    # And one case in full: the key, the model arguments behind it, and the
    # oracle record it was compared against. The drive amplitude appears in
    # both, which is the proof that the two were the same test signal and not
    # merely two signals at the same frequency.
    drive = 10.0 ** (-6.0 / 20.0)
    assert "fc1000.0_lvl-6.0_os4" in out["cases"]
    assert (1000.0, 0.4, 4, drive) in model_calls
    assert (1000.0, 2.0, fe.ORDER, drive) in oracle_calls


def test_collect_nonlinear_flags_the_cases_the_model_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "SWEEP_N", 4096)

    def fake_run_model(model, signal, cutoff, resonance, oversample, *a, **k):
        if (cutoff, oversample) == (1000.0, 4):
            return None
        return np.asarray(signal, dtype=np.float64)

    class Oracle:
        def process(self, x, fs, fc, k, n):
            return 0.5 * np.asarray(x, dtype=np.float64)

    monkeypatch.setattr(fe, "run_model", fake_run_model)
    out = fe.collect_nonlinear("Dummy", "unused", str(tmp_path), Oracle(), 0.4)

    assert out["flagged"] == [
        f"fc1000.0_lvl{lvl}_os4" for lvl in fe.LEVELS_DBFS
    ]
    assert "5 of 30" in out["flag_reason"]
    assert out["cases"]["fc1000.0_lvl-6.0_os4"] == {"error": "no finite output"}
    # The 25 survivors still score, and the aggregate is over them.
    assert 0.0 <= out["nonlinear_score"] <= 100.0


def test_collect_nonlinear_drives_a_hard_switched_square_at_the_requested_level(
    tmp_path, monkeypatch
):
    """The drive is 10**(level/20) of a 440 Hz square, and the axis rests on it.

    10**(level/10) is the power-ratio reading and gives twice the amplitude;
    nothing downstream of collect_nonlinear would notice, because the model and
    the oracle would both be driven by the wrong signal and still agree.
    """
    monkeypatch.setattr(fe, "SWEEP_N", 4410)  # 44 whole cycles at 440 Hz over 44100
    seen = []

    def fake_run_model(model, signal, cutoff, resonance, oversample, *a, **k):
        seen.append(np.asarray(signal, dtype=np.float64))
        return np.asarray(signal, dtype=np.float64)

    class Oracle:
        def process(self, x, fs, fc, k, n):
            return np.asarray(x, dtype=np.float64)

    monkeypatch.setattr(fe, "run_model", fake_run_model)
    fe.collect_nonlinear("Dummy", "unused", str(tmp_path), Oracle(), 0.4)

    by_level = {round(20.0 * np.log10(np.max(np.abs(x))), 9): x for x in seen}
    assert set(by_level) == {round(lvl, 9) for lvl in fe.LEVELS_DBFS}

    x = by_level[round(-6.0, 9)]
    amp = 10.0 ** (-6.0 / 20.0)
    assert amp == pytest.approx(0.501187, abs=1e-6)
    assert np.max(np.abs(x)) == pytest.approx(amp)
    # Two levels only: a hard switch, not a sine and not a triangle.
    assert set(np.unique(np.abs(x))) == {0.0, amp}
    # 440 Hz over a 44100 Hz sample rate is one period every 100.2273 samples,
    # so 4410 samples hold 44 whole cycles and 88 zero crossings. The record
    # starts on a crossing and np.sign(0) is 0, which is one of the counted
    # changes and is why this is 88 rather than 87. A record too short to hold
    # whole cycles would split a crossing between two samples and still count it,
    # so the divisor is what makes this a measurement of the frequency.
    assert np.count_nonzero(np.diff(np.sign(x))) == 88


def test_collect_selfoscillation_reports_ringing_per_resonance(tmp_path, monkeypatch):
    def fake_run_model(model, signal, cutoff, resonance, oversample, *a, **k):
        if resonance == 1.0:
            return None  # diverged: no finite output to measure a tail from
        y = np.zeros(len(signal))
        if resonance >= 0.9:
            # Still swinging at -43 dBFS after the 12000-sample burst ended. A
            # constant tail would sit exactly on the -60 dBFS threshold, which
            # tests the comparison operator instead of the ringing.
            tail = np.arange(len(signal) - len(signal) // 2)
            y[len(signal) // 2:] = 1e-2 * np.sin(2 * np.pi * 50.0 * tail / fe.SAMPLE_RATE)
        return y

    monkeypatch.setattr(fe, "run_model", fake_run_model)
    out = fe.collect_selfoscillation("Dummy", "unused", str(tmp_path))

    # The return holds one record per resonance plus the gate's own two keys, so
    # a reader has to separate them rather than index it as a flat mapping.
    per_r = {k: v for k, v in out.items() if k not in ("flagged", "flag_reason")}
    assert set(per_r) == {str(r) for r in fe.SELFOSC_RESONANCES}
    assert out["0.5"]["ringing"] is False
    assert out["0.5"]["tail_rms_db"] < -60.0
    assert out["0.9"]["ringing"] is True
    assert out["0.9"]["tail_rms_db"] > -60.0
    # A diverged run is not a model that self-oscillates. Reading it as ringing
    # made "at least one model never settles" unfalsifiable from the flag alone,
    # so divergence is reported as not ringing, flagged, and named.
    assert out["1.0"]["ringing"] is False
    assert out["1.0"]["tail_rms_db"] is None
    assert out["1.0"]["note"] == "model diverged"
    assert out["1.0"]["flagged"] is True
    assert out["flagged"] == ["r1.00"]
    assert "r1.00" in out["flag_reason"]
    # Only the diverged resonance is flagged: 0.9 genuinely rings.
    assert out["0.9"].get("flagged", False) is False


def test_collect_selfoscillation_says_so_when_the_output_is_shorter_than_the_burst(
    tmp_path, monkeypatch
):
    """A tail that was never measured is not a tail that did not ring.

    len(y) below the burst index leaves y[burst:] empty, the RMS of nothing is
    NaN, and NaN > -60 is False, so the model was silently reported as settled
    on a measurement that was never taken.
    """
    monkeypatch.setattr(fe, "run_model", lambda *a, **k: np.zeros(4096))
    out = fe.collect_selfoscillation("Dummy", "unused", str(tmp_path))

    record = out["0.5"]
    assert record["ringing"] is False
    assert record["tail_rms_db"] is None
    assert record["note"] == "output shorter than burst (len 4096)"
    # Not flagged: a short output is a measurement that was not taken, which is
    # not the gate's condition of non-finite output.
    assert out["flagged"] == []


def test_main_writes_one_json_per_model_with_the_shared_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "RUNFILTERS_EXE", "unused")
    monkeypatch.setattr(
        fe,
        "collect_linear",
        lambda *a: {
            "requested": {"resonance": 0.4},
            "measured": {},
            "linear_score": 50.0,
            "score_parts": {},
            "flagged": [],
            "flag_reason": "",
        },
    )
    monkeypatch.setattr(
        fe,
        "collect_nonlinear",
        lambda *a: {"cases": {}, "nonlinear_score": 25.0, "flagged": [], "flag_reason": ""},
    )
    monkeypatch.setattr(
        fe,
        "collect_selfoscillation",
        lambda *a: {"0.5": {"ringing": False, "tail_rms_db": -200.0}, "flagged": []},
    )
    out_dir = tmp_path / "run1"
    assert fe.main(["--models", "Stilson, Improved", "--out-dir", str(out_dir)]) == 0

    for name in ("Stilson", "Improved"):
        record = json.loads((out_dir / "metrics" / f"{name}.json").read_text())
        assert record["model"] == name
        assert record["status"] == "ok"
        assert record["flagged"] == []
        assert record["linear"]["linear_score"] == 50.0
        assert record["nonlinear"]["nonlinear_score"] == 25.0
        assert record["selfosc"]["0.5"]["ringing"] is False
        # The envelope is what makes two runs comparable later, so every file
        # carries the same provenance keys the driver was handed.
        assert set(record) >= {"args", "generated_at", "git_head", "software"}
        assert set(record["software"]) == {"python", "numpy", "scipy"}
        assert record["args"]["models"] == "Stilson, Improved"
        assert record["args"]["out_dir"] == str(out_dir)


def test_main_flags_a_model_whose_run_failed(tmp_path, monkeypatch):
    """End to end: a failing operating point reaches the record as status flagged.

    Nothing here is stubbed below the binary. The stand-in RunFilters exits 1 at
    fc=5000 with 4x oversampling, so the collectors see a real failed run at one
    grid point, and the driver has to name it and drop the model from the
    ranking rather than averaging over the survivors and reporting ok.
    """
    monkeypatch.setattr(fe, "SWEEP_N", 4096)  # keep the oracle loop affordable
    monkeypatch.setattr(
        fe,
        "RUNFILTERS_EXE",
        _stub_runfilters(tmp_path, model="Stilson", fail_for=[(5000.0, 4)]),
    )
    out_dir = tmp_path / "run3"
    assert fe.main(["--models", "Stilson", "--out-dir", str(out_dir)]) == 0

    record = json.loads((out_dir / "metrics" / "Stilson.json").read_text())
    assert record["status"] == "flagged"
    assert record["flagged"], "a failed run must name the offending parameters"
    assert "fc5000.0_os4" in record["flagged"]
    assert all(key.startswith("fc5000.0") for key in record["flagged"]), record["flagged"]
    assert "fc5000.0_os4" in record["linear"]["flag_reason"]
    assert record["linear"]["measured"]["fc5000.0_os4"] == {"error": "no finite output"}
    # Partial scores survive as diagnostics; the flag is the removal mechanism.
    assert record["linear"]["linear_score"] is not None


def test_main_records_an_unknown_model_name_and_keeps_sweeping(tmp_path, monkeypatch):
    """--models is a name list, and a typo is not a filter that diverges.

    An unrecognized name used to be swept like any other: 21 calibrations that
    all fail, a null resonance, and three self-oscillation runs that each
    reported "model diverged" for a model that does not exist, all of it under
    status ok. The name is now checked against MODEL_NAMES before anything is
    run, the bad name gets its own error record, and the good ones still run.
    """
    monkeypatch.setattr(fe, "RUNFILTERS_EXE", "unused")
    monkeypatch.setattr(
        fe,
        "collect_linear",
        lambda model, *a: {
            "requested": {"resonance": 0.4},
            "measured": {},
            "linear_score": 50.0,
            "score_parts": {},
            "flagged": [],
            "flag_reason": "",
        },
    )
    monkeypatch.setattr(
        fe, "collect_nonlinear", lambda *a: {"cases": {}, "nonlinear_score": 25.0,
                                              "flagged": [], "flag_reason": ""}
    )
    monkeypatch.setattr(fe, "collect_selfoscillation", lambda *a: {"flagged": []})
    out_dir = tmp_path / "run4"

    code = fe.main(["--models", "Stilson,improved", "--out-dir", str(out_dir)])

    assert code != 0, "a run that scored only 1 of the 2 models asked for is not a clean run"
    bad = json.loads((out_dir / "metrics" / "improved.json").read_text())
    assert bad["model"] == "improved"
    assert bad["status"] == "error: unknown model 'improved'"
    assert bad["linear"] is None and bad["nonlinear"] is None and bad["selfosc"] is None
    assert bad["flagged"] == []
    good = json.loads((out_dir / "metrics" / "Stilson.json").read_text())
    assert good["status"] == "ok"
    assert good["linear"]["linear_score"] == 50.0


def test_an_unknown_model_name_cannot_write_outside_the_metrics_directory(tmp_path, monkeypatch):
    """The name is user input, so it cannot be pasted straight into a path."""
    monkeypatch.setattr(fe, "RUNFILTERS_EXE", "unused")
    out_dir = tmp_path / "run5"
    code = fe.main(["--models", "../../escaped", "--out-dir", str(out_dir)])
    assert code != 0
    written = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*.json"))
    assert written == ["metrics/.._.._escaped.json"], written
    assert not (tmp_path / "escaped.json").exists()


def test_main_records_a_collector_failure_and_still_writes_the_json(tmp_path, monkeypatch):
    """One model's failure is that model's record, not the end of the sweep.

    The JSON is written inside the same try's other side, so a model that
    raises still leaves a file behind: a driver that swallowed the exception
    without writing would leave a hole where a ranking entry belongs.
    """

    def boom(*a, **k):
        raise RuntimeError("no binary")

    monkeypatch.setattr(fe, "RUNFILTERS_EXE", "unused")
    monkeypatch.setattr(fe, "collect_linear", boom)
    monkeypatch.setattr(fe, "collect_nonlinear", lambda *a: {})
    monkeypatch.setattr(fe, "collect_selfoscillation", lambda *a: {})
    out_dir = tmp_path / "run2"

    assert fe.main(["--models", "Stilson", "--out-dir", str(out_dir)]) == 0
    record = json.loads((out_dir / "metrics" / "Stilson.json").read_text())
    assert record["model"] == "Stilson"
    assert record["status"] == "error: RuntimeError: no binary"
    assert record["linear"] is None
    assert record["nonlinear"] is None
    assert record["selfosc"] is None
    assert record["flagged"] == []


def test_main_runs_every_model_name_the_enum_lists():
    """The default sweep covers all twelve, and only names RunFilters actually writes."""
    assert fe.MODEL_NAMES == tuple(fe.FILTER_NAMES)
    assert len(fe.MODEL_NAMES) == 12
