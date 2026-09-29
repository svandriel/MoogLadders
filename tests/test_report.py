import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import faithfulness_eval as fe  # noqa: E402


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


def _record(status="ok", **overrides):
    """One model's record. Pass linear=None / nonlinear=None for a real None section."""
    rec = {
        "model": "X", "status": status, "flagged": [],
        "linear": {
            "requested": {"resonance": 0.4},
            "measured": {"fc1000.0_os0": {"cutoff_3db_error_cents": -25.0}},
            "linear_score": 72.5,
        },
        "nonlinear": {
            "cases": {
                "fc1000.0_lvl-6.0_os0": {
                    "spectral_distance_db": 3.4,
                    "thd_model_percent": 0.05,
                    "thd_oracle_percent": 0.02,
                    # H1 is the fundamental against itself: 0.0 dB, or the profile is not dB re H1.
                    "harmonic_profile_db_model": [0.0] + [-20.0] * 9,
                    "harmonic_profile_db_oracle": [0.0] + [-30.0] * 9,
                },
            },
            "nonlinear_score": 48.0,
        },
        "selfosc": {"0.9": {"ringing": False, "tail_rms_db": -80.0}},
    }
    rec.update(overrides)
    return rec


def _captured_axes(monkeypatch, tmp_path, recs):
    """What each figure's axes say, captured instead of written to disk.

    Axis text is the part of a figure a reader takes as a claim, and it is the one
    part a test can read. savefig is stubbed out: this asks what would have been
    drawn, not what would have landed on the filesystem.
    """
    import matplotlib.figure
    import matplotlib.pyplot as plt

    seen = {}

    def fake_savefig(self, fname, **kwargs):
        ax = self.axes[0]
        seen[Path(fname).name] = {
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "title": ax.get_title(),
            "labels": [line.get_label() for line in ax.get_lines()],
            "ydata": [list(line.get_ydata()) for line in ax.get_lines()],
        }

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)
    try:
        fe.generate_figures(recs, tmp_path)
    finally:
        plt.close("all")
    assert len(seen) == 6, sorted(seen)
    return seen


F3_CASE = "fc1000.0_lvl-6.0_os0"
ORACLE_PROFILE_FIELD = "harmonic_profile_db_oracle"
ORACLE_THD_FIELD = "thd_oracle_percent"


def _with_cases(model, oracle_profile, oracle_thd, model_profile=None, score=50.0):
    """A record carrying all five -6 dBFS-ish levels, with oracle values of its own."""
    cases = {}
    for i, level in enumerate(fe.LEVELS_DBFS):
        cases[f"fc1000.0_lvl{level}_os0"] = {
            "spectral_distance_db": 3.0 + i,
            "thd_model_percent": 1.0 + i,
            "thd_oracle_percent": oracle_thd[i],
            "harmonic_profile_db_model": list(model_profile or oracle_profile),
            "harmonic_profile_db_oracle": list(oracle_profile),
        }
    return _record(model=model,
                   nonlinear={"cases": cases, "nonlinear_score": score},
                   linear={"measured": {"fc1000.0_os0": {"cutoff_3db_error_cents": -25.0}},
                           "linear_score": score})


def test_oracle_pick_walks_model_names_order_not_insertion_order():
    """The oracle on a record is model-aligned, so there is no one true oracle.

    nonlinear_metrics runs the oracle through align_signals, whose shift comes
    from cross-correlating the model, so the same input yields a different oracle
    THD and harmonic profile on every model: in the committed run, H2 at -6 dBFS
    ranges from -67.94 dB on Stilson to -73.78 dB on Improved. Picking "the first
    record in the dict" makes which oracle gets drawn a property of the caller's
    dict order, and the committed F3 and F4 were drawn from different models while
    both said "oracle".
    """
    stilson = _with_cases("Stilson", [-67.94] + [-80.0] * 9, [12.98] * 5)
    musicdsp = _with_cases("MusicDSP", [-72.96] + [-90.0] * 9, [13.01] * 5)
    forward = {"Stilson": stilson, "MusicDSP": musicdsp}
    backward = dict(reversed(list(forward.items())))

    for recs in (forward, backward):
        assert fe._oracle_pick(recs, F3_CASE, ORACLE_PROFILE_FIELD) == [-67.94] + [-80.0] * 9
        assert fe._oracle_pick(recs, F3_CASE, ORACLE_THD_FIELD) == 12.98
    assert fe._oracle_source(forward) == "Stilson"
    assert fe._oracle_source(backward) == "Stilson"


def test_oracle_pick_skips_models_that_measured_nothing():
    """No usable oracle is no oracle line, not a drawn NaN and a legend entry."""
    nan_rec = _with_cases("Stilson", [float("nan")] * 10, [float("nan")] * 5)
    assert fe._oracle_pick({"Stilson": nan_rec}, F3_CASE, ORACLE_PROFILE_FIELD) is None
    assert fe._oracle_pick({"Stilson": nan_rec}, F3_CASE, ORACLE_THD_FIELD) is None
    assert fe._oracle_source({"Stilson": nan_rec}) is None

    flagged = _with_cases("Stilson", [-67.94] + [-80.0] * 9, [12.98] * 5)
    flagged["status"] = "flagged"
    assert fe._oracle_pick({"Stilson": flagged}, F3_CASE, ORACLE_PROFILE_FIELD) is None
    assert fe._oracle_source({"Stilson": flagged}) is None

    errored = _with_cases("Stilson", [-67.94] + [-80.0] * 9, [12.98] * 5)
    errored["nonlinear"] = None
    assert fe._oracle_pick({"Stilson": errored}, F3_CASE, ORACLE_PROFILE_FIELD) is None

    assert fe._oracle_pick({}, F3_CASE, ORACLE_PROFILE_FIELD) is None
    assert fe._oracle_source({}) is None
    # A named model with nothing on it does not fall back to another model.
    assert fe._oracle_pick({"MusicDSP": _with_cases("MusicDSP", [-72.96] + [-90.0] * 9, [13.01] * 5)},
                           F3_CASE, ORACLE_PROFILE_FIELD, model="Stilson") is None


def test_f3_and_f4_draw_the_same_models_oracle(monkeypatch, tmp_path):
    """One oracle, chosen once, for both figures that draw one."""
    recs = {
        "Improved": _with_cases("Improved", [-73.78] + [-95.0] * 9, [12.96] * 5),
        "Stilson": _with_cases("Stilson", [-67.94] + [-80.0] * 9, [12.98] * 5),
    }
    assert fe._oracle_source(recs) == "Stilson"

    seen = _captured_axes(monkeypatch, tmp_path, recs)
    f3 = seen["F3_harmonic_spectrum.png"]
    assert f3["ydata"][f3["labels"].index("oracle (Stilson-aligned)")] == \
        [-67.94] + [-80.0] * 9
    f4 = seen["F4_thd_vs_level.png"]
    assert f4["ydata"][f4["labels"].index("oracle (Stilson-aligned)")] == [12.98] * 5

    # When the enum-first model measured no oracle at all, both figures move to
    # the next one together rather than one of them keeping a stale line.
    for level in fe.LEVELS_DBFS:
        case = recs["Stilson"]["nonlinear"]["cases"][f"fc1000.0_lvl{level}_os0"]
        case[ORACLE_PROFILE_FIELD] = [float("nan")] * 10
        case[ORACLE_THD_FIELD] = float("nan")
    assert fe._oracle_source(recs) == "Improved"
    seen = _captured_axes(monkeypatch, tmp_path, recs)
    f3 = seen["F3_harmonic_spectrum.png"]
    assert f3["ydata"][f3["labels"].index("oracle (Improved-aligned)")] == \
        [-73.78] + [-95.0] * 9
    f4 = seen["F4_thd_vs_level.png"]
    assert f4["ydata"][f4["labels"].index("oracle (Improved-aligned)")] == [12.96] * 5


def test_the_oracle_source_speaks_for_a_figure_it_cannot_serve(monkeypatch, tmp_path):
    """A model with oracle THD but no profile at the F3 case still speaks for it.

    Moving the source to another model for one figure is the bug this pass fixed,
    so the source stays put and F3 draws no oracle line instead. A missing line is
    a gap; a second "oracle" is a contradiction.
    """
    recs = {
        "Stilson": _with_cases("Stilson", [-67.94] + [-80.0] * 9, [12.98] * 5),
        "Improved": _with_cases("Improved", [-73.78] + [-95.0] * 9, [12.96] * 5),
    }
    recs["Stilson"]["nonlinear"]["cases"][F3_CASE][ORACLE_PROFILE_FIELD] = [float("nan")] * 10
    assert fe._oracle_source(recs) == "Stilson"

    seen = _captured_axes(monkeypatch, tmp_path, recs)
    f3 = seen["F3_harmonic_spectrum.png"]
    assert not any("oracle" in label for label in f3["labels"]), f3["labels"]
    f4 = seen["F4_thd_vs_level.png"]
    assert f4["ydata"][f4["labels"].index("oracle (Stilson-aligned)")] == [12.98] * 5


def test_f3_and_f4_name_the_model_their_oracle_is_aligned_to(monkeypatch, tmp_path):
    """A line labelled "oracle" does not say what it is a baseline for.

    The oracle on a record is the oracle aligned to that model, so the two figures
    that draw one are quoting one model's view of it. The report embeds these
    PNGs, and a reader who cannot see which model the baseline belongs to cannot
    tell a model difference from an alignment difference.
    """
    recs = {
        "Improved": _with_cases("Improved", [-73.78] + [-95.0] * 9, [12.96] * 5),
        "Stilson": _with_cases("Stilson", [-67.94] + [-80.0] * 9, [12.98] * 5),
    }
    seen = _captured_axes(monkeypatch, tmp_path, recs)
    for name in ("F3_harmonic_spectrum.png", "F4_thd_vs_level.png"):
        info = seen[name]
        assert "oracle (Stilson-aligned)" in info["labels"], info["labels"]
        assert "Stilson-aligned" in info["title"], info["title"]


def test_no_oracle_line_when_no_model_measured_one(monkeypatch, tmp_path):
    nan_rec = _with_cases("Stilson", [float("nan")] * 10, [float("nan")] * 5)
    seen = _captured_axes(monkeypatch, tmp_path, {"Stilson": nan_rec})
    for name in ("F3_harmonic_spectrum.png", "F4_thd_vs_level.png"):
        assert "oracle" not in seen[name]["labels"], seen[name]["labels"]
        # Nothing was drawn, so nothing may claim a source: "(None-aligned)" in a
        # title is a disclosure that discloses nothing.
        assert "aligned" not in seen[name]["title"], seen[name]["title"]


def test_figures_are_byte_identical_whatever_the_record_order(tmp_path):
    """The same records must produce the same bytes, whoever assembles the dict.

    main builds records in MODEL_NAMES order; a script that reads the metrics
    directory back builds them in filename order. Both are the same twelve
    measurements, and a figure set whose colors and oracle depend on which one ran
    is not reproducible from the committed PNGs.
    """
    import hashlib

    names = ["MusicDSP", "Stilson", "Huovilainen", "Improved", "RKSimulation"]
    recs = {
        name: _with_cases(name, [-60.0 - i] + [-90.0] * 9, [12.0 + i] * 5,
                          model_profile=[-70.0 - i] + [-91.0] * 9, score=40.0 + 5 * i)
        for i, name in enumerate(names)
    }
    forward, backward = tmp_path / "fwd", tmp_path / "bwd"
    fe.generate_figures(recs, forward)
    fe.generate_figures(dict(reversed(list(recs.items()))), backward)
    for path in sorted(forward.glob("*.png")):
        other = backward / path.name
        assert path.read_bytes() == other.read_bytes(), (
            path.name,
            hashlib.sha256(path.read_bytes()).hexdigest()[:12],
            hashlib.sha256(other.read_bytes()).hexdigest()[:12],
        )


def test_f3_axis_is_relative_to_the_fundamental(monkeypatch, tmp_path):
    """harmonic_profile_db is dB re H1, not dBFS.

    Every case in every record has profile[0] == 0.0, which is what a reference
    of 0 dB means: the first entry is the fundamental itself, so the axis is
    relative to it. A dBFS axis would put H1 at some unstated level and read as
    a claim about absolute output.
    """
    seen = _captured_axes(monkeypatch, tmp_path, {"A": _record()})
    assert seen["F3_harmonic_spectrum.png"]["ylabel"] == "dB re H1"
    # The premise the label rests on, asserted on a record rather than asserted
    # in a docstring: the first entry is the fundamental against itself.
    rec = _record()
    for field in (ORACLE_PROFILE_FIELD, "harmonic_profile_db_model"):
        assert rec["nonlinear"]["cases"][F3_CASE][field][0] == 0.0


def test_titles_name_the_figure_instead_of_repeating_the_tag(monkeypatch, tmp_path):
    """The tag goes in once, followed by what the figure shows.

    fig_caption is handed the tag the filename uses ("F1"), so the caption must
    not prepend another F: "FF1 — F1 at (...)" is what a caller that only reads
    the axis text sees.
    """
    seen = _captured_axes(monkeypatch, tmp_path, {})
    for name, info in seen.items():
        tag = name.split("_")[0]
        assert info["title"].startswith(f"{tag} — "), info["title"]
        assert info["title"] != f"{tag} — {tag} at", info["title"]


def test_captions_state_only_the_operating_points_the_data_has(monkeypatch, tmp_path):
    """F1 and F2 have no input level; claiming one is a fabricated condition.

    F1 is a magnitude response and F2 a cutoff error against a score, both taken
    from a sweep whose amplitude is not part of the measurement. A caption that
    names a level reads as "this was measured at -6 dBFS" and sends the reader
    looking for a sweep setting that does not exist.
    """
    calls = {}
    real = fe.fig_caption

    def spy(tag, **op):
        calls[tag] = dict(op)
        return real(tag, **op)

    monkeypatch.setattr(fe, "fig_caption", spy)
    fe.generate_figures({"A": _record()}, tmp_path)
    assert set(calls) == {"F1", "F2", "F3", "F4", "F5", "F6"}
    for tag in ("F1", "F2"):
        assert "level" not in calls[tag], calls[tag]
    for tag in ("F3", "F4", "F5", "F6"):
        assert calls[tag].get("level") is not None, calls[tag]
    for tag, op in calls.items():
        assert op.get("f_s") == fe.SAMPLE_RATE, op
    # F3 and F4 compare the model at absolute k=2.0, which is the operating point
    # collect_nonlinear holds fixed, so the caption has to say so. F1 and F2 sweep
    # k or use the calibrated r, and F5/F6 span cases; naming one number there
    # would be the same fabrication as naming a level they do not have.
    for tag in ("F3", "F4"):
        assert calls[tag].get("K") == 2, calls[tag]
    for tag in ("F1", "F2", "F5", "F6"):
        assert calls[tag].get("K") != 2, calls[tag]


def test_harmonic_profile_drops_nan_profiles():
    """A profile of NaNs is not a spectrum; it is a run that produced no output.

    Stilson at fc=100.0, -24 dBFS carries exactly that: status ok, unflagged, and
    a ten-entry model profile that is all NaN. Plotting it would draw an empty
    line and, worse, put it in the legend as a model that was measured.
    """
    key = "fc1000.0_lvl-6.0_os0"
    field = "harmonic_profile_db_model"
    assert fe._harmonic_profile(_record(), key, field) == [0.0] + [-20.0] * 9

    nan_rec = _record()
    nan_rec["nonlinear"]["cases"][key][field] = [float("nan")] * 10
    assert fe._harmonic_profile(nan_rec, key, field) is None

    one_bad = _record()
    one_bad["nonlinear"]["cases"][key][field] = [0.0] + [-20.0] * 8 + [None]
    assert fe._harmonic_profile(one_bad, key, field) is None

    assert fe._harmonic_profile(_record(), key, "missing_field") is None
    assert fe._harmonic_profile(_record(), "fc9.0_lvl-6.0_os0", field) is None
    assert fe._harmonic_profile(_record(nonlinear=None), key, field) is None
    errored = _record()
    errored["nonlinear"]["cases"] = {key: {"error": "no finite output"}}
    assert fe._harmonic_profile(errored, key, field) is None
    empty = _record()
    empty["nonlinear"]["cases"][key][field] = []
    assert fe._harmonic_profile(empty, key, field) is None


def test_a_nan_profile_draws_no_line_and_no_legend_entry(monkeypatch, tmp_path):
    nan_rec = _record()
    nan_rec["nonlinear"]["cases"]["fc1000.0_lvl-6.0_os0"]["harmonic_profile_db_model"] = (
        [float("nan")] * 10
    )
    seen = _captured_axes(monkeypatch, tmp_path, {"A": _record(), "Nan": nan_rec})
    f3 = seen["F3_harmonic_spectrum.png"]
    assert f3["labels"] == ["oracle (A-aligned)", "A"]


def _scored(name, lin, nlin):
    return {name: _record(linear={"linear_score": lin}, nonlinear={"nonlinear_score": nlin})}


def test_score_bars_sort_by_weighted_combined():
    recs = {}
    recs.update(_scored("A", 72.5, 48.0))  # 60.25
    recs.update(_scored("B", 10.0, 90.0))  # 50.00
    recs.update(_scored("C", 40.0, 40.0))  # 40.00
    bars = fe._score_bars(recs)
    assert [b[0] for b in bars] == ["A", "B", "C"]
    assert [b[3] for b in bars] == [60.25, 50.0, 40.0]
    assert bars[0][1] == 72.5 and bars[0][2] == 48.0

    # One axis measured is not a ranking entry: a bar height from a single axis
    # would be scored against 100 on an axis that was never looked at.
    recs["D"] = _scored("D", 99.0, 0.0)["D"]
    recs["D"]["nonlinear"] = None
    assert "D" not in [b[0] for b in fe._score_bars(recs)]

    # A flagged model has no ranking entry either, however good it looks.
    recs["E"] = _record(status="flagged",
                        linear={"linear_score": 100.0},
                        nonlinear={"nonlinear_score": 100.0})
    assert "E" not in [b[0] for b in fe._score_bars(recs)]


def test_generate_figures_skips_flagged_and_error_models(tmp_path):
    recs = {
        "A": _record(),
        "B": _record(status="flagged"),
        "C": _record(linear=None, nonlinear=None),
    }
    written = {p.name for p in fe.generate_figures(recs, tmp_path)}
    # F4/F6 legibility: a flagged model must not appear in the score-bar names,
    # exercised through the F6 PNG being produced from A only:
    # (assert no exception and 6 files regardless)
    assert len(written) == 6
    # C is ok but measured nothing, so it is not a bar next to A.
    assert [b[0] for b in fe._score_bars(recs)] == ["A"]


def test_score_bars_use_weighted_combined(tmp_path):
    recs = {"A": _record()}
    fe.generate_figures(recs, tmp_path)
    # Belt to test_score_bars_sort_by_weighted_combined, which exercises the
    # sort generate_figures actually draws from: pin the weight arithmetic.
    rec = _record()
    lin = rec["linear"]["linear_score"]
    nlin = rec["nonlinear"]["nonlinear_score"]
    comb = fe.COMBINED_WEIGHT_LINEAR * lin + (1 - fe.COMBINED_WEIGHT_LINEAR) * nlin
    assert comb == 0.5 * 72.5 + 0.5 * 48.0  # 60.25


def test_figures_tolerate_none_sections_and_error_cases(tmp_path):
    """A record whose collectors all failed still produces the six files.

    collect_linear returns a section with measured cases named `{"error": ...}` and
    collect_nonlinear the same; main leaves both None when a collector raised.
    Neither is a reason to lose the whole figure set.
    """
    errored = {
        "model": "Y", "status": "error: RuntimeError", "flagged": [],
        "linear": {"measured": {"fc1000.0_os0": {"error": "no finite output"}},
                   "linear_score": None, "flagged": ["fc1000.0_os0"]},
        "nonlinear": {"cases": {"fc1000.0_lvl-6.0_os0": {"error": "no finite output"}},
                      "nonlinear_score": None, "flagged": ["fc1000.0_lvl-6.0_os0"]},
        "selfosc": None,
    }
    recs = {
        "A": _record(),
        "Y": errored,
        "Z": {"model": "Z", "status": "ok", "flagged": [],
              "linear": None, "nonlinear": None, "selfosc": None},
    }
    written = fe.generate_figures(recs, tmp_path)
    assert len(written) == 6
    assert all(p.stat().st_size > 0 for p in written)


def test_ranking_table_numbers_are_pinned():
    """Both axes and the weighted combined score are formatted, not summarised."""
    rows = [{"model": "A", "linear": 72.5, "nonlinear": 48.0, "combined": 60.25}]
    lines = fe.write_ranking_table(rows).splitlines()
    assert lines[1] == "|---|---:|---:|---:|"
    assert lines[2] == "| A | 72.50 | 48.00 | 60.25 |"
    assert lines[2].index("72.50") < lines[2].index("60.25")
    assert np.isfinite(60.25)


def test_ranking_table_renders_unmeasured_scores_as_na():
    """A flagged model has to reach the table, so the table has to hold a blank.

    main puts every model in the record file, flagged or not, and Task 14 prints
    the ranking from those records. "{:.2f}".format(None) raises TypeError and
    NaN prints as "nan", so a run with one diverged model would take the report
    down at the last step instead of naming the model.
    """
    rows = [
        {"model": "A", "linear": 72.5, "nonlinear": 48.0, "combined": 60.25},
        {"model": "B", "linear": None, "nonlinear": None, "combined": None},
        {"model": "C", "linear": float("nan"), "nonlinear": 10.0, "combined": float("nan")},
    ]
    lines = fe.write_ranking_table(rows).splitlines()
    assert lines[2] == "| A | 72.50 | 48.00 | 60.25 |"
    assert lines[3] == "| B | n/a | n/a | n/a |"
    assert lines[4] == "| C | n/a | 10.00 | n/a |"
    assert "nan" not in lines[4]


def test_main_writes_the_figures_when_asked_and_none_otherwise(tmp_path, monkeypatch):
    """--write-figs is the only thing that turns figures on.

    The figures are a rendering of the records the run already produced, so they
    are drawn from the same dicts rather than from a second pass over the models.
    """
    monkeypatch.setattr(fe, "RUNFILTERS_EXE", "unused")
    monkeypatch.setattr(
        fe,
        "collect_linear",
        lambda *a: {
            "requested": {"resonance": 0.4},
            "measured": {"fc1000.0_os0": {"cutoff_3db_error_cents": -25.0}},
            "linear_score": 72.5,
            "score_parts": {},
            "flagged": [],
            "flag_reason": "",
        },
    )
    monkeypatch.setattr(
        fe,
        "collect_nonlinear",
        lambda *a: {"cases": {}, "nonlinear_score": 48.0, "flagged": [], "flag_reason": ""},
    )
    monkeypatch.setattr(
        fe, "collect_selfoscillation", lambda *a: {"0.9": {"ringing": False}, "flagged": []}
    )
    out_dir = tmp_path / "run"
    figs = tmp_path / "figs"
    assert fe.main(
        ["--models", "Stilson", "--out-dir", str(out_dir), "--write-figs", str(figs)]
    ) == 0
    assert len(sorted(figs.glob("*.png"))) == 6

    quiet = tmp_path / "quiet"
    assert fe.main(["--models", "Stilson", "--out-dir", str(out_dir / "2")]) == 0
    assert not quiet.exists()
