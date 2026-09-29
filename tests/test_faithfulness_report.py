import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import faithfulness_eval as fe  # noqa: E402
import faithfulness_report as fr  # noqa: E402

from tests.test_report import _record  # noqa: E402

RUN_ENVELOPE = {
    "args": ["--models", "All", "--out-dir", "filter_validation/faithfulness/run1"],
    "generated_at": "20260928_201435",
    "git_head": "0123456",
    "software": {"python": "3.14.7", "numpy": "2.5.3", "scipy": "1.18.1"},
}

PLOT_NAMES = [
    "F1_reference_magnitude_vs_k.png",
    "F2_cutoff_error_vs_score.png",
    "F3_harmonic_spectrum.png",
    "F4_thd_vs_level.png",
    "F5_spectral_distance_vs_fc.png",
    "F6_score_bars.png",
]


def _scored(name, lin, nlin, **overrides):
    rec = _record(
        model=name,
        linear={"requested": {"resonance": 0.5}, "measured": {},
                "linear_score": lin, "score_parts": {}},
        nonlinear={"cases": {}, "nonlinear_score": nlin},
        **overrides,
    )
    rec["selfosc"] = {
        "0.5": {"ringing": False, "tail_rms_db": -100.0},
        "flagged": [],
        "flag_reason": "",
    }
    return rec


RINGING_MODEL = "MusicDSP"


def _run_records(n=len(fe.MODEL_NAMES)):
    """One ok record per model, spread across the three score axes.

    MusicDSP rings at r=0.9 because the real run's does, and because the gate
    refuses a run in which nothing rang: a fixture where every tail is quiet
    would make the figure and gate tests below assert on a run the gate rejects.
    """
    recs = {}
    for i, name in enumerate(fe.MODEL_NAMES):
        lin = 10.0 + 3.0 * i
        nlin = 90.0 - 3.0 * i
        recs[name] = _scored(name, lin, nlin)
    recs[RINGING_MODEL]["selfosc"]["0.9"] = {"ringing": True, "tail_rms_db": -6.6}
    return recs


def _quiet(records):
    """Every selfosc reading settled below the -60 dBFS ring threshold."""
    for rec in records.values():
        rec["selfosc"] = {
            "0.5": {"ringing": False, "tail_rms_db": -100.0},
            "0.9": {"ringing": False, "tail_rms_db": -90.0},
            "1.0": {"ringing": False, "tail_rms_db": -80.0},
            "flagged": [],
            "flag_reason": "",
        }
    return records


def _write_run(run_dir, records, envelope=None):
    metrics = Path(run_dir) / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    for name, rec in records.items():
        payload = dict(envelope if envelope is not None else RUN_ENVELOPE)
        payload.update(rec)
        (metrics / f"{name}.json").write_text(json.dumps(payload))
    return Path(run_dir)


def _write_plots(plots_dir, drop=None):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name in PLOT_NAMES:
        if name == drop:
            continue
        (plots_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    return plots_dir


def test_load_records_reads_every_models_metrics_json(tmp_path):
    recs = _run_records()
    run = _write_run(tmp_path / "run", recs)
    loaded = fr.load_records(run)
    assert sorted(loaded) == sorted(fe.MODEL_NAMES)
    assert loaded["Stilson"]["linear"]["linear_score"] == 10.0
    assert loaded["Stilson"]["git_head"] == "0123456"


def test_load_records_keeps_bare_nan_literals(tmp_path):
    """The real records carry 158 bare `NaN` tokens; a strict reader drops the run.

    json.dump writes float('nan') as the bare literal `NaN`, which RFC 8259 does
    not define. Python's reader accepts it, so the record round-trips; what must
    not happen is a reader that rejects the file and reports the run as empty.
    """
    rec = _scored("Improved", 15.0, 80.0)
    rec["linear"]["measured"]["fc5000.0_os0"] = {
        "cutoff_3db_error_cents": float("nan"),
        "magnitude_rms_db_error": float("nan"),
    }
    run = _write_run(tmp_path / "run", _run_records() | {"Improved": rec})
    raw = (run / "metrics" / "Improved.json").read_text()
    assert "NaN" in raw, raw[:200]
    loaded = fr.load_records(run)
    assert loaded["Improved"]["linear"]["measured"]["fc5000.0_os0"][
        "cutoff_3db_error_cents"] != loaded["Improved"]["linear"]["measured"][
            "fc5000.0_os0"]["cutoff_3db_error_cents"]


def test_gate_names_the_model_with_no_record(tmp_path):
    recs = _run_records()
    del recs["Krajeski"]
    run = _write_run(tmp_path / "run", recs)
    with pytest.raises(AssertionError) as exc:
        fr.gate(fr.load_records(run), _write_plots(tmp_path / "plots"))
    assert "Krajeski" in str(exc.value)


def test_gate_rejects_a_run_with_no_ranking(tmp_path):
    recs = {name: _record(model=name, linear=None, nonlinear=None)
            for name in fe.MODEL_NAMES}
    run = _write_run(tmp_path / "run", recs)
    with pytest.raises(AssertionError) as exc:
        fr.gate(fr.load_records(run), _write_plots(tmp_path / "plots"))
    assert "ranking" in str(exc.value).lower()


def test_gate_names_a_model_that_is_ranked_but_not_ok(tmp_path, monkeypatch):
    """A flagged model must not reach the ranking, and the gate says so by name.

    A model that is not ok has no ranking entry, however good its survivors look,
    so ranking one would put a number in the table that does not stand for the
    model. `_score_bars` already enforces that; the gate re-checks it, so a
    change to either one that broke the other fails here instead of shipping a
    flagged model in the leaderboard. The stub is what makes that reachable.
    """
    recs = _run_records()
    recs["Stilson"] = _scored("Stilson", 100.0, 100.0)
    recs["Stilson"]["status"] = "flagged"
    run = _write_run(tmp_path / "run", recs)
    loaded = fr.load_records(run)
    assert "Stilson" not in [b[0] for b in fe._score_bars(loaded)]
    plots = _write_plots(tmp_path / "plots")

    monkeypatch.setattr(
        fe, "_score_bars",
        lambda r: [("Stilson", 100.0, 100.0, 100.0), ("MusicDSP", 10.0, 10.0, 10.0)],
    )
    with pytest.raises(AssertionError) as exc:
        fr.gate(loaded, plots)
    assert "Stilson" in str(exc.value)
    assert "flagged" in str(exc.value)


def test_gate_passes_a_run_that_meets_every_invariant(tmp_path):
    recs = _run_records()
    run = _write_run(tmp_path / "run", recs)
    assert fr.gate(fr.load_records(run), _write_plots(tmp_path / "plots")) is None


def test_gate_fires_when_no_model_rings(tmp_path):
    """The self-oscillation risk is only reportable if something actually rang.

    A run where every tail sits below -60 dBFS is a measurement, and the report
    has to be able to say so. A gate that only checked the field existed would
    let the report state a self-oscillation risk it never observed.
    """
    run = _write_run(tmp_path / "run", _quiet(_run_records()))
    with pytest.raises(AssertionError) as exc:
        fr.gate(fr.load_records(run), _write_plots(tmp_path / "plots"))
    assert "ring" in str(exc.value).lower()


def test_gate_ignores_the_selfosc_meta_keys(tmp_path):
    """`flagged` and `flag_reason` are list/str, not {ringing, tail_rms_db}.

    Scanning selfosc for ringing walks whatever is in the dict, and the two meta
    keys are in there. Reading a `ringing` off either is a TypeError or a truthy
    stand-in for a tail that was never measured, so a run whose real tails all
    sit quiet would be reported as ringing.
    """
    recs = _quiet(_run_records())
    recs[RINGING_MODEL]["selfosc"] = {
        "0.5": {"ringing": False, "tail_rms_db": -106.0},
        "flagged": ["r1.00"],
        "flag_reason": "no finite output at 1 of 3 resonances: r1.00",
    }
    run = _write_run(tmp_path / "run", recs)
    plots = _write_plots(tmp_path / "plots")
    with pytest.raises(AssertionError):
        fr.gate(fr.load_records(run), plots)
    # And the same record with one real ringing key passes.
    recs[RINGING_MODEL]["selfosc"]["0.9"] = {"ringing": True, "tail_rms_db": -6.6}
    _write_run(run, recs)
    assert fr.gate(fr.load_records(run), plots) is None


def test_gate_names_a_missing_committed_figure(tmp_path):
    recs = _run_records()
    run = _write_run(tmp_path / "run", recs)
    plots = _write_plots(tmp_path / "plots", drop="F3_harmonic_spectrum.png")
    with pytest.raises(AssertionError) as exc:
        fr.gate(fr.load_records(run), plots)
    assert "F3_harmonic_spectrum.png" in str(exc.value)


def test_gate_requires_all_six_figures(tmp_path):
    assert len(PLOT_NAMES) == 6
    assert [f.filename for f in fr.FIGURE_SPECS] == PLOT_NAMES
    recs = _run_records()
    run = _write_run(tmp_path / "run", recs)
    plots = _write_plots(tmp_path / "plots")
    assert fr.gate(fr.load_records(run), plots) is None


def _oracle_case(name, profile, thd):
    cases = {}
    for i, level in enumerate(fe.LEVELS_DBFS):
        cases[f"fc1000.0_lvl{level}_os0"] = {
            "spectral_distance_db": 3.0 + i,
            "thd_model_percent": 1.0 + i,
            "thd_oracle_percent": thd,
            "harmonic_profile_db_model": list(profile),
            "harmonic_profile_db_oracle": list(profile),
        }
    return cases


def _full_run(tmp_path, recs=None, envelope=None):
    """A run dir with all twelve records, oracle cases on Stilson, and six PNGs."""
    recs = recs if recs is not None else _run_records()
    recs["Stilson"] = _scored("Stilson", 0.3, 32.84)
    recs["Stilson"]["nonlinear"]["cases"] = _oracle_case(
        "Stilson", [0.0] + [-67.94] * 9, 12.98)
    recs["Stilson"]["linear"]["measured"] = {
        "fc100.0_os0": {
            "magnitude_rms_db_error": 534.358877486308,
            "cutoff_3db_error_cents": float("nan"),
            "passband_gain_error_db": -591.2382330372437,
        },
    }
    # As in the real record: the fc=100.0 cutoff error has no finite value to
    # average over, and the dead case dominates the RMS error.
    recs["Stilson"]["linear"]["score_parts"] = {
        "magnitude_rms_db_error": 116.63468310869115,
        "cutoff_3db_error_cents": 577.5001811142538,
        "passband_gain_error_db": -90.21581140270855,
        "stopband_slope_error_db_per_oct": 20.369742615765208,
        "peak_gain_error_db": 5.848031083207826,
        "peak_freq_error_cents": -1851.6104989340386,
        "n_cases": {
            "magnitude_rms_db_error": 6, "cutoff_3db_error_cents": 5,
            "passband_gain_error_db": 6, "stopband_slope_error_db_per_oct": 4,
            "peak_gain_error_db": 6, "peak_freq_error_cents": 6,
        },
    }
    recs["Improved"]["linear"]["score_parts"] = {
        "magnitude_rms_db_error": 11.838,
        "cutoff_3db_error_cents": float("nan"),
        "passband_gain_error_db": -5.35,
        "stopband_slope_error_db_per_oct": 2.129,
        "peak_gain_error_db": -4.538,
        "peak_freq_error_cents": -842.532,
        "n_cases": {
            "magnitude_rms_db_error": 6, "cutoff_3db_error_cents": 5,
            "passband_gain_error_db": 6, "stopband_slope_error_db_per_oct": 4,
            "peak_gain_error_db": 6, "peak_freq_error_cents": 6,
        },
    }
    run = _write_run(tmp_path / "run", recs, envelope)
    plots = _write_plots(tmp_path / "plots")
    return run, plots, recs


def _render(tmp_path, recs=None, envelope=None):
    run, plots, recs = _full_run(tmp_path, recs, envelope)
    text = fr.render(fr.load_records(run), plots_dir=plots)
    return text, recs, run, plots


def _lines_with(text, needle):
    return [line for line in text.splitlines() if needle in line]


def _ranking_rows(text):
    """The rows of every combined ranking table, as (model, linear, nonlinear, combined)."""
    rows, in_table = [], False
    for line in text.splitlines():
        if line.startswith("| Model | Linear | Nonlinear | Combined |"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("|---"):
            continue
        if not line.startswith("| "):
            break
        rows.append([c.strip() for c in line.split("|")[1:-1]])
    return rows


def _card_text(text, name):
    """The body of one model's card from section 5."""
    m = re.search(r"^### " + re.escape(name) + r"\n(.*?)(?=\n### |\n## )",
                  text, re.S | re.M)
    assert m is not None, f"no card for {name}"
    return m.group(1)


def _stands_alone(text, word):
    """True if `word` appears in `text` as a word, not inside another one.

    A substring search for "nan" matches "resonance", and a check that passes
    on that basis is not checking what it claims to check.
    """
    return re.search(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])",
                     text, re.IGNORECASE) is not None


def test_report_headline_names_the_measured_leader_and_its_margin(tmp_path):
    """The verdict is the measured order, not the one the design predicted.

    The design doc expected Stilson to lead; the records put it last, on a
    linear score of 0.30. A verdict that hedged, or that named an expected
    leader, would be a claim the table under it contradicts.
    """
    text, recs, _run, _plots = _render(tmp_path)
    assert "## 1. Verdict" in text
    verdict = text.split("## 1. Verdict")[1].split("\n## ")[0]
    bars = fe._score_bars(recs)
    top, second = bars[0], bars[1]
    assert top[0] in verdict and f"{top[3]:.2f}" in verdict, verdict
    assert f"{top[3] - second[3]:.2f}" in verdict
    assert second[0] in verdict
    # The order it states is the order the table renders, best combined first.
    assert [row[0] for row in _ranking_rows(verdict)] == [b[0] for b in bars]
    assert bars[-1][0] == "Stilson"
    assert bars[-1][0] in verdict


def test_ranking_table_carries_both_axes_and_n_a_not_nan(tmp_path):
    text, recs, _run, _plots = _render(tmp_path)
    bars = fe._score_bars(recs)
    expected = fe.write_ranking_table([
        {"model": b[0], "linear": b[1], "nonlinear": b[2], "combined": b[3]}
        for b in bars
    ])
    assert expected in text
    rows = _ranking_rows(text)
    assert len(rows) == len(bars) == 12
    for row in rows:
        assert len(row) == 4
        model, lin, nlin, comb = row
        assert lin != "n/a" and nlin != "n/a" and comb != "n/a", row
        # Both axes are printed before the total, so a reader can see which one
        # a model failed on instead of reading a single averaged number.
        assert [b[0] for b in bars if b[0] == model][0] is not None
    assert not _stands_alone(expected, "nan")
    # The combined column is the weighted mean of the two, not a third opinion.
    for row, bar in zip(rows, bars):
        assert float(row[3]) == pytest.approx(
            fe.COMBINED_WEIGHT_LINEAR * bar[1]
            + (1 - fe.COMBINED_WEIGHT_LINEAR) * bar[2])


def test_unmeasured_axis_renders_as_n_a(tmp_path):
    """A record with one axis missing must reach the table as a blank, not crash.

    A model whose collector raised has `None` on that axis. "{:.2f}".format(None)
    raises TypeError, and a NaN prints as "nan" in the column a reader scans for
    the leader, which is the one column they read.
    """
    recs = _run_records()
    recs["Microtracker"] = _scored("Microtracker", 20.77, 70.38, status="flagged")
    text, _recs, _run, _plots = _render(tmp_path, recs)
    assert "## 4." in text
    excluded = text.split("### 4.4")[1].split("### 4.5")[0]
    assert "| Microtracker | flagged |" in excluded
    assert not any(row[0] == "Microtracker" for row in _ranking_rows(text))
    assert not _stands_alone(text, "nan")
    # The other eleven are still ranked, so the table is not collateral damage.
    assert len(_ranking_rows(text)) == 11


def test_bare_nan_in_a_record_does_not_leak_into_the_report(tmp_path):
    """The real records carry 158 bare `NaN` tokens; none may reach a reader.

    Stilson at fc=100.0 has an unmeasurable cutoff error because its output sits
    on the -600 dB floor, and Improved, Microtracker, HyperionLegacy and
    RKSimulation never cross -3 dB at all. Those are absent measurements: the
    report has to say n/a rather than print nan or raise on the format spec.
    """
    text, _recs, _run, _plots = _render(tmp_path)
    assert not _stands_alone(text, "nan")
    assert "n/a" in text
    for cell in [c for line in text.splitlines() for c in line.split("|")]:
        assert not _stands_alone(cell, "nan"), cell
    # The fields that could not be measured are named, so a blank is explained.
    assert "Unmeasurable on this model" in text
    assert "cutoff error (no finite value" in text


def test_flagged_model_is_excluded_from_the_ranking_with_its_reason(tmp_path):
    recs = _run_records()
    recs["MusicDSP"]["status"] = "flagged"
    recs["MusicDSP"]["linear"]["flagged"] = ["fc5000.0_os4"]
    recs["MusicDSP"]["linear"]["flag_reason"] = (
        "no finite output at 1 of 6 cases: fc5000.0_os4")
    recs["MusicDSP"]["flagged"] = ["fc5000.0_os4"]
    text, _recs, _run, _plots = _render(tmp_path, recs)

    assert not any(row[0] == "MusicDSP" for row in _ranking_rows(text))
    excluded = text.split("### 4.4")[1].split("### 4.5")[0]
    assert "MusicDSP" in excluded
    assert "no finite output at 1 of 6 cases: fc5000.0_os4" in excluded
    # And it is still a model that was measured, so it still gets a card.
    assert "### MusicDSP" in text
    assert "**Not ranked.**" in text


def test_threats_to_validity_leads_with_the_model_not_hardware_caveat(tmp_path):
    text, _recs, _run, _plots = _render(tmp_path)
    threats = text.split("## 6. Threats to validity")[1].split("\n## ")[0]
    body = [line.strip() for line in threats.splitlines() if line.strip()]
    prose = "\n".join(body)
    assert "model" in prose.lower()
    first_bullet = next((line for line in body if line.startswith("-")), "")
    assert "oracle" in first_bullet.lower() or "hardware" in first_bullet.lower(), first_bullet
    # The four other named threats are present, not implied.
    for needle in ("errata", "coincidental", "mislabeled", "-3 dB"):
        assert needle.lower() in prose.lower(), needle


def test_figures_are_referenced_by_relative_path_with_matching_titles(
        tmp_path, monkeypatch):
    """The caption IS the figure's title, so the two cannot drift apart.

    F1 and F2 have no input level to state; F3 and F4 draw one model's oracle
    and the committed PNGs say which. A caption that named a level F1 was not
    measured at, or that dropped the oracle's model, would be a claim the image
    under it does not support.
    """
    from tests.test_report import _captured_axes

    text, recs, _run, plots = _render(tmp_path)
    titles = _captured_axes(monkeypatch, tmp_path / "figs", recs)
    for spec in fr.FIGURE_SPECS:
        assert titles[spec.filename]["title"] in text, spec.filename
        assert f"]({spec.relpath})" in text, spec.filename
    assert "level=" not in _caption_of(text, "F1")
    assert "level=" not in _caption_of(text, "F2")
    assert "oracle (Stilson-aligned)" in _caption_of(text, "F3")
    assert "oracle (Stilson-aligned)" in _caption_of(text, "F4")
    # The run directory is gitignored, so no path into it may be embedded.
    assert "filter_validation" not in text.split("## 8.")[0].split("## 7.")[-1]


def _caption_of(text, tag):
    """The caption line under a figure: the line after that figure's image.

    Pinned to the image rather than to the first line starting with the tag, so
    prose that happens to open with "F3 and F4 ..." is not mistaken for a
    caption.
    """
    spec = next(s for s in fr.FIGURE_SPECS if s.tag == tag)
    lines = text.splitlines()
    at = next(i for i, line in enumerate(lines) if line.endswith(f"]({spec.relpath})"))
    return lines[at + 2]


def test_reproduction_names_the_command_and_the_record_envelope(tmp_path):
    text, _recs, _run, _plots = _render(
        tmp_path, envelope={"generated_at": "20260929_155014", "git_head": "b474371",
                            "software": {"python": "3.14.7", "numpy": "2.5.3",
                                         "scipy": "1.18.1"}})
    repro = text.split("## 8. Reproduction")[1].split("\n## ")[0]
    assert "python scripts/faithfulness_eval.py --models All" in repro
    assert "--out-dir filter_validation/faithfulness/run1" in repro
    for value in ("b474371", "3.14.7", "2.5.3", "1.18.1"):
        assert value in repro, value
    assert "filter_verification.py" in repro
    assert "not modified" in repro or "untouched" in repro


def test_appendix_states_what_was_not_measured(tmp_path):
    text, _recs, _run, _plots = _render(tmp_path)
    appendix = text.split("## 9. Appendix")[1]
    for needle in ("perceptual", "out of band", "aliasing", "intermodulation"):
        assert needle in appendix.lower(), needle


def test_legacy_artifacts_match_the_legacy_records(tmp_path):
    """The legacy-carried artifact sentences rest on numbers the records contain.

    Those values are the ones a reader cannot check in this report (the legacy
    run directory is gitignored, like this one), so wording drift there is
    exactly what a regression test should catch: the card claims "dc_gain =
    0.0000" and "settled … in 27 ms", and both have to be true of the step
    records the sentence says they come from, or the claim is legible and
    wrong.
    """
    records = fr.load_records(
        REPO / "filter_validation" / "faithfulness" / "run1")
    text = fr.render(records)
    step = json.load(
        open(REPO / "filter_validation" / "2026-09-28_201435" / "metrics"
             / "step.json"))
    by_name = {}
    for row in step:
        by_name.setdefault(row["filter_name"], []).append(
            (row["resonance"], row["metrics"]))
    stilson = dict(by_name["Stilson"])
    assert all(d["dc_gain"] == 0.0 for d in stilson.values())
    assert "dc_gain = 0.0000" in text
    improved = dict(by_name["Improved"])
    dc_at_zero = improved[0.0]["dc_gain"]
    assert -1.001 < dc_at_zero < -0.999, dc_at_zero
    legacy = dict(by_name["HyperionLegacy"])
    settle_5 = legacy[0.5]["settling_time_ms"]
    settle_9 = legacy[0.9]["settling_time_ms"]
    settle_0 = legacy[0.0]["settling_time_ms"]
    assert 741.0 < settle_5 < 744.0, settle_5
    assert 741.0 < settle_9 < 744.0, settle_9
    assert settle_0 < 50.0, settle_0  # r=0.00 settled promptly
    card = _card_text(text, "HyperionLegacy")
    assert "r=0.50" in card and "r=0.00" in card
    assert "27" in card  # the prompt settle, from the records
    music = dict(by_name["MusicDSP"])
    never = music[0.9]["settling_time_ms"]
    window = 32768 / fe.SAMPLE_RATE * 1000
    assert abs(never - window) < 1.0, (never, window)  # record-length = never

    # Truthfulness of the appendix's partial-case annotation.
    parts = (records["OberheimVariation"].get("linear") or {}).get("score_parts")
    n = (parts or {}).get("n_cases") or {}
    assert n.get("cutoff_3db_error_cents", 6) < 6, "fixture regression"
    assert "2 of 6 cases" in text


def test_same_records_render_the_same_bytes(tmp_path):
    first, _recs, run, plots = _render(tmp_path / "a")
    recs2 = fr.load_records(run)
    second = fr.render(recs2, plots_dir=plots)
    assert first == second
    # And the record order the reader happens to build does not reach the output.
    backwards = {name: recs2[name] for name in reversed(fe._ordered_names(recs2))}
    assert fr.render(backwards, plots_dir=plots) == first


def _table_rows(text, header):
    """The rows of the first table whose header line is exactly `header`."""
    rows, in_table = [], False
    for line in text.splitlines():
        if line == header:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("|---"):
            continue
        if not line.startswith("| "):
            break
        rows.append([c.strip() for c in line.split("|")[1:-1]])
    return rows


def test_excluded_table_holds_only_actually_excluded_models(tmp_path):
    """A ranked, ok model must not be listed as excluded.

    The excluded table is built by walking the record set and keeping whatever
    is not in the ranking. If that membership test is done against the wrong
    object, every model lands in the table and the table says nothing. This
    asserts on a model that is definitely ranked: a report that lists it here
    is contradicting its own ranking table in the same document.
    """
    text, recs, _, _ = _render(tmp_path)
    assert "### 4.4 Models excluded from the ranking" in text
    excluded = _table_rows(text, "| Model | Status | Reason |")
    ranked = {row[0] for row in _ranking_rows(text)}
    listed = {row[0] for row in excluded}
    assert listed <= ranked, (
        f"listed as excluded but also ranked: {sorted(listed - ranked)}"
    )


def test_excluded_table_is_empty_when_no_model_is_flagged(tmp_path):
    """With 12 ok records the excluded table has no rows and says why.

    The section has to be present either way, so a reader can tell the
    difference between "nobody was dropped" and "the drop list was not built".
    """
    text, _, _, _ = _render(tmp_path)
    assert _table_rows(text, "| Model | Status | Reason |") == []
    assert "No model was flagged" in text  # the empty case, stated
    # The verdict must not promise a list of excluded models that is not there.
    verdict = text.split("## 2.")[0]
    assert "names those separately" not in verdict


def test_fc_is_never_documented_as_the_minus3db_point(tmp_path):
    """The stated fc-to--3 dB ratio must be the reference's own measured one.

    The reference's cascade is 3 dB down well below its leading-pole fc. If the
    report states a ratio of 1, or a number not derivable from the reference, a
    reader would reconstruct the wrong filter. The ratio here is recomputed
    from `moog_ladder_linear` rather than compared to a literal, so a change to
    the port that moves the corner fails the test instead of passing it.
    """
    text, _, _, _ = _render(tmp_path)
    freqs = np.logspace(1, np.log10(0.45 * fe.SAMPLE_RATE), 200000)
    d = fe.reference_magnitude_db(1000.0, 0.0, freqs)
    ratio = freqs[int(np.argmin(np.abs(d + 3.0)))] / 1000.0
    assert f"{ratio:.3f}" in text, f"expected the measured ratio {ratio:.3f} in the report"
    assert "is a -3 dB point near" in text


def test_a_model_at_the_floor_does_not_report_earning_credit(tmp_path):
    """A card must not say a model earned credit when every metric rounds to 0.00.

    Stilson's six linear losses are five exact zeros and one at 0.003, which
    prints as 0.00. "The most credit it earned was ... worth 0.00 of the 0.00 it
    earned" is a sentence about nothing; the honest card says the axis scored
    the floor and names the metrics that were at or past their worst value.
    """
    text, _, _, _ = _render(tmp_path)
    card = _card_text(text, "Stilson")
    assert "it earned on the axis" not in card
    assert "no credit on any of the six metrics" in card
    assert "worst value the scale admits" in card
    # "6 of the six ... and the rest" counts and then has a remainder.
    assert "of the six (" not in card
    assert "the rest" not in card
    # And a model that does earn credit still gets the comparative sentence.
    assert "it earned on the axis" in _card_text(text, "Improved")


def test_reproduction_points_at_where_the_envelope_actually_lives(tmp_path):
    """The reproduction section must name a real key in a real record.

    The environment lines are read off the first record in enum order, and the
    sentence that says so has to name the key they came from. Citing an
    `envelope` key that the records do not have makes the one part of the
    report a reader can check impossible to check.
    """
    run, plots, recs = _full_run(tmp_path)
    written = fr.load_records(run)
    text = fr.render(written, plots_dir=plots)
    first = next(iter(fe._ordered_names(written)))
    for key in ("git_head", "generated_at", "software", "args"):
        assert key in written[first], f"a written record should carry {key}"
        assert "envelope" not in written[first]
    assert "`envelope`" not in text
    assert f"`{first}.json`" in text
    assert "`git_head`" in text


def test_every_section_cross_reference_points_at_a_heading(tmp_path):
    """No "section N.M" may name a section the document does not have.

    Cross-references are prose written once and left there while the sections
    move, and a reader who follows one lands nowhere. This walks every
    reference in the report and checks the heading exists, in the report's own
    order, so renumbering without rewriting the prose fails here.
    """
    text, _, _, _ = _render(tmp_path)
    headings = set(re.findall(r"^#{2,3} (\d+(?:\.\d+)?)\.? ", text, re.M))
    assert headings, "the report should have numbered sections"
    referenced = set(re.findall(r"section (\d+(?:\.\d+)?)", text))
    missing = referenced - headings
    assert not missing, f"cross-references to sections that do not exist: {sorted(missing)}"
    for ref in sorted(referenced, key=lambda s: [int(x) for x in s.split(".")]):
        assert ref in headings


def test_excluded_prose_agrees_with_the_excluded_count(tmp_path):
    """One exclusion is not "these models", and the reason is not doubled up.

    The excluded table is usually empty and the prose usually plural, so a
    report with one flagged model is the case that has never been read.
    """
    recs = _run_records()
    recs["MusicDSP"]["status"] = "flagged"
    recs["MusicDSP"]["linear"]["flagged"] = ["fc5000.0_os4"]
    recs["MusicDSP"]["linear"]["flag_reason"] = "no finite output at 1 of 6 cases"
    recs["MusicDSP"]["flagged"] = ["fc5000.0_os4"]
    text, _recs, _run, _plots = _render(tmp_path, recs)
    section = text.split("### 4.4")[1].split("### 4.5")[0]
    assert "These models produced" not in section
    assert "This model produced" in section
    card = _card_text(text, "MusicDSP")
    # "reason: linear: x" doubles the colon; "reason linear: x" does not.
    assert "reason: linear: " not in card
    assert "reason linear: no finite output at 1 of 6 cases" in card
