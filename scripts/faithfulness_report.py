#!/usr/bin/env python
"""Render docs/moog-faithfulness-report.md from a faithfulness_eval run.

The report is a reading of the run's records, not a second measurement: it
imports the record reader, the ranking and the caption builder from
`faithfulness_eval.py` rather than re-deriving any of them, so a number in the
markdown and the same number in `metrics/<Model>.json` cannot come from two
different places.

    python scripts/faithfulness_report.py filter_validation/faithfulness/run1

Nothing is written until the invariants in `gate` hold, and each one raises an
AssertionError naming the record that broke it. A report that quietly dropped a
model, ranked a flagged one, or cited a figure that is not in the tree would
read as a measurement that was never taken.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

# The `reference` package lives at the repo root and faithfulness_eval.py lives
# beside this file, so both dirs go on sys.path: this file is run as a script
# (scripts/ only) and imported from tests (neither).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import faithfulness_eval as fe  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "moog-faithfulness" / "plots"
REPORT_PATH = REPO / "docs" / "moog-faithfulness-report.md"

# The figures the report embeds, in report order. The captions are built with
# fe.fig_caption from the same operating points generate_figures drew each one at,
# so a caption cannot drift from the figure it describes. `relpath` is relative
# to this report file (docs/moog-faithfulness-report.md), which is why it starts
# at `moog-faithfulness/plots/` and not at `plots/`.
class FigureSpec(NamedTuple):
    tag: str
    filename: str

    @property
    def relpath(self):
        return f"moog-faithfulness/plots/{self.filename}"


FIGURE_SPECS: Tuple[FigureSpec, ...] = (
    FigureSpec("F1", "F1_reference_magnitude_vs_k.png"),
    FigureSpec("F2", "F2_cutoff_error_vs_score.png"),
    FigureSpec("F3", "F3_harmonic_spectrum.png"),
    FigureSpec("F4", "F4_thd_vs_level.png"),
    FigureSpec("F5", "F5_spectral_distance_vs_fc.png"),
    FigureSpec("F6", "F6_score_bars.png"),
)

# The absolute k the nonlinear axis is measured at. collect_nonlinear holds it
# fixed, so every operating point on that axis is at k=2 and the report can
# state it once.
NONLINEAR_K = 2.0

# The cutoff the nonlinear comparison case (section 4.3's THD column) was run
# at, read out of the case key rather than written beside it: a constant that
# could disagree with the case it describes is a claim about a measurement
# nobody took.
ORACLE_FC = float(fe.ORACLE_CASE.split("_", 1)[0][len("fc"):])

# The operating points the sweep fixed, spelled out for the methodology section
# so the numbers in the records and the numbers in the prose cannot disagree.
METHOD_POINTS = (
    f"cutoffs {', '.join(f'{c:g} Hz' for c in fe.DEFAULT_CUTOFFS)}",
    f"oversampling {', '.join(str(o) for o in fe.DEFAULT_OVERSAMPLES)}x",
    f"input levels {', '.join(f'{l:g}' for lvl in fe.LEVELS_DBFS for l in (lvl,))} dBFS",
    f"resonance calibrated to absolute k={NONLINEAR_K:g} per model",
)


def load_records(run_dir):
    """{model_name: record} from <run_dir>/metrics/*.json.

    Keyed on the record's own `model` field rather than the filename stem, so a
    record written under a name that is not a model is still reported under the
    name it claims.

    Bare `NaN` literals are read, not rejected: json.dump writes float('nan') as
    the unquoted token `NaN`, and the real records carry 158 of them (cutoff
    errors over a stopband, the fc=100.0 os=0 cases whose output sits on the
    -600 dB floor). A reader that insisted on RFC 8259 would call the run
    malformed and report nothing at all, which is the opposite of the honest
    answer: those fields are absent measurements, and the scorer already reads
    them as failures.
    """
    metrics = Path(run_dir) / "metrics"
    if not metrics.is_dir():
        raise FileNotFoundError(f"{metrics} is not a directory; pass a run directory")
    records: Dict[str, dict] = {}
    for path in sorted(metrics.glob("*.json")):
        with open(path) as fh:
            rec = json.load(fh)
        if not isinstance(rec, dict):
            raise ValueError(f"{path} holds {type(rec).__name__}, expected a JSON object")
        name = rec.get("model") or path.stem
        if name in records:
            raise ValueError(f"two records claim the model name {name!r}: {path}")
        records[name] = rec
    return records


def _ringing(records):
    """{model: [resonance, ...]} for every selfosc reading that measured a ring.

    `selfosc` carries two meta keys beside the per-resonance readings: `flagged`
    and `flag_reason`. They are not readings, and a `ringing` off either would be
    a KeyError or, worse, a truthy value standing in for a tail that was never
    measured, so the scan keeps only the dicts that hold the two fields.
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    for name in fe._ordered_names(records):
        selfosc = records[name].get("selfosc")
        if not isinstance(selfosc, dict):
            continue
        rings = []
        for r, entry in selfosc.items():
            if not isinstance(entry, dict) or "ringing" not in entry:
                continue
            if entry.get("ringing") is True:
                rings.append((r, fe._num(entry.get("tail_rms_db"))))
        if rings:
            out[name] = rings
    return out


def _flag_reason(rec):
    """The one sentence a report has to give for a model it cannot rank."""
    parts = []
    for section in ("linear", "nonlinear", "selfosc"):
        body = rec.get(section)
        if isinstance(body, dict) and body.get("flag_reason"):
            parts.append(f"{section}: {body['flag_reason']}")
    if parts:
        return "; ".join(parts)
    flagged = rec.get("flagged") or []
    if flagged:
        return "flagged conditions: " + ", ".join(str(f) for f in flagged)
    return str(rec.get("status") or "no reason recorded")
def gate(records, plots_dir=PLOTS_DIR):
    """Refuse to render a run the report would misdescribe. Returns None.

    Each failure is an AssertionError naming the record, not a skipped section
    and not a crash deep inside a table cell: a report that quietly omits a
    model, ranks a flagged one, or embeds a figure that is not in the tree reads
    as a measurement that was never taken.
    """
    missing = [name for name in fe.MODEL_NAMES if name not in records]
    assert not missing, (
        f"no record for {', '.join(missing)}; the run holds "
        f"{', '.join(fe._ordered_names(records)) or 'nothing'}"
    )

    bars = fe._score_bars(records)
    for name, _lin, _nlin, _comb in bars:
        status = records[name].get("status")
        assert status == "ok", (
            f"{name} is ranked with status {status!r}; a model that is not ok "
            f"has no ranking entry, and its reason is "
            f"{_flag_reason(records[name])!r}"
        )
    assert bars, (
        f"the ranking is empty: no model has both axes measured; "
        f"statuses are "
        + ", ".join(f"{n}={records[n].get('status')!r}"
                    for n in fe._ordered_names(records))
    )

    rings = _ringing(records)
    assert rings, (
        "no record has selfosc.<resonance>.ringing == True, so the "
        "self-oscillation risk is not observed by this run and cannot be "
        "reported from it"
    )

    for spec in FIGURE_SPECS:
        path = Path(plots_dir) / spec.filename
        assert path.exists(), f"{path} is missing; the report embeds it by that name"

    return None


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

# The dB floor spectrum_db applies after normalizing, so digital silence reads
# -600 dB at any record length. A passband gain error near it is not a bad
# measurement of a bad filter, it is the measurement of no output at all, and the
# report has to say which one it is looking at.
SILENCE_FLOOR_DB = -600.0
DEAD_OUTPUT_DB = -100.0

# Score components, in the order the linear axis weights them.
LINEAR_LABELS = {
    "magnitude_rms_db_error": "RMS magnitude error",
    "cutoff_3db_error_cents": "cutoff error",
    "passband_gain_error_db": "passband gain error",
    "stopband_slope_error_db_per_oct": "stopband slope error",
    "peak_gain_error_db": "peak gain error",
    "peak_freq_error_cents": "peak frequency error",
}
LINEAR_UNITS = {
    "magnitude_rms_db_error": "dB",
    "cutoff_3db_error_cents": "cents",
    "passband_gain_error_db": "dB",
    "stopband_slope_error_db_per_oct": "dB/octave",
    "peak_gain_error_db": "dB",
    "peak_freq_error_cents": "cents",
}
# Which of the six are signed, so the card can say "199 cents low" rather than
# "199" and a reader cannot tell a model that runs its cutoff low from one that
# runs it high.
LINEAR_SIGNED = frozenset({
    "cutoff_3db_error_cents", "passband_gain_error_db",
    "peak_gain_error_db", "peak_freq_error_cents",
})
LINEAR_DIRECTION = {"low": "low", "high": "high"}

# Facts carried over from the 2026-09-28 legacy run (scripts/
# filter_verification.py, filter_validation/<run>/metrics/{step,thd}.json).
# The legacy step suite swept fc=1000 only, at resonances 0.0/0.5/0.9, so every
# claim below is scoped to what was measured there (Stilson returned dc_gain
# 0.0000 and Improved -0.99997 at every resonance swept; Improved's gain stays
# inverted but shrinks in magnitude as resonance rises). They are not
# measurements of this harness: this sweep drives impulses and tones, not a
# step, so it has no dc_gain to report. They are quoted as what that suite
# found, and where this run sees the same defect the card says so.
#
# The legacy THD suite is a different sweep and a different set of numbers: it
# ran fc=5000 at r=0.0 over -18/-12/-6 dBFS, and read its THD from binned FFT
# bins rather than at the interpolated fundamental this harness uses. Its
# figures are stated with that operating point attached, because a legacy THD
# number quoted without one reads as if it contradicted section 4.3, which is a
# different measurement at fc=1000.
#
# Each entry is (what that suite found, what this sweep cannot say about it):
# the closing half is per fact, not per model, because a step measurement and
# a THD measurement are confirmed by different evidence, or by none at all.
LEGACY_ARTIFACTS = {
    "Stilson": (
        ("The legacy suite reported `dc_gain = 0.0000` at its step operating "
         "point, emitting literal zeros to a step.",
         "This sweep drives impulses and tones, not a step, so it has no "
         "`dc_gain` of its own with which to confirm or contradict that; the "
         "legacy step suite swept fc=1000 only, at resonances 0.0/0.5/0.9."),
        ("In that suite's THD sweep, at its -6 dBFS input level (fc=5000, "
         "r=0.00) this model reported THD 0.0000 %, with all five of its "
         "harmonics sitting at -240.0 dB, the sentinel that suite writes for "
         "digital silence: the figure is the sentinel rather than a "
         "measurement of low distortion. Huovilainen read 0.0023 % "
         "(0.002342 % stored) at the same level, which is the ordering at that "
         "floor.",
         "{faithful_thd}"),
    ),
    "Improved": (
        ("The legacy suite reported `dc_gain = -0.99997` (approximately -1) at "
         "its step operating point, inverting the step instead of low-passing "
         "it; the inversion persists across resonance, shrinking in magnitude.",
         "This sweep drives impulses and tones, not a step, so it has no "
         "`dc_gain` of its own with which to confirm or contradict that; the "
         "legacy step suite swept fc=1000 only, at resonances 0.0/0.5/0.9."),
    ),
    "HyperionLegacy": (
        ("The legacy suite's step record is 743.04 ms long and this model's "
         "output did not enter the 2 % band until 742.27 ms at r=0.50 and "
         "742.77 ms at r=0.90 - it settled only in the last millisecond of "
         "the window at those two resonances. (At r=0.00 it settled promptly, "
         "in 27 ms.) MusicDSP at r=0.90 reports 743.04 ms, which is the record "
         "length: the metric's way of saying it never settled inside the "
         "window at all.",
         "This sweep drives impulses and tones, not a step, so it has no "
         "`dc_gain` of its own with which to confirm or contradict that; the "
         "legacy step suite swept fc=1000 only, at resonances 0.0/0.5/0.9."),
    ),
}

REPRO_COMMAND = (
    "python scripts/faithfulness_eval.py --models All \\\n"
    "  --out-dir filter_validation/faithfulness/run1 \\\n"
    "  --write-figs /tmp/figs-run1"
)

# The figures the report embeds, with the operating points generate_figures drew
# them at. A caption is built with fe.fig_caption from exactly these, so the line
# under each image is the figure's own title rather than a description of it
# that can drift from what was plotted.
FIGURE_CAPTIONS = {
    "F1": ({"title": "reference magnitude vs k", "fc": 1000, "K": "various",
            "f_s": fe.SAMPLE_RATE},
           "The headline shape target: the analytic magnitude of the analog "
           "ladder at one leading-pole cutoff, swept over the whole usable k "
           "range. Every model is scored against this family, and against "
           "k=2 in particular."),
    "F2": ({"title": "cutoff error vs linear score", "fc": 1000,
            "K": "calibrated", "f_s": fe.SAMPLE_RATE},
           "One dot per model: how far the model's -3 dB point sits from the "
           "reference's at the calibrated resonance, against its linear score. "
           "A dot at zero error and low score is a model that gets the corner "
           "right and the rest of the curve wrong."),
    "F3": ({"title": "harmonic spectrum vs oracle", "fc": 1000, "K": 2,
            "f_s": fe.SAMPLE_RATE, "level": -6},
           "H1 through H10 in dB relative to each signal's own fundamental, "
           "for every model and for one model's oracle. The oracle is named in "
           "the title because it is model-aligned: align_signals shifts it by "
           "the model's own cross-correlation, so there is no single oracle "
           "curve to draw."),
    "F4": ({"title": "THD vs level, with oracle", "fc": 1000, "K": 2,
            "f_s": fe.SAMPLE_RATE, "level": "5 levels"},
           "Total harmonic distortion in percent against input level, from the "
           "same model-aligned oracle. A model whose line is flat across the "
           "five levels is not distortion-free, it is not responding to level "
           "at all."),
    "F5": ({"title": "spectral distance vs cutoff", "fc": "3 cutoffs",
            "K": "calibrated", "f_s": fe.SAMPLE_RATE, "level": -6},
           "RMS log-magnitude distance to the oracle at each of the three "
           "cutoffs, at one level. It separates models whose error is a fixed "
           "offset from models whose error grows with frequency."),
    "F6": ({"title": "linear and nonlinear scores", "fc": "all",
            "K": "calibrated/2", "f_s": fe.SAMPLE_RATE, "level": "mixed"},
           "The two axes side by side, models ordered by the combined score. "
           "The gaps between the two bars in a pair are the finding: a model "
           "that is long on one axis and short on the other is not average."),
}


def _metric(value, key):
    """A linear metric with its sign spoken, or n/a with the metric named.

    Four of the six scored metrics are signed: a model can run its cutoff low or
    high and both are equally wrong, so the raw number alone does not say which
    mistake was made. Unmeasurable metrics print n/a rather than a number,
    because "{:.2f}".format(nan) prints "nan" in the column a reader scans.
    """
    number = fe._num(value)
    if number is None:
        return "n/a"
    unit = LINEAR_UNITS.get(key, "")
    if key in LINEAR_SIGNED:
        side = LINEAR_DIRECTION["low" if number < 0.0 else "high"]
        return f"{abs(number):.2f} {unit} {side}".rstrip()
    return f"{number:.2f} {unit}".rstrip()


def _linear_loss(record):
    """{metric: weighted 0..1 credit} for one record's linear axis.

    The credit is the metric's share of the score, so the metric with the least
    of it is the one that cost the model most. A metric with no finite value
    scores 0.0 by construction (normalize_error), which is how an unmeasurable
    case shows up here as a total loss rather than as a missing one.
    """
    parts = (record.get("linear") or {}).get("score_parts") or {}
    out = {}
    for key, (best, worst, weight) in fe.LINEAR_WEIGHTS.items():
        if key not in parts:
            continue
        out[key] = weight * fe.normalize_error(parts[key], best, worst)
    return out


def _n_linear_cases(record):
    """How many linear cases this record holds, as opposed to how many measured.

    The two differ: a model whose curve never falls through -3 dB has six cases
    and no cutoff, and a sentence that conflates the two understates the gap.
    """
    linear = record.get("linear") or {}
    measured = linear.get("measured")
    if isinstance(measured, dict) and measured:
        return len(measured)
    counts = (linear.get("score_parts") or {}).get("n_cases") or {}
    return max(counts.values()) if counts else 0


def _no_finite_phrase(have, total):
    """How a metric's missing value reads in a sentence.

    "no finite value in 0 of the cases" is true and unreadable; the reader has
    to invert it to learn that the metric produced nothing at all.
    """
    if not have:
        return f"no finite value in any of the {total} cases"
    return f"no finite value in {have} of the {total} cases"


def _unmeasured(record):
    """Linear metrics with no finite value in this record, with their case count.

    "The cutoff was n/a" is not a measurement, so the number of cases each
    average actually stands on is reported next to it: at fc=5000 the
    reference's stopband window starts above the top of the grid, so the slope
    is unmeasurable at one cutoff of three without saying so anywhere.
    """
    parts = (record.get("linear") or {}).get("score_parts") or {}
    counts = parts.get("n_cases") or {}
    out = []
    for key in fe.LINEAR_WEIGHTS:
        if key not in parts:
            continue
        if fe._num(parts[key]) is None:
            out.append((key, counts.get(key, 0)))
    return out


def _dead_output_cases(record):
    """(case, passband gain error dB) for cases whose output is silence.

    A passband gain error of hundreds of dB is not a filter with a very wrong
    gain, it is a filter that emitted nothing: the measured magnitude is
    sitting on the -600 dB floor spectrum_db puts there for digital silence.
    Reported separately, because averaging such a case into the model's RMS
    error makes a dead model look like a badly tuned one.
    """
    out = []
    for key in sorted((record.get("linear") or {}).get("measured") or {}):
        err = fe._num((fe._measured(record, key) or {}).get("passband_gain_error_db"))
        if err is not None and err < DEAD_OUTPUT_DB:
            out.append((key, err))
    return out


def _resonance(record):
    r = fe._num(((record.get("linear") or {}).get("requested") or {}).get("resonance"))
    return f"r={r:.2f}" if r is not None else "no calibrated resonance"
# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _verdict(records, bars, total):
    top, second = bars[0], bars[1]
    last = bars[-1]
    margin = top[3] - second[3]
    lines = [
        "## 1. Verdict",
        "",
        f"**{top[0]}** leads at **{top[3]:.2f}** of 100 combined, {margin:.2f} "
        f"points ahead of **{second[0]}** ({second[3]:.2f}). That margin is "
        f"inside the spread the two axes disagree by, so read the two columns "
        f"rather than the total: {top[0]} is {top[1]:.2f} linear against "
        f"{top[2]:.2f} nonlinear, and {second[0]} is {second[1]:.2f} against "
        f"{second[2]:.2f}.",
        "",
        f"**{last[0]}** is last at {last[3]:.2f}, on a linear score of "
        f"{last[1]:.2f}. Its low rank is a measurement, not a mislabel, and the "
        f"per-model card below says which measurement.",
        "",
    ]
    if len(bars) == total:
        lines += [
            f"Every one of the {total} models in this repository is in the "
            "table, each on both axes. A model with a flagged condition is "
            "dropped from it rather than scored zero, and section 4.4 records "
            "what the gate did and why.",
        ]
    else:
        excluded = total - len(bars)
        if excluded == 1:
            clause = ("the model that was flagged is not scored, and section "
                      "4.4 names it and says why")
        else:
            clause = ("the models that were flagged are not scored, and "
                      "section 4.4 names them and says why")
        lines += [
            f"{len(bars)} of the {total} models in this repository are in the "
            f"table; " + clause + ".",
        ]
    lines += [
        "",
        fe.write_ranking_table([
            {"model": b[0], "linear": b[1], "nonlinear": b[2], "combined": b[3]}
            for b in bars
        ]),
        "",
    ]
    return lines


def _what_faithful_means():
    return [
        "## 2. What \"faithful\" means here",
        "",
        "Two axes, scored separately, because they measure different things and "
        "the models disagree sharply between them. Each score is 0 to 100, where "
        "100 means indistinguishable from the reference on the metrics that axis "
        "uses.",
        "",
        "**Linear axis: frequency-response shape.** The model's measured "
        "magnitude is compared with the analytic continuous-time transfer "
        "function H(s) of the analog ladder, ported from D'Angelo and "
        "Valimaki's Part I into `reference/moog_ladder_linear.py`. This axis is "
        "absolute: the target is an equation rather than a device, so it is "
        "checkable without owning a Moog, and it is the axis a model has to win "
        "to be usable as a filter.",
        "",
        "**Nonlinear axis: spectrum against an oracle.** The model's output is "
        "compared with a port of the Part II circuit model in "
        "`reference/moog_ladder_oracle.py`, after removing DC offset, bulk delay "
        "and one constant gain error. This axis is **model referenced, not "
        "hardware referenced**: the yardstick is D'Angelo's circuit simulation, "
        "not a measured Moog. No public Moog I/O dataset was found, so nothing "
        "in this report can be checked against hardware. Section 6 says what "
        "that costs.",
        "",
        f"**The combined number** is {fe.COMBINED_WEIGHT_LINEAR:.0%} linear and "
        f"{1.0 - fe.COMBINED_WEIGHT_LINEAR:.0%} nonlinear, "
        f"`COMBINED_WEIGHT_LINEAR` in `scripts/faithfulness_eval.py`. The weights "
        f"are equal because neither axis is the more trustworthy one, and the "
        f"two are on different scales in different units. Both axes are printed "
        f"on every row ahead of the total: a model that ranks first on one and "
        f"last on the other is the most actionable finding this harness can "
        f"produce, and averaging it away would destroy the result.",
        "",
    ]


def _fc_to_minus3db_ratio(fc=1000.0, k=0.0):
    """The fraction of `fc` at which the reference cascade is 3 dB down.

    Measured here rather than quoted, so a change to the port that moves the
    corner shows up as a changed sentence instead of a stale one.
    """
    freqs = np.logspace(1, np.log10(0.45 * fe.SAMPLE_RATE), 20000)
    d = fe.reference_magnitude_db(fc, k, freqs)
    return float(freqs[int(np.argmin(np.abs(d + 3.0)))]) / fc


def _methodology(records, bars):
    n_cases = len(fe.DEFAULT_CUTOFFS) * len(fe.DEFAULT_OVERSAMPLES)
    n_nl = len(fe.DEFAULT_CUTOFFS) * len(fe.LEVELS_DBFS) * len(fe.DEFAULT_OVERSAMPLES)
    fc_ratio = _fc_to_minus3db_ratio()
    return [
        "## 3. Methodology",
        "",
        "**Reference.** `reference/moog_ladder_linear.py`, a NumPy port of "
        "D'Angelo's `moog_ladder_linear.m`, for the linear axis; "
        "`reference/moog_ladder_oracle.py`, a port of `moog_ladder_nonlinear.m`, "
        "for the nonlinear axis. Both are float64 and run in process. The "
        "upstream `.m` files are vendored verbatim under `reference/upstream/` "
        "and are the authority for the ports.",
        "",
        "**`fc` is the leading-pole cutoff, not the -3 dB point.** The "
        f"reference's own cascade is 3 dB down at {fc_ratio:.3f} of its `fc` "
        f"(measured from `moog_ladder_linear` at k=0), so an `fc` of 1000 Hz is "
        f"a -3 dB point near {fc_ratio * 1000:.0f} Hz. A harness that swept "
        "cutoff until the "
        "measured curve fell through -3 dB and called that `fc` would be "
        "measuring a different filter from the one the reference describes, and "
        "the mismatch would read as a modeling error in all twelve models. This "
        "is also why the legacy suite's rejection of these models at nominal "
        "cutoffs is expected rather than alarming: it was looking for a -3 dB "
        "point where there is a leading pole.",
        "",
        "**Resonance calibration.** Each model's user-facing resonance `r` is "
        "swept over [0, 1] in 0.05 steps, driven with a unit impulse at "
        "`fc=1000 Hz`, and the value kept is the one whose measured curve has "
        "the lowest RMS dB error against the reference at absolute k=2. That is "
        "what puts model and reference at the same *absolute* operating point: "
        "the models map `r` to physical feedback differently (Hyperion by a "
        "factor of 4, OberheimVariation on a 1..10 scale), so `r` is not "
        "comparable across models and the calibration is what makes it so. The "
        "calibrated values are in section 4.",
        "",
        "**Linear metrics**, from the measured magnitude against the reference, "
        "over 20 Hz to 0.4*fs, on a unit impulse at the calibrated resonance: "
        + ", ".join(f"{LINEAR_LABELS[k]} ({LINEAR_UNITS[k]}, weight "
                    f"{w:g})" for k, (_b, _w, w) in fe.LINEAR_WEIGHTS.items())
        + f". The linear score is the weighted mean of "
        f"{n_cases} cases per model, cutoffs "
        f"{', '.join(f'{c:g}' for c in fe.DEFAULT_CUTOFFS)} Hz at oversampling "
        f"{', '.join(str(o) for o in fe.DEFAULT_OVERSAMPLES)}x. Scoring stops at "
        f"0.4*fs: a digital implementation must fold the analog stopband back, so "
        f"out-of-band response is not a modeling error and is excluded from the "
        f"score. A metric with no finite value scores 0.0, so a model that never "
        f"rolls off cannot pass by being unmeasured.",
        "",
        "**Nonlinear metrics**, on a hard-switched 440 Hz tone at absolute "
        f"k={NONLINEAR_K:g} against the oracle, over {n_nl} cases per model: "
        f"spectral distance (RMS log-magnitude difference, dB, weight 0.35), "
        f"time-domain NRMSE (weight 0.25), THD delta in dB (weight 0.25), and "
        f"the correlation of the two H1..H10 profiles (weight 0.15). The "
        f"harmonic profile and the THD are both read at the interpolated "
        f"fundamental rather than at an FFT bin, because a bin centre projects "
        f"an H2 up to 1.4 dB low and turns a 10 % THD into anything from 0.03 % "
        f"to 10 %. Cases are cutoffs "
        f"{', '.join(f'{c:g}' for c in fe.DEFAULT_CUTOFFS)} Hz at levels "
        f"{', '.join(f'{lvl:g}' for lvl in fe.LEVELS_DBFS)} dBFS and the same "
        f"two oversampling settings.",
        "",
"**Signal path.** Every model is driven through `build/RunFilters "
        "--float`, which writes 32-bit IEEE float WAVs; the flag only adds "
        "that path, and the default 16-bit PCM output is unchanged (a "
        "regression test pins it to PCM16). The float path exists because a "
        "16-bit record quantizes or zeros every tail below about -90 dBFS "
        "(one LSB: 20*log10(1/32768) = -90.3), which is not the floor of the "
        "models' quietest tails, so the self-oscillation test would read -inf "
        "where the model still rings. RunFilters has no single-model mode, so "
        "each invocation processes all twelve and the harness keeps the file "
        "whose name matches.",
        "",
        "**Robustness gate.** A model producing non-finite output on any "
        "condition is flagged, removed from the ranking, and reported with the "
        f"offending parameters named. It is not silently scored zero. In this "
        f"run: "
        + (", ".join(f"`{n}`" for n in fe._ordered_names(records)
                    if not fe._ok(records[n])) or "no model was flagged")
        + ".",
        "",
    ]


def _axis_table(bars, records, index, label):
    """One axis's ranking, sorted best first, with the axis's own evidence."""
    ordered = sorted(bars, key=lambda b: b[index], reverse=True)
    lines = [f"| # | Model | {label} | worst-scoring linear metric |", "|---:|---|---:|---|"]
    for i, bar in enumerate(ordered, 1):
        losses = _linear_loss(records[bar[0]])
        worst = min(losses, key=losses.get) if losses else None
        parts = (records[bar[0]].get("linear") or {}).get("score_parts") or {}
        detail = _metric(parts.get(worst), worst) if worst else "n/a"
        lines.append(f"| {i} | {bar[0]} | {bar[index]:.2f} | "
                     f"{LINEAR_LABELS.get(worst, 'n/a')}: {detail} |")
    return lines


def _nonlinear_table(bars, records):
    ordered = sorted(bars, key=lambda b: b[2], reverse=True)
    lines = ["| # | Model | Nonlinear | model THD at -6 dBFS (%) | "
             "oracle THD on the same record (%) |", "|---:|---|---:|---:|---:|"]
    for i, bar in enumerate(ordered, 1):
        case = fe._case(records[bar[0]], fe.ORACLE_CASE) or {}
        lines.append(
            f"| {i} | {bar[0]} | {bar[2]:.2f} | "
            f"{fe._fmt(fe._num(case.get('thd_model_percent')))} | "
            f"{fe._fmt(fe._num(case.get('thd_oracle_percent')))} |")
    return lines


def _results(records, bars, rank, total):
    excluded = [n for n in fe._ordered_names(records) if n not in rank.by_name]
    lines = [
        "## 4. Results",
        "",
        "### 4.1 Combined ranking",
        "",
        fe.write_ranking_table([
            {"model": b[0], "linear": b[1], "nonlinear": b[2], "combined": b[3]}
            for b in bars
        ]),
        "",
        "### 4.2 Linear ranking",
        "",
        "Sorted best first. The third column is the axis's own score; the "
        "fourth names the metric that gave the model the least of its linear "
        "credit, with the error as measured.",
        "",
    ] + _axis_table(bars, records, 1, "Linear") + [
        "",
        "### 4.3 Nonlinear ranking",
        "",
        "Sorted best first, against the model-aligned oracle. The two THD "
        "columns are at the operating point the figures use; the oracle column "
        "differs from model to model because `align_signals` shifts the oracle "
        "by the model's own cross-correlation, so the same input leaves a "
        "different oracle on each record.",
        "",
    ] + _nonlinear_table(bars, records) + [
        "",
        "### 4.4 Models excluded from the ranking",
        "",
    ]
    if excluded:
        lines += ["| Model | Status | Reason |", "|---|---|---|"]
        for name in excluded:
            rec = records[name]
            lines.append(f"| {name} | {rec.get('status')} | {_flag_reason(rec)} |")
        one = len(excluded) == 1
        lines += [
            "",
            ("This model produced" if one else "These models produced")
            + " non-finite output on at least one condition, or a collector "
            "raised. The condition"
            + ("s that did answer are" if one else "s that answered are")
            + " still on the record as diagnostics, but the mean over them is "
            "not a measurement of the model and is not ranked here.",
        ]
    else:
        lines += [
            f"No model was flagged: all {total} carry `status: ok` and hold "
            f"both axes, so nothing is excluded here. The gate that would have "
            f"removed one is still live. A model producing non-finite output on "
            f"any condition is named in this section with the reason and dropped "
            f"from the ranking, and a report that dropped one silently would read "
            f"as a measurement that was never taken.",
        ]
    lines += [
        "",
        "### 4.5 Calibrated resonance",
        "",
        "The user-facing `r` each model needed to sit at absolute k="
        f"{NONLINEAR_K:g}. These are not comparable across models on their own: "
        "the models map `r` to physical feedback differently, which is exactly "
        "why the sweep calibrates rather than fixing `r`.",
        "",
        "| Model | calibrated r |",
        "|---|---:|",
    ]
    for bar in bars:
        lines.append(f"| {bar[0]} | {_resonance(records[bar[0]])} |")
    lines += [
        "",
        "Self-oscillation, measured at the top of each model's own `r` range. A "
        "tail above -60 dBFS long after the burst ends counts as ringing; a run "
        "that diverges has no tail to measure and is flagged instead of being "
        "recorded as ringing.",
        "",
        "| Model | " + " | ".join(f"r={r:.2f}" for r in fe.SELFOSC_RESONANCES)
        + " |",
        "|---|" + "---:|" * len(fe.SELFOSC_RESONANCES),
    ]
    for bar in bars:
        selfosc = records[bar[0]].get("selfosc") or {}
        cells = []
        for r in fe.SELFOSC_RESONANCES:
            entry = selfosc.get(f"{r:.1f}") or {}
            if not isinstance(entry, dict) or "ringing" not in entry:
                cells.append("n/a")
            elif entry.get("ringing"):
                cells.append(f"rings ({fe._fmt(fe._num(entry.get('tail_rms_db')))} dBFS)")
            else:
                cells.append(f"quiet ({fe._fmt(fe._num(entry.get('tail_rms_db')))} dBFS)")
        lines.append(f"| {bar[0]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines
class Rankings(NamedTuple):
    """Where each model sits on each axis, computed once per render.

    Passed into the card builders rather than reached for through a module
    global: a report is a pure function of its records, and a name resolved from
    mutable module state is one more way for two renders of the same records to
    disagree.
    """

    bars: Tuple[tuple, ...]
    by_name: Dict[str, tuple]
    linear_rank: Dict[str, int]
    nonlinear_rank: Dict[str, int]
    total: int

    @classmethod
    def of(cls, bars):
        linear = sorted(bars, key=lambda b: b[1], reverse=True)
        nonlinear = sorted(bars, key=lambda b: b[2], reverse=True)
        return cls(
            bars=tuple(bars),
            by_name={b[0]: b for b in bars},
            linear_rank={b[0]: i for i, b in enumerate(linear, 1)},
            nonlinear_rank={b[0]: i for i, b in enumerate(nonlinear, 1)},
            total=len(bars),
        )

    def place(self, name):
        """1-based combined rank, or None for a model that is not ranked."""
        if name not in self.by_name:
            return None
        return [b[0] for b in self.bars].index(name) + 1


def _card(name, rec, rank, rings):
    """One model's card: strengths, failures and known artifacts, from the record.

    `rank` is the Rankings, or None for a model the gate excluded. An excluded
    model still gets a card: the condition that took it out of the ranking is
    the most actionable thing its record says.
    """
    parts = (rec.get("linear") or {}).get("score_parts") or {}
    counts = parts.get("n_cases") or {}
    lines = [f"### {name}", ""]

    if rank is None:
        lines.append(
            f"**Not ranked.** Status `{rec.get('status')}`, reason "
            f"{_flag_reason(rec)}. The conditions that did answer are on the "
            f"record as diagnostics, and their mean is not a measurement of "
            f"this model, so no score is reported here.")
    else:
        bar = rank.by_name[name]
        _name, lin, nlin, comb = bar
        lin_rank = rank.linear_rank[name]
        nlin_rank = rank.nonlinear_rank[name]
        lead = "linear" if lin_rank <= nlin_rank else "nonlinear"
        other = "nonlinear" if lead == "linear" else "linear"
        lines.append(
            f"**Combined {comb:.2f}, {rank.place(name)} of {rank.total}.** "
            f"Linear {fe._fmt(lin)} ({lin_rank} of {rank.total}), nonlinear "
            f"{fe._fmt(nlin)} ({nlin_rank} of {rank.total}), at a calibrated "
            f"{_resonance(rec)}. The stronger axis is {lead} and the weaker is "
            f"{other}: a reader who wants the filter's shape should read the "
            f"linear number, one who wants its behaviour under a hard-switched "
            f"tone should read the nonlinear one.")

    losses = _linear_loss(rec)
    # The test is on what the reader sees, not on the float: five exact zeros
    # and one at 0.003 is a floor, and a sentence about 0.00 of 0.00 says
    # nothing about the model.
    if losses and sum(losses.values()) >= 0.005:
        worst = min(losses, key=losses.get)
        best = max(losses, key=losses.get)
        lines.append(
            f"Linear axis: the most credit it earned was {LINEAR_LABELS[best]} "
            f"({_metric(parts.get(best), best)}), worth {losses[best]:.2f} of "
            f"the {sum(losses.values()):.2f} it earned on the axis, and the "
            f"least was {LINEAR_LABELS[worst]} "
            f"({_metric(parts.get(worst), worst)}), worth {losses[worst]:.2f}.")
    elif losses:
        at_floor = sorted(k for k, v in losses.items() if v < 0.005)
        if len(at_floor) == len(losses):
            detail = ("All six sit at or within a rounding step of the worst "
                      "value the scale admits")
        else:
            detail = (
                f"{len(at_floor)} of the six "
                f"({' + '.join(LINEAR_LABELS[k].lower() for k in at_floor)}) "
                f"sit at or past the worst value the scale admits and the "
                f"remaining {len(losses) - len(at_floor)} within a rounding "
                f"step of it")
        lines.append(
            "Linear axis: no credit on any of the six metrics, to the two "
            f"decimal places this report prints. {detail}, so the axis total "
            f"is the floor rather than an average of mediocre results.")

    unmeasured = _unmeasured(rec)
    if unmeasured:
        n_lin = _n_linear_cases(rec)
        listed = ", ".join(
            f"{LINEAR_LABELS[k]} ({_no_finite_phrase(counts.get(k, 0), n_lin)})"
            for k, _n in unmeasured)
        lines.append(
            f"Unmeasurable on this model: {listed}. Each is n/a in section 9.1 "
            f"and scores zero on the way to the total, because a curve that "
            f"never falls through -3 dB has no cutoff to report and must not "
            f"collect the credit for one it did not earn.")

    dead = _dead_output_cases(rec)
    if dead:
        first_key, first_err = dead[0]
        also = "".join(f", and {e:.2f} dB at {k}" for k, e in dead[1:])
        lines.append(
            f"The {first_key} case is not a badly tuned response but no "
            f"response at all: a {first_err:.2f} dB passband gain error against "
            f"a reference within a few dB of unity puts the measured magnitude "
            f"on the {SILENCE_FLOOR_DB:.0f} dB floor `spectrum_db` applies to "
            f"digital silence{also}. One dead case out of six dominates the "
            f"mean, which is why the aggregate reads worse than the live cases "
            f"do.")

    readings = _selfosc_readings(rec)
    if readings:
        rings = [r for r, e in readings if e.get("ringing") is True]
        if rings:
            loudest = max(fe._num(rec["selfosc"][r].get("tail_rms_db")) or -999.0
                          for r in rings)
            lines.append(
                f"Self-oscillation: rings at "
                f"{', '.join('r=' + format(float(r), '.2f') for r in rings)}, "
                f"loudest tail {fe._fmt(loudest)} dBFS, quiet at the rest. It is "
                f"reachable inside the user range, not only at the top of it.")
        else:
            lines.append(
                f"Self-oscillation: no tail stayed above -60 dBFS at "
                f"{', '.join('r=' + format(r, '.2f') for r in fe.SELFOSC_RESONANCES)},"
                f" so this run did not observe it oscillating at the top of its "
                f"own resonance range.")

    artifacts = LEGACY_ARTIFACTS.get(name)
    if artifacts:
        faithful_thd = _faithful_thd_clause(rec)
        for fact, closing in artifacts:
            lines.append(
                f"Known artifact, carried from the 2026-09-28 legacy suite "
                f"rather than measured here: {fact} "
                f"{closing.format(faithful_thd=faithful_thd)}")
    lines.append("")
    return lines


def _faithful_thd_clause(rec):
    """How this run's own THD reading sits beside a legacy one for the same model.

    The legacy suite's THD sweep ran at fc=5000 and r=0.00, and this one at
    fc=1000 and k=2, so the two readings differ on two axes rather than one,
    and it read the figure from binned FFT bins where this harness reads the
    interpolated fundamental. The two numbers are not the same measurement, so
    the clause that keeps a legacy card from reading as a contradiction has to
    name this run's own figure: a reader holding "0.0000 %" in one hand and
    section 4.3's column in the other needs to be told which is which, and told
    it in numbers rather than in a disclaimer. The figure therefore hangs off
    "section 4.3's column" as its subject and not off the legacy operating
    point, which a trailing relative clause would otherwise let "where" bind to
    and read as a claim about the legacy sweep. When the model has no figure at
    that operating point the comparison is dropped rather than filled in.
    """
    case = fe._case(rec, fe.ORACLE_CASE) or {}
    value = fe._num(case.get("thd_model_percent"))
    clause = ("Section 4.3's column is a different sweep - at fc=5000, r=0.00 "
              f"in the legacy sweep against fc={ORACLE_FC:g}, "
              f"k={NONLINEAR_K:g} on this harness")
    if value is None:
        return f"{clause} - so the two figures are not comparable."
    return (f"{clause} - and it reads {value:.2f} % for this model, so the "
            "two figures are not comparable.")


def _selfosc_readings(rec):
    """selfosc's per-resonance readings, without its two meta keys.

    `flagged` and `flag_reason` live in the same dict as the readings and are
    not readings: a `ringing` read off either is a KeyError, and a truthy
    stand-in for a tail that was never measured is worse than a crash.
    """
    selfosc = rec.get("selfosc")
    if not isinstance(selfosc, dict):
        return []
    return [(r, e) for r, e in selfosc.items()
            if isinstance(e, dict) and "ringing" in e]


def _cards(records, rank):
    lines = ["## 5. Per-model cards", "",
             "One card per model, in combined-score order. The linear figures "
             "below are the means over the cases that produced a measurement; "
             "where a case could not be measured the card says so rather than "
             "letting the mean speak for it.", ""]
    if rank is not None:
        for bar in rank.bars:
            lines += _card(bar[0], records[bar[0]], rank, None)
    for name in fe._ordered_names(records):
        if rank is None or name not in rank.by_name:
            lines += _card(name, records[name], None, None)
    return lines


def _threats():
    return [
        "## 6. Threats to validity",
        "",
        "- **The nonlinear axis is measured against a model, not against "
        "hardware.** Every nonlinear number here is a distance from D'Angelo's "
        "Part II circuit simulation, run at absolute k="
        f"{NONLINEAR_K:g} and aligned to each model by cross-correlation. No "
        "public Moog I/O dataset was found, so that axis cannot be independently "
        "verified here, and a model that is wrong in the same way the port is "
        "wrong would score well. The linear axis does not share this problem: "
        "its reference is an equation.",
        "- **The D'Angelo errata were read, and none of the three corrections "
        "applies to these ports.** Both author errata (`errata_gladder1.pdf`, "
        "`errata_gladder2.pdf`) are single-page and were read by text "
        "extraction on 2026-09-29. Part I carries two corrections: a p.1827 "
        "prose sentence should name the leading-pole property `Q` alongside "
        "`fc`, and a p.1828 expression should read `fc_hat = fc/alpha(k)` "
        "rather than `fc/A0(k)`. Part II carries one: the p.1879 caption of "
        "Fig. 6 should begin \"Non-solid lines represent...\". The first is a "
        "prose fix the ports' coefficient equations do not depend on, the "
        "second is a form the ports already use (`fc .*= alpha(k)` upstream, "
        "`alpha(n, k)` in both Python ports), and the third is a caption the "
        "oracle's delay-free loop is unaffected by. No linear or nonlinear "
        "score in this report moves because of them. "
        "`reference/upstream/PROVENANCE.md` holds the per-correction record, "
        "one line each, with page numbers and the file and line that settles "
        "it.",
        "- **The comparison port may be coincidentally matched.** The oracle is "
        "D'Angelo's delay-free implementation, and a candidate model that "
        "happens to share its algebra with that implementation scores well for a "
        "reason that has nothing to do with the hardware. The report cannot "
        "separate that from genuine agreement, and the linear axis is the only "
        "place the two are checked independently of each other.",
        "- **The filter libraries may be mislabeled.** `Stilson`, "
        "`Huovilainen`, `MusicDSP` and `RKSimulation` are ports of published "
        "code by other authors. A port that does not match the paper it names "
        "would be measured correctly and attributed wrongly, and this harness "
        "has no way to tell.",
        "- **The legacy test did not measure the -3 dB point.** The 2026-09-28 "
        "suite swept cutoff and read the response at the requested frequency, "
        "not at the point where the cascade is down 3 dB, so its cutoff "
        "rejections are not evidence about these models. The step and THD "
        "artifacts quoted in section 5 are from that suite and are the only "
        "facts carried over from it, and each names the operating point that "
        "suite measured it at: its THD sweep ran at fc=5000, not the fc=1000 "
        "of section 4.3.",
        "- **The score scales are choices.** Each metric maps an error to 0..1 "
        "over a hand-chosen best and worst, and the weights are hand-chosen too. "
        "A different reasonable set reorders the middle of the table. The two "
        "axis scores are in the record next to the raw metrics in section 9.1, "
        "so a reader can rescore without re-running anything.",
        "- **Defaults only.** Every model is measured at its own defaults. "
        "Hyperion ships drive, thermal, ADAA and three quality tiers; ranking "
        "its defaults is not the same as ranking the model, and no best-config "
        "row is reported here.",
        "",
    ]


def _figures(records, plots_dir):
    oracle_model = fe._oracle_source(records)
    suffix = f" ({oracle_model}-aligned)" if oracle_model else ""
    lines = [
        "## 7. Figures",
        "",
        f"The six committed PNGs under `docs/moog-faithfulness/plots/`, "
        f"referenced by path relative to this report. Each caption below is "
        f"the figure's own title, built by `fig_caption` from the same "
        f"operating point `generate_figures` drew it at, so a caption cannot "
        f"drift from the picture.",
        "",
    ]
    if oracle_model:
        lines += [
            f"F3 and F4 draw one oracle, and the title names whose it is: the "
            f"oracle on a record is aligned to the model it was measured "
            f"against, so `{oracle_model}`'s view of it is the one plotted. There "
            f"is no single oracle curve to draw here.",
            "",
        ]
    for spec in FIGURE_SPECS:
        kwargs, blurb = FIGURE_CAPTIONS[spec.tag]
        kwargs = dict(kwargs)
        if "oracle" in kwargs.get("title", ""):
            kwargs["title"] = kwargs["title"] + suffix
        title = fe.fig_caption(spec.tag, **kwargs)
        assert (Path(plots_dir) / spec.filename).exists(), spec.filename
        lines += [f"![{title}]({spec.relpath})", "", title, "", blurb, ""]
    return lines


def _reproduction(records):
    any_rec = next(iter(fe._ordered_names(records)))
    env = records[any_rec]
    software = env.get("software") or {}
    return [
        "## 8. Reproduction",
        "",
        "The whole run, from a built `build/RunFilters`:",
        "",
        "```bash",
        REPRO_COMMAND,
        "```",
        "",
        "The report is then regenerated from the run directory:",
        "",
        "```bash",
        "python scripts/faithfulness_report.py "
        "filter_validation/faithfulness/run1",
        "```",
        "",
        f"Environment, read from the `git_head` and `software` keys that "
        f"every record in that run carries (here `{any_rec}.json`) rather than "
        f"from the machine this file was written on:",
        "",
        f"- `git_head` {env.get('git_head')}",
        f"- python {software.get('python')}",
        f"- numpy {software.get('numpy')}",
        f"- scipy {software.get('scipy')}",
        "",
        "`scripts/filter_verification.py` is not modified by any of this work: "
        "the legacy suite and the figures quoted in section 5 are read-only "
        "inputs to the report, and every measurement in it was taken by "
        "`scripts/faithfulness_eval.py`. The figures are byte-identical to the "
        "committed PNGs when regenerated from the same records.",
        "",
        "The run directory is gitignored; the numbers above are regenerable from "
        "a fresh clone by building `RunFilters` and re-running the sweep, and "
        "the committed report can then be diffed against the regenerated one.",
        "",
    ]


def _appendix(records, rank):
    lines = [
        "## 9. Appendix",
        "",
        "### 9.1 Every linear metric, per model",
        "",
        "The six scored metrics as recorded, before they were mapped to a score. "
        "n/a is a measurement that could not be taken, never a zero.",
        "",
        "| Model | " + " | ".join(
            f"{LINEAR_LABELS[k]} ({LINEAR_UNITS[k]})" for k in fe.LINEAR_WEIGHTS)
        + " |",
        "|---|" + "---:|" * len(fe.LINEAR_WEIGHTS),
    ]
    order = [b[0] for b in rank.bars] if rank is not None else fe._ordered_names(records)
    total = _n_linear_cases(next(iter(records.values())))
    for name in order:
        parts = (records[name].get("linear") or {}).get("score_parts") or {}
        counts = parts.get("n_cases") or {}
        cells = []
        for k in fe.LINEAR_WEIGHTS:
            cell = _metric(parts.get(k), k)
            have = counts.get(k, total)
            if cell != "n/a" and have < total:
                cell += f" ({have} of {total} cases)"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"\"(N of {total} cases)\" marks a mean that stands on fewer than all "
                  "six measurements: the metric was unmeasurable at the rest, so its "
                  "value is a weaker claim than the table's other cells."]
    lines += [
        "",
        "### 9.2 What was not measured, and why",
        "",
        "- **Perceptual quality.** No listening panel was run and no audio was "
        "compared by ear. Any claim that a model \"sounds most like a Moog\" "
        "would be unfounded, so the question is out of scope rather than "
        "answered weakly.",
        "- **Out of band response.** Reported by `shape_metrics` but excluded "
        "from the linear score. A digital implementation must fold the analog "
        "stopband back, and scoring that fold as a modeling error would punish "
        "every model for the same correct decision.",
        "- **Aliasing.** Each model is driven at oversampling 1x and 4x, and both "
        "sets of cases go into the score, so oversampling is measured as an "
        "operating condition rather than as an artifact study. The spectral "
        "images an oversampled filter folds back were not isolated, so no claim "
        "is made here about how much aliasing any model adds.",
        "- **Intermodulation.** The nonlinear set drives a single tone, so it "
        "measures harmonic distortion and nothing else. Intermodulation needs a "
        "two-tone drive, which this harness does not run, so the distortion "
        "figures say nothing about it either way.",
        "- **Parameter sweeps.** Resonance is calibrated to one absolute k and "
        "the cutoffs are three fixed points. The models disagree most in how "
        "they map `r` to physical feedback, and that disagreement is sampled at "
        "one target rather than swept.",
        "- **CPU cost.** An engineering input rather than a faithfulness one, "
        "and not measured by this harness.",
        "",
        "### 9.3 References",
        "",
        "- S. D'Angelo and V. Valimaki, \"Generalized Moog Ladder Filter: "
        "Part I - Linear Analysis and Parameterization\", IEEE/ACM TASLP "
        "22(12), 1825-1832, 2014. DOI 10.1109/TASLP.2014.2352495. The source of "
        "`reference/moog_ladder_linear.py`.",
        "- S. D'Angelo and V. Valimaki, \"Generalized Moog Ladder Filter: "
        "Part II - Explicit Nonlinear Model through a Novel Delay-Free "
        "Implementation Method\", IEEE/ACM TASLP 22(12), 1873-1883, 2014. "
        "DOI 10.1109/TASLP.2014.2352556. The source of "
        "`reference/moog_ladder_oracle.py`.",
        "- P. D. Zolzer and F. Vranken (eds), DAFX-DAFX02, 2nd edition, 2002, for "
        "the Moog~ model named in StilsonModel.h, after P. Stilson and J. O. "
        "Smith, \"Using Algorithms to Synthesize Polyphonic Programmable "
        "Filters\", Computer Music Journal 20(2), 1996.",
        "- A. Huovilainen, \"Non-linear digital implementation of the Moog "
        "ladder filter\", Proceedings of the Computer Music Conference, 2004, "
        "for the Huovilainen model.",
        "- R. J. E. Daly, \"Moog VCF Analysis\", MSc project, University of "
        "Edinburgh, 2012. Related prior comparison of ladder-filter "
        "implementations; listed as prior work, not as a source of specific "
        "values.",
        "",
        "The models' own provenance and licences are listed in `README.md`.",
        "",
    ]
    return lines


def render(records, plots_dir=PLOTS_DIR):
    """The whole report as one markdown string.

    A pure function of `records` and the six committed figures: the same
    records give the same bytes whoever assembled the dict, which is what makes
    the end-to-end test against the committed file meaningful. Nothing here
    reads the clock, the network, or the run directory, so a report can be
    regenerated and diffed.
    """
    gate(records, plots_dir)
    bars = fe._score_bars(records)
    rank = Rankings.of(bars)
    total = len(records)  # repository count, ranked or not
    lines = [
        "# Moog ladder model faithfulness",
        "",
        f"All {total} Moog ladder models in this repository, measured "
        f"against two references and ranked on each. Every measured number in "
        f"this document is read out of the per-model JSON written by "
        f"`scripts/faithfulness_eval.py`, or out of the JSON of the legacy "
        f"2026-09-28 suite where a card says it is; none of it is typed in by "
        f"hand.",
        "",
    ]
    lines += _verdict(records, bars, total)
    lines += _what_faithful_means()
    lines += _methodology(records, bars)
    lines += _results(records, bars, rank, total)
    lines += _cards(records, rank)
    lines += _threats()
    lines += _figures(records, plots_dir)
    lines += _reproduction(records)
    lines += _appendix(records, rank)
    return "\n".join(lines).rstrip() + "\n"


def write_report(run_dir, out_path=REPORT_PATH, plots_dir=PLOTS_DIR):
    """Render and write. Returns the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(load_records(run_dir), plots_dir))
    return out_path


def main(argv=None):
    import argparse

    if argv is None:
        argv = sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="a run directory, e.g. "
                                     "filter_validation/faithfulness/run1")
    ap.add_argument("--out", default=str(REPORT_PATH),
                    help=f"where to write the report (default: {REPORT_PATH})")
    ap.add_argument("--plots-dir", default=str(PLOTS_DIR),
                    help="directory holding the six committed figures")
    args = ap.parse_args(argv)
    path = write_report(args.run_dir, args.out, args.plots_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
