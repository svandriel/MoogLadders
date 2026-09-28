# Which digital model actually sounds like a Moog ladder?

A research design for ranking the 12 digital ladder models in this repo by how faithfully they
emulate the analog 4-pole Moog transistor ladder.

Date: 2026-09-28
Status: design approved, not yet implemented

## Goal

Produce an engineering-facing Markdown report that ranks all 12 models on linear and nonlinear
faithfulness, with enough method detail that a reader can trust or reject the ranking, and
enough practical detail to decide which model to ship.

The reader is a developer choosing a model. The report leads with the ranking and a per-model
verdict, and puts method in a later section.

## Why the existing verification suite cannot answer this

`scripts/filter_verification.py` was run end to end on 2026-09-28 (run `2026-09-28_201435`,
396 cases, 499 plots). It produces numbers, but they are not measurements of the filters. The
defects below are confirmed against the run's own JSON output, not inferred from reading code.

**The cutoff frequency measurement is wrong for every model.** Requested and measured corner
frequencies at `oversample=0`:

| requested (Hz) | measured range across 12 models |
| --- | --- |
| 50 | 0.7 to 24.2 |
| 200 | 0.7 to 80.1 |
| 1000 | 107.7 to 386.3 |
| 12000 | 71.3 to 21403.3 |

Not one model lands on its requested corner. The `passband_ripple_db` column gives the mechanism
away: it falls monotonically from 150.3 dB at 50 Hz requested to 0.1 dB at 12000 Hz requested, so
the measurement window is anchored to a constant frequency band rather than to each filter's own
corner. As the corner moves, more of a fixed window lands in the passband and the reported figure
follows. Code reading indicates the grid is indexed over a fixed 0 to `0.45*fs` band; the exact
line is pinned down in the implementation plan rather than relied on here, since the symptom above
is measured and the line number is not.

`passband_ripple_db` shows the same signature: 150.3 dB at 50 Hz requested, 0.1 dB at 12000 Hz
requested. A real ladder's passband flatness does not change by 150 dB with cutoff. The metric is
measuring window placement, not the filter. `stopband_atten_db` spans 122 to 231 dB, which is
below and above the 16-bit WAV floor respectively and therefore measures the output file format,
not the filter.

**The self-oscillation test is inert.** All 36 entries report `rms_dbfs = -240.0` and
`oscillating = False`. The test feeds digital silence, so it measures nothing.

**The THD test is measuring quantization noise.** At -6 dBFS input:

| model | THD |
| --- | --- |
| Stilson | 0.0000 % |
| Huovilainen | 0.0023 % |
| RKSimulation | 0.3553 % |
| OberheimVariation | 0.5569 % |
| Krajeski | 1.1533 % |
| MusicDSP | 1.9277 % |
| Microtracker | 3.5357 % |
| HyperionLegacy | 5.7137 % |
| Simplified | 5.9100 % |
| HyperionTanh | 6.7693 % |
| Hyperion | 8.3879 % |
| Improved | 8.9453 % |

Two signals that this ordering is backwards. Stilson is a purely linear model, so near-zero THD
is expected for it and uninformative. Huovilainen is a circuit-derived nonlinear model and
should show more distortion than the linear one, yet it reports 0.0023 %. Both values sit at the
16-bit WAV noise floor, which for a full-scale 16-bit file is about 0.0015 % THD. For the models
whose real distortion is above that floor the numbers are usable, but the ordering at the bottom
of the table is floor artifacts.

**Step settling time is a measurement-length artifact.** The reported settling times cluster at
742 to 743 ms, which is the length of the test signal. The metric is reporting "never settled"
as a number. Overshoot values are also suspect: 180.5 % for MusicDSP at Q=0.9 against 63.5 % for
HyperionLegacy at Q=0.0.

**WAV output is 16-bit integer**, which imposes about 96 dB SNR and caps any distortion
measurement from below.

**There is no ground truth anywhere in the suite.** Every test compares the filter to itself
measured a second way. Nothing in the run tells you whether a model is right, only whether it is
self-consistent.

Conclusion: the existing suite stays as a smoke test. It is not modified, and none of its numbers
feed the ranking.

## Repairs made to reach a runnable state

Commit `a968ec6` registered `HyperionLegacyMoog` but did not add `src/HyperionLegacyModel.h`, so
the project did not build. `src/HyperionLegacyModel.h` was restored from `a968ec6^` with the
class, constructor, and destructor renamed and the include guard changed to
`HYPERION_LEGACY_LADDER_H`. The DSP body is byte identical. `.gitignore` now excludes `.venv/`,
`filter_validation/`, and `__pycache__/`. Both binaries build and `RunFilters --bench` exercises
all 12 models.

## Approach

Two axes, scored separately, because they measure different things and models disagree sharply
between them.

**Linear axis.** Analytic comparison against the continuous-time transfer function from
D'Angelo and Valimaki's Part I analysis. Absolute, verifiable, no oracle needed.

**Nonlinear axis.** Waveform differencing against a port of their Part II circuit model. This axis
is model referenced, not hardware referenced. The report states this wherever the axis is used.

The design rejected two alternatives. Absolute targets lifted from papers alone leave most metrics
unmeasurable and cannot catch errors nobody published. A single oracle distance score hides
whether a model is wrong or merely different from D'Angelo.

## Components

### 1. `reference/moog_ladder_linear.py`

Numpy port of `moog_ladder_linear.m`, giving exact continuous-time `H(s)` magnitude and phase for
given `fc`, `K`, `stages=4`. No I/O. This is the linear ground truth.

### 2. `reference/moog_ladder_oracle.py`

Numpy port of `moog_ladder_nonlinear.m` (Part II), float64, callable sample by sample in-process.
This is the nonlinear yardstick.

### 3. `src/` changes

Model code is untouched. `RunFilters` gains an opt in `--float` flag that writes IEEE float32
WAVs with `audioFormat=3`. The existing reader in `example/helpers.hpp` already handles that
format, so no reader change is needed. The default 16-bit path must produce byte identical
output with the flag off, and that is verified.

### 4. `scripts/faithfulness_eval.py`

Signal generation, alignment, metric extraction, scoring, ranking, figure generation. Writes JSON,
plots, and a ranking table. Leaves `scripts/filter_verification.py` alone.

### 5. `docs/moog-faithfulness-report.md`

The deliverable.

## Data flow

```
test signal (numpy float64)
   |-> oracle ----------------> reference output
   |-> float32 WAV -> RunFilters --float -> one WAV per model
                                              |
                    align: DC offset, bulk delay, constant gain error
                                              |
              +-------------------------------+--------------------+
              |                                                    |
        linear axis                                         nonlinear axis
        vs analytic H(s)                                    vs oracle output
        continuous-time                                    float64
              |                                                    |
              +----------------------> score -> rank -> report <----+
```

## Fairness: why the linear axis stops at 0.4 fs

`H(s)` is continuous-time and the models are discrete-time. Above roughly `0.4*fs` a digital model
*must* depart from the analog curve, because the analog filter's stopband folds back into the
digital one. That is a property of digital implementation, not a modeling error.

So the linear score is computed in band only, up to `0.4*fs`. Out of band response is reported
separately as digital-domain behavior and excluded from the score.

## The calibration problem

Each model's `SetResonance(r)` maps `r` in [0,1] to an internal feedback gain `K` differently.
Comparing "model at r=0.5" against "reference at K=2" double-penalizes a model with a
mis-calibrated resonance control, once for the wrong shape and once for the wrong tuning.

So the linear axis splits in two.

### L1, tuning accuracy

Sweep `r` per model and find the value that best matches the reference at a nominal point
(K=2, fc=1 kHz). Report the offset as a calibrated gain and frequency error. Separately score:

- the k to Q law, whether peak gain holds as K rises
- tuning stability across cutoff
- the K at which self-oscillation begins, compared to the analytic marginal stability point

### L2, shape accuracy

At the calibrated operating point, compare the full measured response against `H(s)`:

- `magnitude_rms_db_error` over a log sweep, 20 Hz to 0.4 fs
- `cutoff_3db_error_hz` and `_cents`
- `passband_gain_error_db` at DC against a 1.0 linear gain
- `stopband_slope_error_db_per_oct`, fitted, against the expected -24 dB/oct
- `phase_error_deg` at anchor frequencies
- `peak_gain_error_db` and `peak_freq_error_cents` at K=2 and K=4

## Nonlinear axis

Model output and oracle output receive the same signal, then gain and delay alignment.

Test signals: sine sweeps at -24, -18, -12, -6 and -3 dBFS; two-tone for intermodulation; step;
steady sine near self-oscillation at K=3.5 and 3.9; above onset at K=4.2.

Metrics:

- `spectral_distance_db` and `time_domain_nrmse`
- `thd_delta_db` and `imd_delta_db` against the oracle at each level
- `harmonic_profile_corr` over H2 to H10
- `selfosc_freq_error_cents` and `selfosc_amplitude_error_db`
- `step_overshoot_delta_pct` and a settling time with a documented criterion, on a signal long
  enough that "never settled" is distinguishable from a number

## Scoring and ranking

Each metric normalizes to 0 to 100, where 100 means indistinguishable from the reference. Each
model gets a `linear_score` from L1 and L2, a `nonlinear_score`, and a `combined` at 50/50 by
default.

The report shows both axis rankings side by side ahead of the combined number. A model that ranks
first linear and eleventh nonlinear is the most actionable finding this report can produce, and
averaging it away would destroy the result.

Not scored, each with a stated reason:

- **CPU cost.** An engineering input, not faithfulness. Reported in its own column, using the
  measurements already collected: MusicDSP 18 ns/sample, HyperionTanh 246 ns/sample.
- **Perceptual quality.** No listening panel was run. Any claim that a model "sounds most like a
  Moog" would be unfounded, so it is out of scope and the report says so.
- **Out of band response.** Reported as a note, excluded from score, per the Nyquist argument.

**Robustness gate.** The README warns that some models blow up when parameters exceed
undiscovered limits. Any model producing non-finite output on any condition is flagged, removed
from the ranking, and reported with the offending parameters named. It is not silently scored
zero and it does not abort the run.

## Validation, before anything is scored

The reference implementation is the single point of failure. A porting error in it is
indistinguishable from a modeling error in the models being ranked, so it is validated three ways
and the official errata are applied before it is allowed to score anything.

1. Two independent derivations of `H(s)` agree to better than 1e-12: direct polynomial evaluation
   against a cascaded-section evaluation.
2. The K=4 identity holds. The 4-pole ladder goes marginally stable at K=4, with its poles on the
   imaginary axis at plus or minus j times the cutoff angular frequency, so peak gain diverges and
   the corner sits at `wc`. A port that violates this has the wrong k to Q convention and the
   whole resonance comparison is inverted.
3. Published figure values from Part I and Part II are reproduced, with `errata_gladder1.pdf` and
   `errata_gladder2.pdf` applied.

The evaluation code is also self-validated before scoring models: a synthetic signal with
analytically known THD must have its THD recovered exactly, and a null run establishes the
empirical noise floor with float32 output. If the code cannot recover a known number, no ranking
is credible.

## Risks

**Porting error in the reference.** Covered by the three validation gates above. This is the risk
that would silently corrupt everything.

**Parameterization ambiguity.** The literature uses `K`, `k`, and `4k` for related quantities
across different papers. Getting this wrong inverts the resonance comparison. The K=4 identity
is the guard.

**Defaults underselling a model.** Hyperion has drive, thermal, ADAA, and three quality tiers,
all defaulting to one setting. Ranking defaults alone would show that Hyperion's defaults are
worse, which is not the same as showing Hyperion is worse. Every model with tunable extras also
gets a best-config run, reported as a clearly labeled second row.

**Oversampling confound.** Each model is reported at default oversampling and at `os=4`, since
the oversampling base is shared.

**`--float` regression.** Opt in only, and byte-identical default output is verified.

**Scope.** The parameter grid is fixed in this design rather than discovered during the run.

## Report structure

1. **Verdict.** Ranking table, both axes, one line per model on whether to ship it and for what.
2. **What faithful means here.** The two axes defined.
3. **Methodology.** Reference, calibration, metrics, weights.
4. **Results.** Linear ranking, nonlinear ranking, combined, side by side.
5. **Per-model cards.** Strengths, failures, and known artifacts. Three are already confirmed from
   the 2026-09-28 run: Stilson reports `dc_gain = 0.0000` at all three cutoffs, emitting literal
   zeros to a step; `Improved` reports `dc_gain = -1.0000`, inverting the signal; `HyperionLegacy`
   never settles within a 743 ms window.
6. **Threats to validity.** Leads with the honest one, that the nonlinear axis is measured
   against a model rather than hardware. No public Moog I/O dataset was found, so that axis
   cannot be independently verified here.
7. **Reproduction.** Exact commands.
8. **Appendix.** Full metric tables, references, and what was not measured.

The report cites prior art, D'Angelo and Valimaki, Daly 2012, Stilson and Smith, and Huovilainen,
and states where this work agrees or disagrees with them.

## Figures

Plots are committed under `docs/moog-faithfulness/plots/` and referenced by relative path, because
a report whose images live in a gitignored run directory renders on one machine and nowhere else.
The generated run directory and its 27 MB dashboard stay gitignored and are mentioned only as an
optional local deep dive with the command to regenerate them.

Format is PNG at roughly 200 dpi, total committed figure size budgeted at 2 to 3 MB. Six figures:

| id | figure | purpose |
| --- | --- | --- |
| F1 | magnitude response, all 12 plus analytic `H(s)`, faceted at K=0, 2, 4 | headline shape accuracy |
| F2 | peak gain versus K, all 12 plus reference | k to Q law and tuning quality |
| F3 | harmonic spectrum at -6 dBFS, oracle plus models | nonlinear character |
| F4 | THD versus input level, oracle line plus extremes | level dependence |
| F5 | self-oscillation onset, amplitude and frequency versus K near 4 | where models diverge most |
| F6 | linear versus nonlinear score bars per model | the verdict, visually |

Every caption states its operating point, `fc`, `K`, `fs`, and input level. A ladder response plot
without those is not interpretable, so captions are part of the deliverable. Each figure carries
alt text.

## Sources

- D'Angelo and Valimaki publications index: https://dangelo.audio/publications
- linear reference code: https://dangelo.audio/assets/code/moog_ladder_linear.m
- nonlinear reference code: https://dangelo.audio/assets/code/moog_ladder_nonlinear.m
- Part I errata: https://dangelo.audio/assets/doc/errata_gladder1.pdf
- Part II errata: https://dangelo.audio/assets/doc/errata_gladder2.pdf
- Huovilainen, DAFx-04: https://dafx.de/paper-archive/2004/P_061.PDF
- Stilson and Smith: https://ccrma.stanford.edu/~stilti/papers/moogvcf.pdf
- Helie, DAFx-06: https://dafx.de/paper-archive/2006/papers/p_007.pdf
- Werner and McClellan, DAFx-20: https://dafx2020.mdw.ac.at/proceedings/papers/DAFx2020_paper_70.pdf
- Daly 2012, Edinburgh MSc project: http://www.acoustics.ed.ac.uk/wp-content/uploads/AMT_MSc_FinalProjects/2012__Daly__AMT_MSc_FinalProject_MoogVCF.pdf
- MoogVCFTrap, chadmckell: https://github.com/chadmckell/MoogVCFTrap

Daly 2012 sits behind a bot check and its body text has not been read, so it is listed as a
related prior comparison and not cited as a source of specific values.

## Out of scope

- Modifying `scripts/filter_verification.py`
- Listening tests or perceptual claims
- Scoring CPU cost
- Models outside the 12 already in the repo
- Claiming hardware-referenced accuracy anywhere in the report

## Definition of done

Reference validated three ways, including the K=4 identity. Evaluation code recovers known values
from synthetic signals. All 12 models scored on both axes, with robustness failures named rather
than hidden. Report written with the threats to validity section intact, six committed figures,
captions stating operating points, and exact reproduction commands.
