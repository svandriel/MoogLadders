# Upstream reference code

These Octave/MATLAB files are vendored verbatim from Stefano D'Angelo's public
site. They are the authority for the Python ports in `reference/`. Do not edit
them. When changing a port, diff against these.

| file | source |
| --- | --- |
| `moog_ladder_linear.m` | https://dangelo.audio/assets/code/moog_ladder_linear.m |
| `moog_ladder_nonlinear.m` | https://dangelo.audio/assets/code/moog_ladder_nonlinear.m |
| `moog_ladder_old.m` | https://dangelo.audio/assets/code/moog_ladder_old.m |

Retrieved 2026-09-28. MIT licensed, Copyright (C) 2014 Stefano D'Angelo. The
license text is reproduced in the header of each file.

Papers:

- Part I: S. D'Angelo and V. Valimaki, "Generalized Moog Ladder Filter:
  Part I-Linear Analysis and Parameterization", IEEE/ACM TASLP, 22(12),
  1825-1832, 2014. DOI 10.1109/TASLP.2014.2352495
- Part II: S. D'Angelo and V. Valimaki, "Generalized Moog Ladder Filter:
  Part II-Explicit Nonlinear Model through a Novel Delay-Free Implementation
  Method", IEEE/ACM TASLP, 22(12), 1873-1883, 2014.
  DOI 10.1109/TASLP.2014.2352556

## Errata status

Author-published errata:

- https://dangelo.audio/assets/doc/errata_gladder1.pdf
- https://dangelo.audio/assets/doc/errata_gladder2.pdf

STATUS: read. Both PDFs are single-page and machine-readable; they were read on
2026-09-29 by text extraction (`pypdf`). Each of the three corrections was then
checked against the ports in `reference/`, and all three do not apply:

- Part I, p.1827 (errata_gladder1.pdf, correction 1): the sentence listing the
  controllable parameters `Ictl` and `k` and the leading-pole properties should
  name `fc` and `Q` (for N >= 2) rather than trailing off at `fc` and the case
  condition. Does not apply, because it is a prose correction in the published
  text: the port implements the coefficient equations (`biquad_coeffs`, `alpha`,
  `knorm_factor` in `reference/moog_ladder_linear.py`), which this sentence does
  not change, and Q is not a parameter the port computes. The symbols in the
  `Q` mention at `reference/moog_ladder_linear.py:126-127` are the quadratic of
  an analog prototype section used in a derivation comment, not a pole Q.
- Part I, p.1828 (errata_gladder1.pdf, correction 2): the expression
  `fc_hat = fc/A0(k)` should read `fc_hat = fc/alpha(k)`. Does not apply,
  because the ports already use the corrected alpha(k) form everywhere:
  upstream `fc .*= alpha(k)` at `reference/upstream/moog_ladder_linear.m:96`,
  and `alpha(n, k)` in `reference/moog_ladder_linear.py:40` and
  `reference/moog_ladder_oracle.py:65`. The erroneous `A0(k)` form appears
  nowhere in the ladder ports; the `A0`/`A02` identifiers at
  `reference/upstream/moog_ladder_linear.m:137-141` are the bilinear-transform
  intermediates of the response computation, and the `A0` in
  `scripts/generate_halfband_coeffs.py` and `src/HalfBandFilter.h` is an
  unrelated allpass filter.
- Part II, p.1879 (errata_gladder2.pdf, correction 1): the caption of Fig. 6
  should begin "Non-solid lines represent...". Does not apply, because it is a
  figure-caption correction: `reference/moog_ladder_oracle.py` implements the
  delay-free loop equations, which a caption does not change.

No correction applies to a port, so no measured number in
`docs/moog-faithfulness-report.md` moves. The report's threats-to-validity
section states this outcome.
