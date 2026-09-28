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

STATUS: not yet read. These PDFs were not machine-readable in this environment.
A human must read both and record, one line per correction, either "applies to
our port, fixed in <file>" or "does not apply, because <reason>". The report's
threats-to-validity section must state the outcome plainly, including "could not
be read". Do not leave this STATUS line in place silently.
