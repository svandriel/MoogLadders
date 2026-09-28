## Copyright (C) 2014 Stefano D'Angelo
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.

## -*- texinfo -*-
## @deftypefn {Function File} {@var{y} =} moog_ladder_nonlinear (@var{fs}, @var{N}, @var{x}, @var{fc}, @var{k}, @var{fccomp}, @var{knorm}, @var{gaincomp})
##
## Simulate the generalized Moog ladder filter as described in
## @quotation
## S. D'Angelo and V. Valimaki, ``Generalized Moog Ladder Filter:
## Part II-Explicit Nonlinear Model through a Novel Delay-Free Implementation
## Method,'' @emph{IEEE/ACM Trans. Audio, Speech, and Lang. Process.}, vol. 22,
## no. 12, pp. 1873-1883, December 2014.
## @end quotation
##
## It produces the output vector @var{y} from the audio input vector @var{x}.
## @var{fs} is the sample rate (in Hz) and @var{N} is the filter order.
##
## @var{fc} and @var{k} are, respectively, the cutoff frequency (in Hz) and
## global feedback gain, and they can either be scalars or vectors whose length
## is equal to the length of @var{x}.  If @var{fccomp} is true, @var{fc}
## represents the actual cutoff frequency of the leading poles (@math{f_c}),
## otherwise it represents the natural cutoff frequency (@math{\widehat{f_c}).
## If @var{knorm} is true, @var{k} is the normalized feedback gain value,
## otherwise it is the absolute feedback gain value.
##
## @var{gaincomp} can be: 1 (no gain compensation), 2 (dc gain compensation),
## or 3 (cutoff-slope gain compensation).
##
## @end deftypefn

## Author: Stefano D'Angelo <zanga.mail@gmail.com>
## Maintainer: Stefano D'Angelo <zanga.mail@gmail.com>
## Version: 1.0.0
## Keywords: moog ladder filter lowpass resonant vcf nonlinear

function y = moog_ladder_nonlinear(fs, N, x, fc, k, fccomp, knorm, gaincomp)

  ### Constant values

  VT = 26e-3;

  ### Input arguments

  # Checks

  if (nargin != 8)
    print_usage();
  elseif (!isscalar(fs) || !isnumeric(fs) || !isreal(fs) || fs <= 0)
    error("moog_ladder_nonlinear: FS must be a positive real");
  elseif (!isscalar(N) || !isnumeric(N) || !isreal(N) || isinf(N)
          || floor(N - 1) != abs(N - 1))
    error("moog_ladder_nonlinear: N must be a finite positive integer");
  elseif (isscalar(x) || !isvector(x) || !isnumeric(x) || !isreal(x))
    error("moog_ladder_nonlinear: X must be a vector of real values");
  elseif (!isnumeric(fc) || !isreal(fc)
          || (!isscalar(fc) && !isvector(fc))
          || (!isscalar(fc) && isvector(fc)
              && length(fc) != length(x)))
    error(["moog_ladder_nonlinear: FC must be a scalar or a vector of real " ...
           "values whose length is equal to the length of X"]);
  elseif (!isnumeric(k) || !isreal(k)
          || (!isscalar(k) && !isvector(k))
          || (!isscalar(k) && isvector(k)
              && length(k) != length(x)))
    error(["moog_ladder_nonlinear: K must be a scalar or a vector of real " ...
           "values whose length is equal to the length of X"]);
  elseif (!isscalar(gaincomp)
          || (gaincomp != 1 && gaincomp != 2 && gaincomp != 3))
    error("moog_ladder_nonlinear: GAINCOMP must be 1, 2, or 3");
  endif

  # Convert to row vectors

  if (rows(fc) > columns(fc))
    fc = fc';
  endif
  if (rows(k) > columns(k))
    k = k';
  endif

  # Compensate/normalize parameters

  if (!fccomp)
    fc .*= alpha(k);
  endif

  if (knorm && N <= 2)
    warning(["moog_ladder_linear: KNORM cannot be true when N <= 2, " ...
             "ignoring it"]);
  elseif (knorm)
    k .*= sec(pi / N) ^ N;
  endif

  # Enforce bounds

  fc = limit(fc, 0, fs / 2, "FC");
  k = limit(k, 0, inf, "K");

  # Convert parameters to vectors

  if (isscalar(fc))
    fc = repmat(fc, 1, length(x));
  endif
  if (isscalar(k))
    k = repmat(k, 1, length(x));
  endif

  ### Coefficients

  g    = tan(pi / fs * fc) ./ alpha(N, k);
  VT2  = 2 * VT;
  VT2i = 1 / VT2;

  # Ladder stages

  p0s = 1 ./ (1 + g);
  q0s = 1 - g;
  r1s = -g;
  k0s = VT2 * g .* p0s;

  # Global filter

  gN  = (1 - p0s) .^ N;  # == (g ./ (g + 1)) .^ N
  kgN = k .* gN;
  p0g = 1 ./ (1 + kgN);
  bin = bincoeff(N, 1:N)';
  rg  = -bin * kgN;
  qg  = rg - repmat(bin, 1, length(x)) ...
             .* (repmat((g - 1) .* p0s, N, 1) .^ repmat([1:N]', 1, length(x)));
  k0g = -VT2i * p0g;

  ### Filter

  y  = zeros(1, length(x));
  si = zeros(1, N);
  sf = zeros(1, N);
  sg = zeros(1, N);
  for i = 1:length(x)
    yo = tanh(k0g(i) * (x(i) + sg(1)));

    for n = 1:N
      yi   = yo;
      yd   = k0s(i) * (yi + sf(n));
      y(i) = yd + si(n);
      yo   = tanh(VT2i * y(i));

      si(n) = yd + y(i);
      sf(n) = r1s(i) * yi - q0s(i) * yo;
    endfor

    yf = k(i) * y(i);
    for n = 1:(N-1)
      sg(n) = rg(n, i) * x(i) + qg(n, i) * yf + sg(n + 1);
    endfor
    sg(N) = rg(N, i) * x(i) + qg(N, i) * yf;
  endfor

  ### Gain compensation

  if (gaincomp == 2)
    y .*= 1 + k;
  elseif (gaincomp == 3)
    y .*= alpha(N, k) .^ N;
  endif

endfunction

function y = alpha(N, k)
  if (N == 1)
    y = 1 + k;
  else
    y = Aw(N, 0, k);
  endif
endfunction

function y = Aw(N, w, k)
  y = sqrt(1 + k .^ (2 / N) - 2 * k .^ (1 / N) .* cos(((2 * w + 1) * pi) / N));
endfunction

function y = Bw(N, w, k)
  y = 1 - k .^ (1 / N) .* cos(((2 * w + 1) * pi) / N);
endfunction

function y = limit(x, low, up, name)
  y = x;
  if (min(y) < low || max(y) > up)
    warning(["moog_ladder_nonlinear: limiting " name " to [" num2str(low) ...
             ", " num2str(up) "]"]);
    y(y < low) = low;
    y(y > up)  = up;
  endif
endfunction
