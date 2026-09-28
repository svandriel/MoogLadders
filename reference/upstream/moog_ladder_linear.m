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
## @deftypefn {Function File} {@var{y} =} moog_ladder_linear (@var{fs}, @var{N}, @var{x}, @var{fc}, @var{k}, @var{fccomp}, @var{knorm}, @var{gaincomp})
##
## Simulate the generalized Moog ladder filter as described in
## @quotation
## S. D'Angelo and V. Valimaki, ``Generalized Moog Ladder Filter: Part I-Linear
## Analysis and Parameterization,'' @emph{IEEE/ACM Trans. Audio, Speech, and
## Lang. Process.}, vol. 22, no. 12, pp. 1825-1832, December 2014.
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
## Keywords: moog ladder filter lowpass resonant vcf linear

function y = moog_ladder_linear(fs, N, x, fc, k, fccomp, knorm, gaincomp)

  ### Input arguments

  # Checks

  if (nargin != 8)
    print_usage();
  elseif (!isscalar(fs) || !isnumeric(fs) || !isreal(fs) || fs <= 0)
    error("moog_ladder_linear: FS must be a positive real");
  elseif (!isscalar(N) || !isnumeric(N) || !isreal(N) || isinf(N)
          || floor(N - 1) != abs(N - 1))
    error("moog_ladder_linear: N must be a finite positive integer");
  elseif (isscalar(x) || !isvector(x) || !isnumeric(x) || !isreal(x))
    error("moog_ladder_linear: X must be a vector of real values");
  elseif (!isnumeric(fc) || !isreal(fc)
          || (!isscalar(fc) && !isvector(fc))
          || (!isscalar(fc) && isvector(fc)
              && length(fc) != length(x)))
    error(["moog_ladder_linear: FC must be a scalar or a vector of real " ...
           "values whose length is equal to the length of X"]);
  elseif (!isnumeric(k) || !isreal(k)
          || (!isscalar(k) && !isvector(k))
          || (!isscalar(k) && isvector(k)
              && length(k) != length(x)))
    error(["moog_ladder_linear: K must be a scalar or a vector of real " ...
           "values whose length is equal to the length of X"]);
  elseif (!isscalar(gaincomp)
          || (gaincomp != 1 && gaincomp != 2 && gaincomp != 3))
    error("moog_ladder_linear: GAINCOMP must be 1, 2, or 3");
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
  if (N >= 2)
    k = limit(k, 0, sec(pi / N) ^ N, "K");
  else
    k = limit(k, 0, inf, "K");
  endif

  # Convert parameters to vectors

  if (isscalar(fc) && !isscalar(k))
    fc = repmat(fc, 1, length(x));
  elseif (!isscalar(fc) && isscalar(k))
    k = repmat(k, 1, length(x));
  endif

  ### Coefficients

  D  = tan(pi / fs * fc);

  W = floor(N / 2);
  if (W != 0)
   A = zeros(W, length(k));
   B = zeros(W, length(k));
   for w = 1:W
     A(w, :) = Aw(N, w - 1, k);
     B(w, :) = Bw(N, w - 1, k);
   endfor
   D2   = repmat(D .* D, W, 1);
   AD2  = (A .* A) .* D2;
   A0BD = (repmat(A(1, :) .* D, W, 1)) .* B;
   A02  = repmat(A(1, :) .* A(1, :), W, 1);
   E    = AD2 + 2 * A0BD + A02;
   a1   = 2 * (AD2 - A02) ./ E;
   a2   = 1 - 4 * A0BD ./ E;
   b0   = D2 ./ E;
  endif

  if (fmod(N, 2) == 1)
    a     = alpha(N, k);
    D1kN  = D .* (1 + k .^ (1 / N));
    den   = D1kN + a;
    b0odd = D ./ den;
    a1odd = (D1kN - a) ./ den;
  endif

  ### Filter

  if (isscalar(fc) && isscalar(k))

    # Time-invariant case

    for w = 1:W
      y = filter(b0(w) * [1 2 1], [1 a1(w) a2(w)], x);
      x = y;
    endfor
    if (fmod(N, 2) == 1)
      y = filter(b0odd, [1 a1odd], x);
    endif

  else

    # Time-varying case

    y = zeros(1, length(x));
    for w = 1:W
      # TDF-II
      b0x = b0(w, :) .* x;
      s1  = 0;
      s2  = 0;
      for i = 1:length(x)
        y(i) = s1 + b0x(i);
        s2   = b0x(i) - a2(w, i) * y(i);
        s1   = b0x(i) + b0x(i) + s2 - a1(w, i) * y(i);
      endfor
      x = y;
    endfor
    if (fmod(N, 2) == 1)
      sx  = 0;
      sy  = 0;
      for i = 1:length(x)
        y(i) = b0odd(i) * (x(i) + sx) - a1odd(i) * sy;
        sx   = x(i);
        sy   = y(i);
      endfor
    endif

  endif

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
    warning(["moog_ladder_linear: limiting " name " to [" num2str(low) ", " ...
             num2str(up) "]"]);
    y(y < low) = low;
    y(y > up)  = up;
  endif
endfunction
