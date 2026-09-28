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
## @deftypefn {Function File} {@var{y} =} moog_ladder_old (@var{fs}, @var{x}, @var{fc}, @var{k})
##
## Simulate the Moog ladder filter as described in
## @quotation
## S. D'Angelo and V. Valimaki, ``An improved virtual analog model of the Moog
## ladder filter,'' in @emph{Proc. Intl. Conf. on Acoust., Speech, and Signal
## Process. (ICASSP 2013)}, pp. 729-723, Vancouver, Canada, May 2013.
## @end quotation
##
## It produces the output vector @var{y} from the audio input vector @var{x}.
## @var{fs} is the sample rate (in Hz).
##
## @var{fc} and @var{k} are, respectively, the cutoff frequency (in Hz) and
## global feedback gain, and they can either be scalars or vectors whose length
## is equal to the length of @var{x}.
##
## @end deftypefn

## Author: Stefano D'Angelo <zanga.mail@gmail.com>
## Maintainer: Stefano D'Angelo <zanga.mail@gmail.com>
## Version: 1.0.0
## Keywords: moog ladder filter lowpass resonant vcf nonlinear

function y = moog_ladder_old(fs, x, fc, k)

  ### Input arguments

  # Checks

  if (nargin != 4)
    print_usage();
  elseif (!isscalar(fs) || !isnumeric(fs) || !isreal(fs) || fs <= 0)
    error("moog_ladder_old: FS must be a positive real");
  elseif (isscalar(x) || !isvector(x) || !isnumeric(x) || !isreal(x))
    error("moog_ladder_old: X must be a vector of real values");
  elseif (!isnumeric(fc) || !isreal(fc)
          || (!isscalar(fc) && !isvector(fc))
          || (!isscalar(fc) && isvector(fc)
              && length(fc) != length(x)))
    error(["moog_ladder_old: FC must be a scalar or a vector of real " ...
           "values whose length is equal to the length of X"]);
  elseif (!isnumeric(k) || !isreal(k)
          || (!isscalar(k) && !isvector(k))
          || (!isscalar(k) && isvector(k)
              && length(k) != length(x)))
    error(["moog_ladder_old: K must be a scalar or a vector of real values " ...
           "whose length is equal to the length of X"]);
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

  VT   = 26e-3;
  VT2i = 1 / (2 * VT);
  X    = pi * fc / fs;
  g    = X .* (1 - X) ./ (1 + X);
  g1   = -g;
  g4   = 2 * VT * g;

  ### Filter

  y   = zeros(1, length(x));
  st1 = 0;
  st2 = 0;
  st3 = 0;
  st4 = 0;
  sd1 = 0;
  sd2 = 0;
  sd3 = 0;
  sd4 = 0;
  sy  = 0;
  for i = 1:length(x)
    d1   = g1(i) * (tanh(VT2i * (x(i) + k(i) * sy)) + st1);
    v1   = d1 + sd1;
    sd1  = d1 + v1;
    st1  = tanh(v1);

    d2   = g(i) * (st1 - st2);
    v2   = d2 + sd2;
    sd2  = d2 + v2;
    st2  = tanh(v2);

    d3   = g(i) * (st2 - st3);
    v3   = d3 + sd3;
    sd3  = d3 + v3;
    st3  = tanh(v3);

    d4   = g4(i) * (st3 - st4);
    v4   = d4 + sd4;
    sd4  = d4 + v4;
    st4  = tanh(VT2i * v4);

    y(i) = v4;
    sy   = v4;
  endfor

endfunction

function y = limit(x, low, up, name)
  y = x;
  if (min(y) < low || max(y) > up)
    warning(["moog_ladder_old: limiting " name " to [" num2str(low) ...
             ", " num2str(up) "]"]);
    y(y < low) = low;
    y(y > up)  = up;
  endif
endfunction
