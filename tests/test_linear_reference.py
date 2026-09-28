import math

import numpy as np
import pytest

from reference import moog_ladder_linear as mll

N = 4
FS = 44100.0


def test_knorm_factor_is_four_for_n4():
    assert mll.knorm_factor(4) == pytest.approx(4.0, abs=1e-12)


def test_alpha_at_zero_k_is_one():
    assert mll.alpha(4, 0.0) == pytest.approx(1.0, abs=1e-12)


def test_alpha_at_k_four_is_one():
    # alpha(4,4) = sqrt(1 + 4^0.5 - 2*4^0.25*cos(pi/4)) = 1
    assert mll.alpha(4, 4.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("w", [0, 1])
def test_aw_matches_closed_form(w):
    expected = math.sqrt(
        1.0
        + 2.0 ** (2.0 / N)
        - 2.0 * 2.0 ** (1.0 / N) * math.cos((2 * w + 1) * math.pi / N)
    )
    assert mll.aw(N, w, 2.0) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("w", [0, 1])
def test_bw_matches_closed_form(w):
    expected = 1.0 - 2.0 ** (1.0 / N) * math.cos((2 * w + 1) * math.pi / N)
    assert mll.bw(N, w, 2.0) == pytest.approx(expected, rel=1e-12)


def _mag_db(fc, k, freqs):
    return mll.magnitude_db(N, FS, fc, k, freqs)


@pytest.mark.parametrize("fc", [100.0, 1000.0, 5000.0])
def test_dc_gain_is_exactly_one(fc):
    # Measured during planning: section DC gain came out to 1.0000000000000024,
    # unity to float64 round-off. A wrong cosine argument in aw/bw breaks this.
    # The probe frequency is 1e-4 Hz, close enough to DC that the true near-DC
    # rolloff (~1.7e-11 dB at fc=100) is two orders below this test's tolerance.
    # (At a probe of 1e-3 Hz the fc=100 case sits at -1.76e-9 dB of genuine
    # rolloff, marginally past the 1e-9 dB bound.)
    assert _mag_db(fc, 0.0, np.array([1e-4, 1.0, 10.0]))[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("fc", [100.0, 1000.0, 5000.0])
def test_response_is_monotonically_decreasing_at_zero_k(fc):
    freqs = np.logspace(1, np.log10(0.45 * FS), 2000)
    d = _mag_db(fc, 0.0, freqs)
    assert np.all(np.diff(d) <= 1e-9)


def test_fc_to_minus3db_ratio_is_pinned():
    """fc is the leading-pole cutoff, not the -3 dB point of the cascade.

    Measured during planning at k=0: the ratio is 0.4342 at fc=100, 0.4348 at
    fc=1000 and 0.4496 at fc=5000, so it is mildly fc-dependent. This test
    records the fc=1000 value so any future port change is caught. The
    evaluation code must never assume the ratio is 1.
    """
    freqs = np.logspace(1, np.log10(0.45 * FS), 200000)
    d = _mag_db(1000.0, 0.0, freqs)
    f3 = freqs[int(np.argmin(np.abs(d + 3.0)))]
    assert f3 / 1000.0 == pytest.approx(0.4348, abs=5e-4)


def test_k_four_is_marginal_stability_with_poles_at_the_corner():
    """At absolute k=4 the 4-pole ladder is marginally stable.

    The analog transfer function is H(s) = wc**4 / ((s+wc)**4 + k*wc**4). At k=4
    the denominator (s+wc)**4 + 4*wc**4 has all four roots on the imaginary axis
    at s = +/- j*wc, so the pole frequency equals the corner frequency.

    Verified during planning: with fc=1000 and wc = 2*fs*tan(pi*fc/fs), the roots
    at k=4 all sit at 1001.6951 Hz against a corner of 1000 Hz.

    This is the guard against a wrong k convention. If normalized and absolute get
    swapped anywhere, this fails and every resonance number in the report is wrong.
    """
    fc = 1000.0
    wc = 2.0 * FS * np.tan(np.pi * fc / FS)
    roots = np.roots([1.0, 4.0 * wc, 6.0 * wc**2, 4.0 * wc**3, wc**4 * 5.0])
    pole_hz = np.sort(np.abs(roots.imag)) / (2.0 * np.pi)
    assert len(pole_hz) == 4
    assert np.allclose(pole_hz, wc / (2.0 * np.pi), rtol=1e-9)


def test_ported_coeffs_match_independent_analog_derivation():
    """The ported cascade must equal the analog prototype it discretizes.

    analog_response evaluates the ladder's continuous-time transfer function in
    closed form in k, without touching aw, bw or biquad_coeffs, and maps the
    requested frequencies onto the analog axis with the inverse bilinear warp.
    The ported cascade is a prewarped bilinear transform of that prototype, so the
    two must agree. They agree to 1.5e-13 dB at k=0, 1.4e-13 dB at k=2 and
    2.7e-11 dB at k=3.99, so the 1e-3 dB bound below is eight orders of magnitude
    of headroom.

    Two wrong versions of this check were tried and rejected while writing it,
    and neither should be reintroduced:

    - Comparing against H = 1/((1+s/wc)**4 + k), the closed loop of four unity
      one-poles with loop gain k. That prototype is not the one the paper
      factors. The two coincide at k=0 and k=4 and part company by up to 9.6 dB
      in the midband in between, so this version fails on a correct port.
    - Evaluating the prototype at 2*pi*f as if f were an analog frequency.
      biquad_coeffs is a bilinear transform, so the analog frequency for digital
      f is fs/pi*tan(pi*f/fs); skipping the warp is worth up to 23 dB of
      spurious error in the upper stopband.

    The -120 dB band mask from the original brief is kept as cheap insurance
    against differencing two numerically-zero stopbands in dB, which produces
    meaningless tens-of-dB errors. It is no longer load-bearing: with the warp the
    analog magnitude is evaluated at |u| < 90 across this sweep, so the stopband
    stays well away underflow.
    """
    fc = 1000.0
    freqs = np.logspace(1, np.log10(0.45 * FS), 4000)
    for k in (0.0, 1.0, 2.0, 3.0, 3.99):
        ported = mll.magnitude_db(N, FS, fc, k, freqs)
        analog = 20.0 * np.log10(np.abs(mll.analog_response(fc, FS, k, freqs)))
        band = ported > -120.0
        assert np.max(np.abs(ported[band] - analog[band])) < 1e-3


def test_peak_gain_grows_toward_k_four():
    freqs = np.logspace(1, np.log10(0.45 * FS), 20000)
    peak3 = _mag_db(1000.0, 3.0, freqs).max()
    peak399 = _mag_db(1000.0, 3.99, freqs).max()
    assert peak399 > peak3
    assert peak399 > 40.0