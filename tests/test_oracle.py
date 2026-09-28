import numpy as np
import pytest

from reference import moog_ladder_linear as mll
from reference import moog_ladder_oracle as oracle

FS = 44100.0
N = 4


def test_zero_input_gives_zero_output():
    y = oracle.process(np.zeros(1000), FS, 1000.0, 2.0, N)
    assert np.all(y == 0.0)


def test_output_is_finite_and_bounded_for_small_input():
    x = 0.01 * np.sin(2 * np.pi * 440.0 * np.arange(4410) / FS)
    y = oracle.process(x, FS, 1000.0, 2.0, N)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 1.0


@pytest.mark.parametrize("fc", [200.0, 1000.0, 5000.0])
def test_dc_gain_is_one_at_zero_k(fc):
    """At k=0 the DC gain is -1 exactly, inverted: a 0.05 input settles to -0.05.

    0.05 is NOT the linear region, and this test does not rely on it being one.
    The input stage is tanh(VT2i * x) with VT2i = 19.23, so 0.05 is a 0.9615
    drive, u = tanh(0.9615) = 0.745, and every cascade tanh sits at that same u.
    The result is still exact because the fixed point inverts the saturation back
    out: at DC each stage is a DC follower (yo = yi, forced by the yd = 0
    equilibrium that boundedness of si requires), and atanh(tanh(z)) = z, so the
    saturated 0.745 maps back to the state y = -p0g * x. Instrumented at fc=200,
    k=0: the four internal nodes all settle at exactly -0.05, gain -1.00000000.

    0.05 is the right probe because the only cost of that saturation is numerical.
    atanh is conditioned by (1 - u**2)**-1 = 2.25x at u = 0.745, which is why the
    measured error below is ~1e-15 rather than ~1e-16, still twelve orders inside
    the 1e-3 tolerance the assertion allows. Raise the drive and two separate
    things break. Settling time explodes as dtanh/dv -> 0 shrinks the effective
    loop gain: the law is still exact at 0.1, first drifts by 1.2% at 0.15 and by
    19% at 0.2 on a 20000-sample window, and 0.2 does reach exactly -0.2 given 2e6
    samples instead. Then, past that, the DC-follower fixed point stops existing
    altogether: 0.5 pins at -0.2840 (a 43% gain error) and 1.0 pins at the same
    -0.2840, never converging at any sample count tried. 0.05 is 2x below the
    first breakdown and 10x below the hard one, and its measured error leaves the
    assertion twelve orders of headroom.

    What this test does and does not pin. It pins the SIGN, the unity gain
    magnitude, and the structural relations r1s = -g, q0s = 1-g and
    k0g = -VT2i * p0g. It does NOT pin the value of g: the DC law collapses to
    exactly -1/(1+k) for ANY g, because the yd = 0 equilibrium forces the stage
    relations no matter what the coefficients are. Deleting the 1/alpha factor
    from g outright, a 100% error at k > 0 and a literal no-op at k = 0, leaves
    this test passing. p0s and k0s are equally invisible to it. The only test in
    this file that exercises the value of g is
    test_oracle_converges_to_linear_reference_for_tiny_signals. Read this as a
    polarity and topology pin, not a coefficient pin.

    The sign is NEGATIVE and that is the upstream model's, not a port artifact.
    Four independent pieces of evidence, all recorded here because the sign is
    easy to "fix" by accident and must never be flipped silently:

    - moog_ladder_nonlinear.m line 147 sets k0g = -VT2i * p0g, and line 156 applies
      it as yo = tanh(k0g * (x + sg(1))). tanh is odd, so that is an inverting
      input stage, and each ladder stage has DC gain +1 (at DC the state si
      accumulates 2*yd every sample, so the equilibrium forces yd = 0, hence
      sf = -yi, hence yo = yi). Four stages of +1 times one inversion of -1.
    - The vendored 2013 model reference/upstream/moog_ladder_old.m derives the
      identical law. Its states sd1..sd4 also accumulate 2*d per sample, so the
      DC equilibrium forces d1..d4 = 0; that makes st1=st2=st3=st4=s and
      v1=v2=v3=v4=atanh(s), and since y = v4 the stage-1 equation
      d1 = g1 * (tanh(VT2i * (x + k*sy)) + st1) = 0 gives
      tanh(VT2i * (x + k*y)) = -s, hence x + k*y = -atanh(s)/VT2i = -y and
      y = -x/(1+k). Note g1 = -g multiplies the whole parenthesised sum
      tanh(VT2i*(x + k*sy)) + st1, not the tanh alone. Porting its loop and
      measuring gives -0.05, -0.025, -0.016667, -0.0125 at k = 0, 1, 2, 3:
      the same -1/(1+k), to 8e-15. Two upstream files, one law.
    - The oracle's complex response is exactly -1e-5 times the Part I reference
      response over the whole in-band sweep (pinned by
      test_oracle_converges_to_linear_reference_for_tiny_signals below), so the
      inversion is a clean global sign and nothing else differs.
    - The upstream gaincomp == 2 branch multiplies by (1 + k). That compensates a
      POSITIVE DC gain of 1/(1+k); the model is expected to be a lowpass that
      passes DC, and the sign is a property of the ladder's current-steering
      node convention, shared by both upstream files.

    Measured relative error against -0.05: 4e-14 at fc=200, 1e-14 at fc=1000,
    2e-15 at fc=5000.
    """
    x = np.full(20000, 0.05)
    y = oracle.process(x, FS, fc, 0.0, N)
    assert y[-1000:].mean() == pytest.approx(-0.05, rel=1e-3)


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_dc_gain_is_one_over_one_plus_k(k):
    """DC gain is exactly -1/(1+k), the same law as the linear reference.

    This is the load-bearing companion to test_dc_gain_is_one_at_zero_k: the
    k=0 case cannot see the global feedback at all, because at k=0 both rg and the
    k*yf term that drives qg are zero. Sweeping k is therefore what pins the
    feedback structure (rg = -bin*kgN, qg = rg - bin*((g-1)*p0s)**j, and the sg
    recursion order that consumes them). It still does not pin the value of g, for
    the reason given in the sibling test: dropping the 1/alpha factor from g leaves
    this one passing too.

    Measured relative error at fc=1000: 1.1e-15 at k=1, 6.2e-16 at k=2, 6.9e-16
    at k=3.
    """
    x = np.full(20000, 0.05)
    y = oracle.process(x, FS, 1000.0, k, N)
    assert y[-1000:].mean() == pytest.approx(-0.05 / (1.0 + k), rel=1e-3)


def test_larger_k_gives_larger_peak_gain():
    """More resonance gives a bigger peak, so this needs a drive below saturation.

    The drive level here is load-bearing and 0.05, as originally briefed, is wrong
    for this test. The input stage is tanh(VT2i * x) with VT2i = 19.23, so a 0.05
    peak drive is 0.96 into the tanh: 22% compressed, i.e. deep in saturation, not
    "in the linear region". At that level the output peak is pinned to the
    saturation ceiling and stops tracking resonance at all. Measured peaks for a
    1000 Hz sine at fc=1000, by k:

        drive   k=0.5     k=1.0     k=2.0     k=3.0    k3/k1
        0.05    0.02000   0.01850   0.01770   0.01683   0.91   <- ceiling, wrong way
        0.005   0.00261   0.00287   0.00387   0.00606   2.11
        5e-4    0.00026   0.00029   0.00040   0.00074   2.57   <- used here
        5e-5    0.00003   0.00003   0.00004   0.00007   2.57

    The 0.05 row even decreases with k. The 5e-4 row tracks the linear reference,
    whose peak rises from -3.72 dB at k=1 to +3.50 dB at k=3, a ratio of 2.29.
    5e-4 drives the input tanh at 0.0096, where it is linear to four decimals.
    """
    x = 5e-4 * np.sin(2 * np.pi * 1000.0 * np.arange(44100) / FS)
    low = oracle.process(x, FS, 1000.0, 1.0, N)
    high = oracle.process(x, FS, 1000.0, 3.0, N)
    assert np.max(np.abs(high)) > 2.0 * np.max(np.abs(low))


def test_oracle_converges_to_linear_reference_for_tiny_signals():
    """As drive vanishes the tanh stages linearize and the oracle matches Part I.

    Compared in band only, to 0.4*fs, because the two are not expected to agree
    once the analog stopband folds into the digital one. Shapes are compared
    after normalizing each to its own DC value, since the absolute gain of an
    impulse scaled by 1e-5 is arbitrary.

    The complex ratio is then pinned to exactly -1e-5, which is stronger than the
    magnitude shape check above: it makes the global inversion explicit, so the
    sign convention documented in test_dc_gain_is_one_at_zero_k is load-bearing
    here too and a silent sign flip in either the oracle or the linear reference
    fails this test rather than hiding behind a magnitude comparison.

    The ratio is -1e-5 to 1.27e-8 relative, and that residual is not noise: it is
    the cubic term of tanh, uniform in band rather than concentrated in the deep
    stopband, and it is what "drive vanishes" actually means. The imaginary part
    of the ratio stays below 7e-15. The 1e-6 bound leaves 80x headroom over the
    measured value while staying orders of magnitude below anything a mistranscribed
    coefficient or index would produce.
    """
    n_fft = 1 << 14
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)
    imp = np.zeros(n_fft)
    imp[0] = 1e-5

    h_oracle = np.abs(np.fft.rfft(oracle.process(imp, FS, 1000.0, 2.0, N)))
    h_ref = np.abs(mll.frequency_response(N, FS, 1000.0, 2.0, freqs))

    band = (freqs >= 20.0) & (freqs <= 0.4 * FS)
    got = 20.0 * np.log10(h_oracle[band] / h_oracle[0])
    want = 20.0 * np.log10(h_ref[band] / h_ref[0])
    assert np.max(np.abs(got - want)) < 0.5

    ratio = np.fft.rfft(oracle.process(imp, FS, 1000.0, 2.0, N))[band] / mll.frequency_response(
        N, FS, 1000.0, 2.0, freqs
    )[band]
    assert np.max(np.abs(ratio / -1e-5 - 1.0)) < 1e-6
