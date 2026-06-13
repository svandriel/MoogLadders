// HyperionMoog v2 - a zero-delay-feedback Moog ladder filter combining:
// - Correct TPT (trapezoidal) discretization of dy/dt = wc*(S(x) - S(y)) per stage
// - Antiderivative Antialiasing (ADAA1) on each nonlinearity's forward path
// - A pluggable, derivative-consistent saturator policy (algebraic or tanh):
//   the instantaneous value S, slope S', and antiderivative F always come from
//   the same curve, which ADAA correctness depends on
// - Newton-Raphson per-stage implicit solves with guaranteed-positive Jacobian
// - Multi-mode output (LP, HP, BP, Notch)
// - By Dimitri Diakopoulos and Claude, 2025-2026 (Public Domain/Unlicense)
//
// See HyperionReference.md for the full derivation and the v1 errata.

#pragma once

#ifndef HYPERION_LADDER_H
#define HYPERION_LADDER_H

#include "LadderFilterBase.h"
#include <cmath>
#include <algorithm>

// Newton iterations per stage solve (after the relinearized warm start).
// Measured: 1 iteration is transparent (THD and alias floor identical to 4
// decimal places vs 2) and ~38% faster; raise for offline rendering if the
// stage solve must be exact to machine precision.
#ifndef HYPERION_NR_ITERS
#define HYPERION_NR_ITERS 1
#endif

// Apply ADAA only at the input saturator (0, default) or additionally at
// each ladder stage (1). Measured at 44.1k: input-only is better on BOTH
// axes - aliasing (-46 dB vs -38 dB at the 10 kHz stress test) and resonance
// tuning (0.96 vs 0.84 self-osc pitch ratio at fc=1 kHz) - because each
// averaged site inside the recursive loop adds ~half a sample of delay and
// mixes time references between S_avg(x) and the instantaneous S(y).
#ifndef HYPERION_STAGE_ADAA
#define HYPERION_STAGE_ADAA 0
#endif

// Precomputed constants shared by the saturator policies. All saturators are
// normalized to unity slope at the origin and saturate to +/- 2*Vt.
struct HyperionSatParams
{
    double twoVt;
    double invTwoVt;
    double fourVtSq;
    double adaaEps; // |dx| threshold below which ADAA falls back to midpoint S

    void Set(double Vt)
    {
        twoVt = 2.0 * Vt;
        invTwoVt = 1.0 / twoVt;
        fourVtSq = 4.0 * Vt * Vt;
        // Relative threshold: at 1e-4 * 2Vt the quotient's cancellation error
        // (~eps_double*|F|/dx) and the midpoint truncation error (~dx^2/24*S'')
        // are both far below -180 dB, so the branch switch is seamless.
        adaaEps = 1e-4 * twoVt;
    }
};

// Algebraic sigmoid saturator (default): S(x) = x / sqrt(1 + (x/2Vt)^2).
// Bounded to +/- 2Vt, S'(0) = 1, no transcendentals (sqrt + div only),
// exact closed-form antiderivative. Slightly softer knee than tanh.
struct HyperionAlgSat
{
    static inline double S(double x, const HyperionSatParams& p)
    {
        double u = x * p.invTwoVt;
        return x / std::sqrt(1.0 + u * u);
    }

    // Ratio gain S(x)/x, the linearization that is exact at the operating
    // point (used for the relinearized ZDF prediction). No singularity at 0.
    static inline double SRatio(double x, const HyperionSatParams& p)
    {
        double u = x * p.invTwoVt;
        return 1.0 / std::sqrt(1.0 + u * u);
    }

    // Returns S(x), writes S'(x) = (1 + u^2)^(-3/2) to sp. One sqrt, one div.
    static inline double SAndPrime(double x, const HyperionSatParams& p, double& sp)
    {
        double u = x * p.invTwoVt;
        double invR = 1.0 / std::sqrt(1.0 + u * u);
        sp = invR * invR * invR;
        return x * invR;
    }

    // F(x) = int S = 4Vt^2 * (sqrt(1 + u^2) - 1), F(0) = 0, F' = S exactly.
    static inline double F(double x, const HyperionSatParams& p)
    {
        double u = x * p.invTwoVt;
        return p.fourVtSq * (std::sqrt(1.0 + u * u) - 1.0);
    }

    // F(x) and S(x)/x share one sqrt: R = sqrt(1+u^2), F = 4Vt^2(R-1), ratio = 1/R
    static inline double FAndRatio(double x, const HyperionSatParams& p, double& ratio)
    {
        double u = x * p.invTwoVt;
        double R = std::sqrt(1.0 + u * u);
        ratio = 1.0 / R;
        return p.fourVtSq * (R - 1.0);
    }
};

// Exact transistor-pair saturator: S(x) = 2Vt * tanh(x/2Vt).
// The classic Moog curve; costs one tanh per evaluation and exp/log1p for F.
struct HyperionTanhSat
{
    static inline double S(double x, const HyperionSatParams& p)
    {
        return p.twoVt * std::tanh(x * p.invTwoVt);
    }

    // Ratio gain S(x)/x = tanh(u)/u; -> 1 as u -> 0
    static inline double SRatio(double x, const HyperionSatParams& p)
    {
        double u = x * p.invTwoVt;
        double au = std::fabs(u);
        if (au < 1e-8) return 1.0;
        return std::tanh(u) / u;
    }

    static inline double SAndPrime(double x, const HyperionSatParams& p, double& sp)
    {
        double t = std::tanh(x * p.invTwoVt);
        sp = 1.0 - t * t; // sech^2, the exact derivative of S w.r.t. x
        return p.twoVt * t;
    }

    // F(x) = 4Vt^2 * logcosh(x/2Vt) via the numerically stable exact identity
    // logcosh(a) = |a| + log1p(exp(-2|a|)) - ln(2).
    static inline double F(double x, const HyperionSatParams& p)
    {
        double a = std::fabs(x * p.invTwoVt);
        double lc = (a > 20.0) ? (a - MOOG_LN2) : (a + log1p(std::exp(-2.0 * a)) - MOOG_LN2);
        return p.fourVtSq * lc;
    }

    static inline double FAndRatio(double x, const HyperionSatParams& p, double& ratio)
    {
        ratio = SRatio(x, p);
        return F(x, p);
    }
};

template <typename Sat>
class HyperionMoogT : public LadderFilterBase
{
public:

    enum FilterMode { LP2, LP4, BP2, BP4, HP2, HP4, NOTCH };

    // Feedback resolution quality / CPU trade-off:
    //  STATIC        - small-signal (linear) feedback prediction, cheapest
    //  RELINEARIZED  - per-sample ratio-gain relinearization from the previous
    //                  sample's operating point (default). Tracks resonance and
    //                  loop gain correctly under heavy saturation.
    //  OUTER2        - RELINEARIZED, then a second full pass with gains
    //                  recomputed at this sample's solution. Most accurate.
    enum SolverQuality { QUALITY_STATIC, QUALITY_RELINEARIZED, QUALITY_OUTER2 };

    HyperionMoogT(float sampleRate) : LadderFilterBase(sampleRate)
    {
        satp.Set(0.312); // Thermal voltage scaled for numerical convenience

        drive = 1.0;
        K = 0.0;
        g = 0.0;
        G = 0.0;
        gamma = 0.0;
        alpha0 = 1.0;
        adaaEnabled = true;

        std::fill(std::begin(beta), std::end(beta), 0.0);
        Reset();

        SetFilterMode(LP4);
        SetCutoff(1000.0f);
        SetResonance(0.1f);
    }

    virtual ~HyperionMoogT() {}

    void Reset()
    {
        std::fill(std::begin(z), std::end(z), 0.0);
        std::fill(std::begin(xPrev), std::end(xPrev), 0.0);
        std::fill(std::begin(FxPrev), std::end(FxPrev), 0.0);
        std::fill(std::begin(ratioPrev), std::end(ratioPrev), 1.0); // S(v)/v -> 1 at v = 0
        uPrev = 0.0;
        FuPrev = 0.0;
        y3Prev = 0.0;
    }

    virtual void Process(float* samples, uint32_t n) override
    {
        for (uint32_t s = 0; s < n; ++s)
        {
            samples[s] = Tick(samples[s]);
        }
    }

    virtual void SetCutoff(float c) override
    {
        // tan() blows up approaching Nyquist; 0.45*fs keeps g finite and useful
        cutoff = std::max(10.0f, std::min(c, 0.45f * sampleRate));
        UpdateCoefficients();
    }

    virtual void SetResonance(float r) override
    {
        resonance = r;
        K = 4.0 * r;
        UpdateCoefficients();
    }

    void SetFilterMode(FilterMode mode)
    {
        // Multi-mode output coefficients: output = c[0]*u + c[1]*y0 + c[2]*y1 + c[3]*y2 + c[4]*y3
        switch (mode) {
            case LP4:   modeCoeffs[0] = 0; modeCoeffs[1] = 0;  modeCoeffs[2] = 0;  modeCoeffs[3] = 0;  modeCoeffs[4] = 1;  break;
            case LP2:   modeCoeffs[0] = 0; modeCoeffs[1] = 0;  modeCoeffs[2] = 1;  modeCoeffs[3] = 0;  modeCoeffs[4] = 0;  break;
            case HP4:   modeCoeffs[0] = 1; modeCoeffs[1] = -4; modeCoeffs[2] = 6;  modeCoeffs[3] = -4; modeCoeffs[4] = 1;  break;
            case HP2:   modeCoeffs[0] = 1; modeCoeffs[1] = -2; modeCoeffs[2] = 1;  modeCoeffs[3] = 0;  modeCoeffs[4] = 0;  break;
            case BP4:   modeCoeffs[0] = 0; modeCoeffs[1] = 0;  modeCoeffs[2] = 4;  modeCoeffs[3] = -8; modeCoeffs[4] = 4;  break;
            case BP2:   modeCoeffs[0] = 0; modeCoeffs[1] = 2;  modeCoeffs[2] = -2; modeCoeffs[3] = 0;  modeCoeffs[4] = 0;  break;
            case NOTCH: modeCoeffs[0] = 1; modeCoeffs[1] = -4; modeCoeffs[2] = 6;  modeCoeffs[3] = -4; modeCoeffs[4] = 0;  break;
        }
    }

    void SetDrive(float d) { drive = d; }

    void SetQuality(SolverQuality q) { quality = q; }

    void SetThermalVoltage(float Vt)
    {
        satp.Set(Vt);
        RefreshAdaaState();
    }

    void SetAdaaEnabled(bool enable)
    {
        adaaEnabled = enable;
        if (enable) RefreshAdaaState();
    }

    double GetStoredEnergy() const
    {
        // Sum of squared state variables (proportional to stored energy)
        return z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3];
    }

private:

    // Tuning compensation, measured at 44.1 kHz and expressed in x = fc/fs
    // (validated to hold at 48 kHz). Two fitted cubics, both anchored at 1:
    //
    //  Pg corrects the small linear-cutoff droop caused by the input ADAA
    //  averager's cos(w*T/2) magnitude (fit to impulse-measured -12 dB points
    //  over fc = 200..16000 at r = 0; <= 1.6% residual).
    //
    //  Pk corrects the resonance/self-oscillation pitch: the input averager
    //  also sits inside the feedback loop, and its half-sample lag moves the
    //  loop's -180 degree crossing down. Fit so that self-oscillation at K=4
    //  lands on the commanded cutoff (fc = 200..9000; <= 0.1% residual).
    //  Above the fitted range the cubics are evaluated at the range edge.
    //
    // The effective pole frequency blends linearly in K/4 between the two.
    void UpdateCoefficients()
    {
        double x = cutoff / sampleRate;

        double xg = std::min(x, 0.3628); // Pg fitted up to 16 kHz @ 44.1k
        double Pg = 1.0 + xg * (-0.004760 + xg * (4.395003 + xg * -8.333518));

        double xk = std::min(x, 0.2041); // Pk fitted up to 9 kHz @ 44.1k
        double Pk = 1.0 + xk * (1.927090 + xk * (1.493155 + xk * -15.779799));

        double blend = 0.25 * K;
        double fcEff = cutoff * (Pg + blend * (Pk - Pg));
        fcEff = std::min(fcEff, 0.49 * (double)sampleRate); // keep prewarp finite

        double wd = 2.0 * MOOG_PI * fcEff;
        double T = 1.0 / sampleRate;
        double wa = (2.0 / T) * tan(wd * T / 2.0);

        // TPT integrator coefficient (raw); G is the linearized one-pole gain
        g = wa * T / 2.0;
        G = g / (1.0 + g);
        gamma = G * G * G * G;

        // Feedback weights: with the linearized stage y = G*x + (1-G)*z the
        // ladder output is y3 = G^4*u + sigma, sigma = sum(beta[i]*z[i])
        double gInv = 1.0 / (1.0 + g);
        beta[0] = G * G * G * gInv;
        beta[1] = G * G * gInv;
        beta[2] = G * gInv;
        beta[3] = gInv;

        alpha0 = 1.0 / (1.0 + K * gamma);
    }

    // Per-sample feedback prediction coefficients (filled per quality tier)
    struct LoopGains
    {
        double beta[4]; // sigma weights on z[i]
        double alpha0;  // feedback resolution
        double Gf[4];   // per-stage linearized forward gains (Newton warm start)
        double d[4];    // per-stage linearized state gains
        bool haveStageGains;
    };

    // Build the relinearized loop gains from ratio gains S(v)/v evaluated at
    // an operating point. Stage i: y = z + g*(ri*x - ro*y) linearizes to
    //   y = Gf*x + d*z,  d = 1/(1 + g*ro),  Gf = g*ri*d
    // where ri is the input-side ratio gain (sgain[i]) and ro the output-side
    // one (sgain[i+1]: stage i's output is stage i+1's input, sgain[4] closes
    // y3). The cascade gives the sigma weights, and the loop closes through
    // the input saturator's ratio gain su and drive:
    //   alpha0 = 1/(1 + K*su*drive*Gf0*Gf1*Gf2*Gf3)
    inline void ComputeGainsFromRatios(const double sgain[5], double su, LoopGains& lg) const
    {
        for (int i = 0; i < 4; i++) {
            lg.d[i] = 1.0 / (1.0 + g * sgain[i + 1]);
            lg.Gf[i] = g * sgain[i] * lg.d[i];
        }
        lg.beta[3] = lg.d[3];
        lg.beta[2] = lg.Gf[3] * lg.d[2];
        lg.beta[1] = lg.Gf[3] * lg.Gf[2] * lg.d[1];
        lg.beta[0] = lg.Gf[3] * lg.Gf[2] * lg.Gf[1] * lg.d[0];
        double gammaHat = lg.Gf[0] * lg.Gf[1] * lg.Gf[2] * lg.Gf[3];
        lg.alpha0 = 1.0 / (1.0 + K * su * drive * gammaHat);
        lg.haveStageGains = true;
    }

    // One feedback resolution + stage cascade pass. Does NOT commit any state;
    // outputs the saturated input u, stage outputs y[4], and the per-site
    // antiderivatives and ratio gains needed to commit afterwards.
    inline void RunPass(double input, const LoopGains& lg,
                        double& u, double& uDrive, double& Fu,
                        double y[4], double Fx[4], double ratios[5]) const
    {
        double sigma = lg.beta[0] * z[0] + lg.beta[1] * z[1] + lg.beta[2] * z[2] + lg.beta[3] * z[3];

        double inputScaled = input * (1.0 + K); // passband gain compensation
        double uRaw = (inputScaled - K * sigma) * lg.alpha0;
        uDrive = uRaw * drive;

        Fu = 0.0;
        if (adaaEnabled) {
            Fu = Sat::FAndRatio(uDrive, satp, ratios[4]);
            double du = uDrive - uPrev;
            u = (std::fabs(du) < satp.adaaEps) ? Sat::S(0.5 * (uDrive + uPrev), satp)
                                               : (Fu - FuPrev) / du;
        } else {
            ratios[4] = Sat::SRatio(uDrive, satp);
            u = ratios[4] * uDrive; // S(x) = ratio * x exactly
        }

        double x = u;
        for (int i = 0; i < 4; i++)
        {
            double SxAvg;
#if HYPERION_STAGE_ADAA
            if (adaaEnabled) {
                Fx[i] = Sat::FAndRatio(x, satp, ratios[i]);
                double dx = x - xPrev[i];
                SxAvg = (std::fabs(dx) < satp.adaaEps) ? Sat::S(0.5 * (x + xPrev[i]), satp)
                                                       : (Fx[i] - FxPrev[i]) / dx;
            } else
#endif
            {
                Fx[i] = 0.0;
                ratios[i] = Sat::SRatio(x, satp);
                SxAvg = ratios[i] * x;
            }

            // Warm start from the (re)linearized stage model
            double y0 = lg.Gf[i] * x + lg.d[i] * z[i];
            y[i] = SolveStage(SxAvg, z[i], y0);
            x = y[i];
        }
    }

    inline float Tick(double input)
    {
        LoopGains lg;
        if (quality == QUALITY_STATIC) {
            for (int i = 0; i < 4; i++) {
                lg.beta[i] = beta[i];
                lg.Gf[i] = G;        // linear TPT warm start
                lg.d[i] = 1.0 - G;   // = 1/(1+g)
            }
            lg.alpha0 = alpha0;
            lg.haveStageGains = true;
        } else {
            // Operating point: previous sample's committed solution, reusing
            // the ratio gains cached by the previous commit (y3 is the only
            // signal that is not also some stage's input)
            double sgain[5] = { ratioPrev[0], ratioPrev[1], ratioPrev[2], ratioPrev[3],
                                Sat::SRatio(y3Prev, satp) };
            ComputeGainsFromRatios(sgain, ratioPrev[4], lg);
        }

        double u, uDrive, Fu;
        double y[4], Fx[4], ratios[5];
        RunPass(input, lg, u, uDrive, Fu, y, Fx, ratios);

        if (quality == QUALITY_OUTER2) {
            // Second pass with gains taken at THIS sample's solution
            double xOp[4] = { u, y[0], y[1], y[2] };
            double sgain[5];
            for (int i = 0; i < 4; i++) sgain[i] = Sat::SRatio(xOp[i], satp);
            sgain[4] = Sat::SRatio(y[3], satp);
            ComputeGainsFromRatios(sgain, Sat::SRatio(uDrive, satp), lg);
            RunPass(input, lg, u, uDrive, Fu, y, Fx, ratios);
        }

        // Commit state
        double x = u;
        for (int i = 0; i < 4; i++)
        {
            z[i] = 2.0 * y[i] - z[i]; // trapezoidal state update
            xPrev[i] = x;
            FxPrev[i] = Fx[i];
            ratioPrev[i] = ratios[i];
            x = y[i];
        }
        uPrev = uDrive;
        FuPrev = Fu;
        ratioPrev[4] = ratios[4];
        y3Prev = y[3];

        return static_cast<float>(
            modeCoeffs[0] * u + modeCoeffs[1] * y[0] + modeCoeffs[2] * y[1] +
            modeCoeffs[3] * y[2] + modeCoeffs[4] * y[3]);
    }

    // Solve the implicit stage equation, the trapezoidal (TPT) discretization
    // of dy/dt = wc*(S(x) - S(y)):
    //   y = z + g * (Savg(x) - S(y))
    // Raw g (not G = g/(1+g)): in the linear limit this reduces to the
    // standard TPT one-pole y = G*x + (1-G)*z. Savg(x) is the ADAA1 mean of S
    // over [xPrev, x]; the exact identity (F(x)-F(p))/(x-p) = S(m) +
    // (dx^2/24)*S''(m) + O(dx^4), m = (x+p)/2, makes midpoint S the correct
    // small-interval fallback (computed by the caller). With S' in (0,1] the
    // Jacobian 1 + g*S'(y) >= 1, so the residual is strictly monotonic in y:
    // the root is unique and Newton is well-conditioned everywhere.
    inline double SolveStage(double SxAvg, double zi, double yGuess) const
    {
        double y = yGuess;
        for (int iter = 0; iter < nrIters; ++iter)
        {
            double sp;
            double Sy = Sat::SAndPrime(y, satp, sp);
            double residual = y - zi - g * (SxAvg - Sy);
            double delta = residual / (1.0 + g * sp);
            y -= delta;
            if (std::fabs(delta) < newtonTol) break;
        }
        return y;
    }

    // Recompute stored antiderivatives and ratio gains so every ADAA quotient
    // differences two F values from the same curve (required after Vt changes
    // or re-enabling ADAA)
    void RefreshAdaaState()
    {
        for (int i = 0; i < 4; i++) FxPrev[i] = Sat::FAndRatio(xPrev[i], satp, ratioPrev[i]);
        FuPrev = Sat::FAndRatio(uPrev, satp, ratioPrev[4]);
    }

    // TPT integrator states (cap voltages)
    double z[4];

    // TPT coefficients
    double g;       // Raw integrator coefficient wa*T/2
    double G;       // g/(1+g), linearized one-pole gain
    double gamma;   // G^4, linearized open-loop ladder gain
    double alpha0;  // Feedback resolution 1/(1 + K*gamma)
    double K;       // Resonance [0, 4]
    double beta[4]; // Feedback weights for the ZDF sigma sum

    // ADAA state: previous input and antiderivative per nonlinearity site
    double xPrev[4];
    double FxPrev[4];
    double uPrev;
    double FuPrev;

    // Previous sample's final stage output (operating point for relinearization;
    // the other stage outputs equal xPrev[1..3])
    double y3Prev;

    // Cached ratio gains S(v)/v at the committed operating point:
    // [0..3] at xPrev[i], [4] at uPrev. Shares the sqrt with F for AlgSat.
    double ratioPrev[5];

    HyperionSatParams satp;
    bool adaaEnabled;
    SolverQuality quality = QUALITY_RELINEARIZED;

    double drive;
    double modeCoeffs[5];

    int nrIters = HYPERION_NR_ITERS;
    static constexpr double newtonTol = 1e-10;
};

// Default: algebraic saturator (fastest, transcendental-free).
using HyperionMoog = HyperionMoogT<HyperionAlgSat>;
// Classic curve: exact tanh transistor-pair saturation.
using HyperionMoogTanh = HyperionMoogT<HyperionTanhSat>;

#endif // HYPERION_LADDER_H
