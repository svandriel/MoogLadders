/*
HyperionMoog - A novel Moog ladder filter combining:
- Zero-Delay Feedback via Topology-Preserving Transform (TPT)
- Antiderivative Antialiasing (ADAA) for reduced aliasing without oversampling
- Per-stage nonlinearity with adaptive thermal voltage modeling
- Multi-mode output (LP, HP, BP, Notch)

References:
- Pirkle (2012): TPT framework for virtual analog
- Bilbao & Välimäki (2017): Antiderivative antialiasing
- Huovilainen (2004): Nonlinear ladder modeling

This implementation is self-contained with all ADAA functions inline.
*/

#pragma once

#ifndef HYPERION_LADDER_H
#define HYPERION_LADDER_H

#include "LadderFilterBase.h"
#include "Util.h"
#include <cmath>
#include <cstring>

class HyperionMoog : public LadderFilterBase
{
public:
    enum FilterMode { LP2, LP4, BP2, BP4, HP2, HP4, NOTCH };

    HyperionMoog(float sampleRate) : LadderFilterBase(sampleRate)
    {
        // Initialize state
        memset(z, 0, sizeof(z));
        memset(v_prev, 0, sizeof(v_prev));
        memset(F_prev, 0, sizeof(F_prev));
        memset(Vt_prev, 0, sizeof(Vt_prev));
        u_prev = 0.0;
        Fu_prev = 0.0;
        Vt_u_prev = 0.0;

        // Default parameters
        Vt = 0.312;            // Thermal voltage scaled for numerical convenience
        VtAlpha = 0.05;        // Adaptive coefficient
        adaptiveVtEnabled = true;
        drive = 1.0;
        K = 0.0;
        g = 0.0;
        G = 0.0;
        gamma = 0.0;
        alpha0 = 1.0;
        memset(beta, 0, sizeof(beta));

        // Default to LP4 mode
        SetFilterMode(LP4);
        SetCutoff(1000.0f);
        SetResonance(0.1f);
    }

    virtual ~HyperionMoog() {}

    virtual void Process(float* samples, uint32_t n) override
    {
        for (uint32_t s = 0; s < n; ++s)
        {
            // 1. Compute zero-delay feedback sum (TPT ladder weights)
            double sigma =
                beta[0] * z[0] +
                beta[1] * z[1] +
                beta[2] * z[2] +
                beta[3] * z[3];

            // 2. Input stage: feedback subtraction and input saturation with ADAA
            double inputScaled = samples[s] * (1.0 + K);
            double u_raw = (inputScaled - K * sigma) * alpha0;
            double u_drive = u_raw * drive;

            // Apply ADAA to input saturation
            double Fu_out;
            double Vt_u = GetEffectiveVt(u_drive);
            double u;
            if (Vt_u_prev > 0.0 && fabs(Vt_u - Vt_u_prev) > (VtRelEps * Vt_u_prev)) {
                // Vt changed significantly - use instantaneous normalized tanh
                u = 2.0 * Vt_u * tanh(u_drive / (2.0 * Vt_u));
                Fu_out = TanhAntiderivative(u_drive, Vt_u);
            } else {
                u = TanhADAA(u_drive, u_prev, Fu_prev, Vt_u, Fu_out);
            }
            u_prev = u_drive;
            Fu_prev = Fu_out;
            Vt_u_prev = Vt_u;

            // 3. Four-stage cascade with per-stage ADAA nonlinearity
            double y[4];
            double x = u;

            for (int i = 0; i < 4; i++)
            {
                y[i] = SolveStageADAA(i, x);
                x = y[i];  // Output feeds next stage
            }

            // 4. Multi-mode output mixing
            samples[s] = static_cast<float>(
                modeCoeffs[0] * u +
                modeCoeffs[1] * y[0] +
                modeCoeffs[2] * y[1] +
                modeCoeffs[3] * y[2] +
                modeCoeffs[4] * y[3]
            );
        }
    }

    virtual void SetCutoff(float c) override
    {
        cutoff = c;

        // Prewarp for bilinear transform
        double wd = 2.0 * MOOG_PI * cutoff;
        double T = 1.0 / sampleRate;
        double wa = (2.0 / T) * tan(wd * T / 2.0);

        // TPT integrator coefficient
        g = wa * T / 2.0;
        G = g / (1.0 + g);
        gamma = G * G * G * G;

        // TPT ladder feedback weights (aligned with OberheimVariationModel)
        double gInv = 1.0 / (1.0 + g);
        beta[0] = G * G * G * gInv;
        beta[1] = G * G * gInv;
        beta[2] = G * gInv;
        beta[3] = gInv;

        // Update feedback resolution
        alpha0 = 1.0 / (1.0 + K * gamma);
    }

    virtual void SetResonance(float r) override
    {
        resonance = r;
        K = 4.0 * r;  // Map [0,1] to [0,4]
        alpha0 = 1.0 / (1.0 + K * gamma);
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

    void SetAdaptiveVt(bool enable, float alpha = 0.05f)
    {
        adaptiveVtEnabled = enable;
        VtAlpha = alpha;
        Vt_u_prev = 0.0;
        memset(Vt_prev, 0, sizeof(Vt_prev));
    }

    // Energy monitoring (for validation/debugging)
    double GetStoredEnergy() const
    {
        // Sum of squared state variables (proportional to stored energy)
        return z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3];
    }

private:
    // ==================== INLINE ADAA FUNCTIONS ====================

    // Numerically stable log(cosh(x))
    // For |x| > 20, log(cosh(x)) ≈ |x| - ln(2)
    inline double LogCosh(double x) const
    {
        double ax = fabs(x);
        if (ax > 20.0) return ax - MOOG_LN2;  // Avoid overflow
        return ax + log1p(exp(-2.0 * ax)) - MOOG_LN2;
    }

    // Normalized saturation: S(x) = 2*Vt * tanh(x / (2*Vt))
    // This has unity gain at the origin: S'(0) = 1
    // Antiderivative: F(x) = 4*Vt^2 * ln(cosh(x / (2*Vt)))
    inline double TanhAntiderivative(double x, double Vt_eff) const
    {
        return 4.0 * Vt_eff * Vt_eff * LogCosh(x / (2.0 * Vt_eff));
    }

    // ADAA1: First-order antialiased normalized tanh
    // Returns average value of S(x) = 2*Vt*tanh(x/(2*Vt)) over interval [x_prev, x_curr]
    inline double TanhADAA(double x_curr, double x_prev, double F_prev_val, double Vt_eff, double& F_out) const
    {
        F_out = TanhAntiderivative(x_curr, Vt_eff);
        double denom = x_curr - x_prev;
        if (fabs(denom) < 1e-12) {
            // Degenerate case: return instantaneous normalized tanh
            return 2.0 * Vt_eff * tanh(x_curr / (2.0 * Vt_eff));
        }
        return (F_out - F_prev_val) / denom;
    }

    // Derivative of normalized tanh: d/dx [2*Vt * tanh(x/(2*Vt))] = sech^2(x/(2*Vt))
    // Note: unity at origin (sech^2(0) = 1)
    inline double TanhDerivative(double x, double Vt_eff) const
    {
        double scaled = x / (2.0 * Vt_eff);
        // Avoid overflow for large |x|
        if (fabs(scaled) > 20.0) return 0.0;
        double c = cosh(scaled);
        return 1.0 / (c * c);
    }

    // Fast tanh approximation for initial Newton guess (from Util.h)
    inline double FastTanh(double x) const
    {
        double x2 = x * x;
        return x * (27.0 + x2) / (27.0 + 9.0 * x2);
    }

    // Adaptive thermal voltage
    inline double GetEffectiveVt(double x) const
    {
        return adaptiveVtEnabled ? Vt * (1.0 + VtAlpha * fabs(x)) : Vt;
    }

    // ==================== NEWTON-RAPHSON STAGE SOLVER ====================

    // Solve the implicit stage equation with ADAA:
    //   y = G * S_avg(x - y) + (1 - G) * z[i]
    // where S_avg is the ADAA-averaged tanh
    double SolveStageADAA(int i, double x)
    {
        double Vt_eff = GetEffectiveVt(x);
        bool vt_changed = (Vt_prev[i] > 0.0) && (fabs(Vt_eff - Vt_prev[i]) > (VtRelEps * Vt_prev[i]));

        // Initial guess using fast tanh approximation (normalized: 2*Vt*tanh(v/(2*Vt)))
        double y = G * 2.0 * Vt_eff * FastTanh((x - z[i]) / (2.0 * Vt_eff)) + (1.0 - G) * z[i];

        // Newton-Raphson iteration
        for (int iter = 0; iter < 4; iter++)
        {
            double v = x - y;  // Voltage across nonlinearity

            // ADAA: compute average tanh over [v_prev[i], v]
            double F_curr = TanhAntiderivative(v, Vt_eff);
            double denom = v - v_prev[i];
            double S_avg;
            if (!vt_changed && fabs(denom) > 1e-12) {
                S_avg = (F_curr - F_prev[i]) / denom;
            } else {
                // Use instantaneous normalized tanh
                S_avg = 2.0 * Vt_eff * tanh(v / (2.0 * Vt_eff));
            }

            // Residual: y - G*S_avg - (1-G)*z[i] = 0
            double residual = y - G * S_avg - (1.0 - G) * z[i];

            // Jacobian approximation using instantaneous derivative
            // d(residual)/dy = 1 + G * dS/dv * dv/dy = 1 + G * dS (since dv/dy = -1)
            double dS = TanhDerivative(v, Vt_eff);
            double jacobian = 1.0 + G * dS;

            double delta = residual / jacobian;
            y -= delta;

            if (fabs(delta) < 1e-8) break;
        }

        // Update TPT state (trapezoidal integrator)
        z[i] = 2.0 * y - z[i];

        // Update ADAA state for next sample
        double v_final = x - y;
        v_prev[i] = v_final;
        F_prev[i] = TanhAntiderivative(v_final, Vt_eff);
        Vt_prev[i] = Vt_eff;

        return y;
    }

    // ==================== FILTER STATE ====================

    // TPT integrator states (capacitor voltages)
    double z[4];

    // TPT coefficients
    double G;              // g/(1+g) integrator gain
    double g;              // Raw integrator coefficient
    double gamma;          // G^4 for feedback
    double alpha0;         // Feedback resolution: 1/(1 + K*gamma)
    double K;              // Resonance [0, 4]
    double beta[4];        // Feedback weights for TPT ladder sum

    // ADAA state: previous nonlinearity input and antiderivative per stage
    double v_prev[4];
    double F_prev[4];
    double Vt_prev[4];

    // Input saturation ADAA state
    double u_prev;
    double Fu_prev;
    double Vt_u_prev;

    // Thermal voltage modeling
    double Vt;             // Base thermal voltage (scaled for numerical convenience)
    double VtAlpha;        // Adaptive coefficient
    bool adaptiveVtEnabled;

    // Drive/saturation
    double drive;

    // Multi-mode output coefficients: output = c[0]*u + c[1]*y0 + c[2]*y1 + c[3]*y2 + c[4]*y3
    double modeCoeffs[5];

    static constexpr double VtRelEps = 1e-6;
};

#endif // HYPERION_LADDER_H
