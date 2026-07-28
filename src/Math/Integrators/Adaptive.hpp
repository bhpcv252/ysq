#pragma once

#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>

#include <cstddef>

namespace ysq {

/// Dormand-Prince 5(4): a seven-stage explicit Runge-Kutta pair that produces
/// a fifth-order solution and a fourth-order one from the same stages, so the
/// difference between them estimates the error for free.
///
/// Coefficients from Dormand and Prince, "A family of embedded Runge-Kutta
/// formulae", J. Comp. Appl. Math. 6 (1980), 19-26. This is the pair behind
/// MATLAB's ode45 and SciPy's RK45.
///
/// **FSAL: First Same As Last.** The seventh stage is evaluated at
/// (t + h, y_next) with exactly the weights of the fifth-order solution, so it
/// *is* the first stage of the next step. Carrying it across costs six
/// evaluations per accepted step rather than seven, a saving of one in seven
/// on every step of a long run. It is also the reason a stepper has to be an
/// object: that carried derivative is state between calls.
///
/// The cache is keyed on the time it belongs to, so a rejected step simply
/// misses and re-evaluates rather than reusing a derivative from a step that
/// was thrown away.
///
/// The step() without an error argument advances by the fifth-order solution
/// and discards the estimate, which lets the fixed-step driver run this method
/// and measure its order.
template <OdeState S>
class DormandPrince54Stepper {
public:
    using State = S;
    using Scalar = StateScalarT<S>;

    /// The order of the solution that is propagated. The step controller uses
    /// this exponent, and the fixed-step order test measures it.
    static constexpr int order = 5;
    /// The order of the embedded estimate the error is measured against.
    static constexpr int embeddedOrder = 4;

    template <OdeSystem<State> System>
    void step(const System& system, Scalar time, const State& state, Scalar h, State& out,
              State& error) {
        computeStages(system, time, state, h);

        out = state +
              (m_k1 * b1() + m_k3 * b3() + m_k4 * b4() + m_k5 * b5() + m_k6 * b6()) * h;

        // The seventh stage lands on the new state, which is what makes it
        // reusable as the next step's first.
        m_k7 = system(time + h, out);
        ++m_evaluations;

        // The difference of the two solutions' weights, applied to the shared
        // stages. Formed from b and bStar at run time rather than from
        // precomputed differences, so there is one set of coefficients to get
        // right instead of two.
        error = (m_k1 * (b1() - bStar1()) + m_k3 * (b3() - bStar3()) +
                 m_k4 * (b4() - bStar4()) + m_k5 * (b5() - bStar5()) +
                 m_k6 * (b6() - bStar6()) + m_k7 * (-bStar7())) *
                h;

        m_cachedDerivative = m_k7;
        m_cachedTime = time + h;
        m_hasCache = true;
    }

    /// Advances by the fifth-order solution and throws the estimate away, so
    /// the fixed-step driver can run this method. The estimate still has to be
    /// written somewhere, and that somewhere is a member rather than a local:
    /// for a heap-allocated state a local would allocate on every step.
    template <OdeSystem<State> System>
    void step(const System& system, Scalar time, const State& state, Scalar h,
              State& out) {
        step(system, time, state, h, out, m_discarded);
    }

    /// Drops the carried derivative. Needed only if the caller restarts the
    /// integration somewhere unrelated to where it left off.
    void reset() { m_hasCache = false; }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    template <OdeSystem<State> System>
    void computeStages(const System& system, Scalar time, const State& state, Scalar h) {
        if (m_hasCache && m_cachedTime == time) {
            m_k1 = m_cachedDerivative;
        } else {
            m_k1 = system(time, state);
            ++m_evaluations;
        }

        m_scratch = state + m_k1 * (h * a21());
        m_k2 = system(time + h * c2(), m_scratch);

        m_scratch = state + (m_k1 * a31() + m_k2 * a32()) * h;
        m_k3 = system(time + h * c3(), m_scratch);

        m_scratch = state + (m_k1 * a41() + m_k2 * a42() + m_k3 * a43()) * h;
        m_k4 = system(time + h * c4(), m_scratch);

        m_scratch =
            state + (m_k1 * a51() + m_k2 * a52() + m_k3 * a53() + m_k4 * a54()) * h;
        m_k5 = system(time + h * c5(), m_scratch);

        m_scratch = state + (m_k1 * a61() + m_k2 * a62() + m_k3 * a63() + m_k4 * a64() +
                             m_k5 * a65()) *
                                h;
        m_k6 = system(time + h, m_scratch);

        m_evaluations += 5;
    }

    // Written as exact rational expressions rather than as decimal literals:
    // a mistyped digit in a Butcher tableau collapses the method's order, and
    // a ratio of two integers is checkable by eye against the paper.
    static constexpr Scalar c2() { return Scalar{1} / Scalar{5}; }
    static constexpr Scalar c3() { return Scalar{3} / Scalar{10}; }
    static constexpr Scalar c4() { return Scalar{4} / Scalar{5}; }
    static constexpr Scalar c5() { return Scalar{8} / Scalar{9}; }

    static constexpr Scalar a21() { return Scalar{1} / Scalar{5}; }

    static constexpr Scalar a31() { return Scalar{3} / Scalar{40}; }
    static constexpr Scalar a32() { return Scalar{9} / Scalar{40}; }

    static constexpr Scalar a41() { return Scalar{44} / Scalar{45}; }
    static constexpr Scalar a42() { return Scalar{-56} / Scalar{15}; }
    static constexpr Scalar a43() { return Scalar{32} / Scalar{9}; }

    static constexpr Scalar a51() { return Scalar{19372} / Scalar{6561}; }
    static constexpr Scalar a52() { return Scalar{-25360} / Scalar{2187}; }
    static constexpr Scalar a53() { return Scalar{64448} / Scalar{6561}; }
    static constexpr Scalar a54() { return Scalar{-212} / Scalar{729}; }

    static constexpr Scalar a61() { return Scalar{9017} / Scalar{3168}; }
    static constexpr Scalar a62() { return Scalar{-355} / Scalar{33}; }
    static constexpr Scalar a63() { return Scalar{46732} / Scalar{5247}; }
    static constexpr Scalar a64() { return Scalar{49} / Scalar{176}; }
    static constexpr Scalar a65() { return Scalar{-5103} / Scalar{18656}; }

    // The fifth-order weights. The second stage does not appear, which is a
    // property of the pair rather than an omission.
    static constexpr Scalar b1() { return Scalar{35} / Scalar{384}; }
    static constexpr Scalar b3() { return Scalar{500} / Scalar{1113}; }
    static constexpr Scalar b4() { return Scalar{125} / Scalar{192}; }
    static constexpr Scalar b5() { return Scalar{-2187} / Scalar{6784}; }
    static constexpr Scalar b6() { return Scalar{11} / Scalar{84}; }

    // The embedded fourth-order weights.
    static constexpr Scalar bStar1() { return Scalar{5179} / Scalar{57600}; }
    static constexpr Scalar bStar3() { return Scalar{7571} / Scalar{16695}; }
    static constexpr Scalar bStar4() { return Scalar{393} / Scalar{640}; }
    static constexpr Scalar bStar5() { return Scalar{-92097} / Scalar{339200}; }
    static constexpr Scalar bStar6() { return Scalar{187} / Scalar{2100}; }
    static constexpr Scalar bStar7() { return Scalar{1} / Scalar{40}; }

    State m_k1{};
    State m_k2{};
    State m_k3{};
    State m_k4{};
    State m_k5{};
    State m_k6{};
    State m_k7{};
    State m_scratch{};

    State m_discarded{};
    State m_cachedDerivative{};
    Scalar m_cachedTime{};
    bool m_hasCache = false;
    std::size_t m_evaluations = 0;
};

}  // namespace ysq
