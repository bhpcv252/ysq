#pragma once

#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>

#include <cstddef>

namespace ysq {

/// The classical fourth-order Runge-Kutta method.
///
/// Four evaluations per step, and the global error falls as the fourth power
/// of the step: halving the step buys sixteen times the accuracy, where Euler
/// buys two. That is what makes it the default for a general system.
///
/// **It is not symplectic.** On a periodic system its energy error is not
/// bounded; it drifts in one direction, slowly, forever. Over a few orbits
/// that is invisible and RK4 is far more accurate than any second-order
/// symplectic method. Over a million it is the only thing that matters, and
/// Verlet wins despite being two orders worse per step. Which of those regimes
/// a run is in is the question worth asking before picking either.
///
///     k1 = f(t,       y)
///     k2 = f(t + h/2, y + (h/2) k1)
///     k3 = f(t + h/2, y + (h/2) k2)
///     k4 = f(t + h,   y + h k3)
///     y' = y + (h/6)(k1 + 2 k2 + 2 k3 + k4)
///
/// The weights are the ones that make Simpson's rule out of the four stages,
/// which is why the same 1, 2, 2, 1 pattern appears in both.
template <OdeState S>
class Rk4Stepper {
public:
    using State = S;
    using Scalar = StateScalarT<S>;

    static constexpr int order = 4;

    template <OdeSystem<State> System>
    void step(const System& system, Scalar time, const State& state, Scalar h,
              State& out) {
        const Scalar half = h / Scalar{2};

        m_k1 = system(time, state);

        m_scratch = state + m_k1 * half;
        m_k2 = system(time + half, m_scratch);

        m_scratch = state + m_k2 * half;
        m_k3 = system(time + half, m_scratch);

        m_scratch = state + m_k3 * h;
        m_k4 = system(time + h, m_scratch);

        m_evaluations += 4;

        out =
            state + (m_k1 + m_k2 * Scalar{2} + m_k3 * Scalar{2} + m_k4) * (h / Scalar{6});
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    State m_k1{};
    State m_k2{};
    State m_k3{};
    State m_k4{};
    State m_scratch{};
    std::size_t m_evaluations = 0;
};

}  // namespace ysq
