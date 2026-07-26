#pragma once

#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>

#include <cstddef>

namespace ysq {

/// The first-order methods, and the two-stage Runge-Kutta pair.
///
/// Explicit Euler is here to be measured against, not to be used. Its global
/// error falls only in proportion to the step, so buying two digits costs a
/// hundred times the work, and it is unstable on anything oscillatory: the
/// energy of a harmonic oscillator grows without bound however small the step
/// is. Semi-implicit Euler costs exactly the same and does neither of those
/// things, which is the cheapest available demonstration that the structure of
/// a method matters more than its order.

/// y' = f(t, y), advanced by one tangent line. Order 1.
template <OdeState S>
class ExplicitEulerStepper {
public:
    using State = S;
    using Scalar = StateScalarT<S>;

    static constexpr int order = 1;

    template <OdeSystem<State> System>
    void step(const System& system, Scalar time, const State& state, Scalar h,
              State& out) {
        m_derivative = system(time, state);
        ++m_evaluations;
        out = state + m_derivative * h;
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    State m_derivative{};
    std::size_t m_evaluations = 0;
};

/// The midpoint rule: one Euler probe to the half step, then the derivative
/// there for the whole step. Order 2.
template <OdeState S>
class MidpointStepper {
public:
    using State = S;
    using Scalar = StateScalarT<S>;

    static constexpr int order = 2;

    template <OdeSystem<State> System>
    void step(const System& system, Scalar time, const State& state, Scalar h,
              State& out) {
        m_first = system(time, state);
        m_scratch = state + m_first * (h / Scalar{2});
        m_second = system(time + h / Scalar{2}, m_scratch);
        m_evaluations += 2;
        out = state + m_second * h;
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    State m_first{};
    State m_second{};
    State m_scratch{};
    std::size_t m_evaluations = 0;
};

/// Heun's method: an Euler probe to the far end, then the average of the two
/// derivatives. Order 2, the same as midpoint, at the same cost; they differ
/// in their error constants and in their stability regions.
template <OdeState S>
class HeunStepper {
public:
    using State = S;
    using Scalar = StateScalarT<S>;

    static constexpr int order = 2;

    template <OdeSystem<State> System>
    void step(const System& system, Scalar time, const State& state, Scalar h,
              State& out) {
        m_first = system(time, state);
        m_scratch = state + m_first * h;
        m_second = system(time + h, m_scratch);
        m_evaluations += 2;
        out = state + (m_first + m_second) * (h / Scalar{2});
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    State m_first{};
    State m_second{};
    State m_scratch{};
    std::size_t m_evaluations = 0;
};

/// Semi-implicit Euler, also called symplectic Euler: update the velocity
/// first, then move the position with the velocity you just computed.
///
/// One line different from explicit Euler and a completely different method.
/// It is symplectic, so the energy of a periodic system oscillates within a
/// bound set by the step size instead of growing, and it is exactly reversible
/// in the sense that matters for a long integration. Still order 1: the
/// trajectory error grows in proportion to the step. Order and structure are
/// separate properties, and this is the pair that shows it.
///
/// Takes an acceleration a(t, q), not a derivative.
template <OdeState S>
class SemiImplicitEulerStepper {
public:
    using State = PhaseState<S>;
    using Scalar = StateScalarT<S>;

    static constexpr int order = 1;

    template <AccelerationField<S> Acceleration>
    void step(const Acceleration& acceleration, Scalar time, const State& state,
              Scalar h, State& out) {
        m_acceleration = acceleration(time, state.position);
        ++m_evaluations;

        // The velocity is advanced first, and the position then moves with the
        // new velocity rather than the old one. That single reordering is what
        // makes the map area-preserving.
        out.velocity = state.velocity + m_acceleration * h;
        out.position = state.position + out.velocity * h;
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    S m_acceleration{};
    std::size_t m_evaluations = 0;
};

}  // namespace ysq
