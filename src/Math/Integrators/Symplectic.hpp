#pragma once

#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>

#include <cmath>
#include <cstddef>

namespace ysq {

/// Symplectic integrators for a separable system, H = T(p) + V(q).
///
/// All of these take an acceleration a(t, q) and advance a PhaseState. They
/// are built the same way: alternate moving the position at fixed velocity
/// (a drift) with changing the velocity at fixed position (a kick). Each half
/// is an exactly solvable, area-preserving map, so any composition of them is
/// too, whatever the coefficients.
///
/// **What that buys, and what it does not.** A symplectic method does not
/// conserve the energy of the system it was given; it exactly conserves the
/// energy of a nearby one. The difference between the two is fixed by the step
/// size and does not accumulate, so the energy error oscillates within a band
/// forever rather than drifting. It says nothing about the trajectory error,
/// which grows with time for these exactly as it does for anything else. A
/// symplectic method is the right choice for a long run whose invariants
/// matter, not for a short run that has to end up in precisely the right
/// place.
///
/// **Strictly, the guarantee holds for an autonomous system.** The time is
/// threaded through the substages so an explicitly time-dependent force still
/// integrates correctly, but a driven system is not a Hamiltonian one and the
/// bounded energy error is not promised there.

/// Velocity Verlet, also called the kick-drift-kick leapfrog. Order 2.
///
/// The workhorse. Two evaluations per step, time-reversible, and the standard
/// choice for molecular dynamics and for gravitational N-body over many
/// orbits.
template <OdeState S>
class VelocityVerletStepper {
public:
    using State = PhaseState<S>;
    using Scalar = StateScalarT<S>;

    static constexpr int order = 2;

    template <AccelerationField<S> Acceleration>
    void step(const Acceleration& acceleration, Scalar time, const State& state,
              Scalar h, State& out) {
        const Scalar half = h / Scalar{2};

        // Half a kick, a full drift, then the other half kick with the
        // acceleration at the new position. The symmetry of that sandwich is
        // what makes it time-reversible and second order rather than first.
        m_acceleration = acceleration(time, state.position);
        m_halfVelocity = state.velocity + m_acceleration * half;

        out.position = state.position + m_halfVelocity * h;

        m_acceleration = acceleration(time + h, out.position);
        out.velocity = m_halfVelocity + m_acceleration * half;

        m_evaluations += 2;
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    S m_acceleration{};
    S m_halfVelocity{};
    std::size_t m_evaluations = 0;
};

/// Forest-Ruth, the fourth-order symplectic method from Yoshida's composition
/// of three Verlet-like steps with one negative middle coefficient. Order 4,
/// three evaluations per step.
///
/// The middle step runs backwards in time, which is unavoidable: no
/// composition of forward-only symplectic steps reaches fourth order. That is
/// also why it is less accurate per evaluation than its order suggests, and
/// why PEFRL exists.
///
/// Yoshida, "Construction of higher order symplectic integrators", Phys. Lett.
/// A 150 (1990), 262-268.
template <OdeState S>
class ForestRuthStepper {
public:
    using State = PhaseState<S>;
    using Scalar = StateScalarT<S>;

    static constexpr int order = 4;

    template <AccelerationField<S> Acceleration>
    void step(const Acceleration& acceleration, Scalar time, const State& state,
              Scalar h, State& out) {
        // theta = 1 / (2 - 2^(1/3)), the root of the condition that the third
        // order error terms cancel. Computed rather than transcribed.
        const Scalar theta = Scalar{1} / (Scalar{2} - std::cbrt(Scalar{2}));

        const Scalar drift0 = theta / Scalar{2};
        const Scalar drift1 = (Scalar{1} - theta) / Scalar{2};
        const Scalar kick0 = theta;
        const Scalar kick1 = Scalar{1} - Scalar{2} * theta;

        S position = state.position;
        S velocity = state.velocity;
        Scalar at = time;

        position += velocity * (drift0 * h);
        at += drift0 * h;
        velocity += acceleration(at, position) * (kick0 * h);

        position += velocity * (drift1 * h);
        at += drift1 * h;
        velocity += acceleration(at, position) * (kick1 * h);

        position += velocity * (drift1 * h);
        at += drift1 * h;
        velocity += acceleration(at, position) * (kick0 * h);

        position += velocity * (drift0 * h);

        m_evaluations += 3;
        out.position = position;
        out.velocity = velocity;
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    std::size_t m_evaluations = 0;
};

/// Position Extended Forest-Ruth Like. Order 4, four evaluations per step.
///
/// One more evaluation than Forest-Ruth and a considerably smaller error
/// constant, so it is the better fourth-order choice per unit of work despite
/// costing more per step. The coefficients have no closed form; they are the
/// numerical solution of the order conditions.
///
/// Omelyan, Mryglod and Folk, "Optimized Forest-Ruth- and Suzuki-like
/// algorithms for integration of motion in many-body systems", Comput. Phys.
/// Commun. 146 (2002), 188-202.
template <OdeState S>
class PefrlStepper {
public:
    using State = PhaseState<S>;
    using Scalar = StateScalarT<S>;

    static constexpr int order = 4;

    template <AccelerationField<S> Acceleration>
    void step(const Acceleration& acceleration, Scalar time, const State& state,
              Scalar h, State& out) {
        const auto xi = static_cast<Scalar>(0.1786178958448091);
        const auto lambda = static_cast<Scalar>(-0.2123418310626054);
        const auto chi = static_cast<Scalar>(-0.06626458266981849);

        const Scalar outerKick = (Scalar{1} - Scalar{2} * lambda) / Scalar{2};
        const Scalar middleDrift = Scalar{1} - Scalar{2} * (chi + xi);

        S position = state.position;
        S velocity = state.velocity;
        Scalar at = time;

        position += velocity * (xi * h);
        at += xi * h;
        velocity += acceleration(at, position) * (outerKick * h);

        position += velocity * (chi * h);
        at += chi * h;
        velocity += acceleration(at, position) * (lambda * h);

        position += velocity * (middleDrift * h);
        at += middleDrift * h;
        velocity += acceleration(at, position) * (lambda * h);

        position += velocity * (chi * h);
        at += chi * h;
        velocity += acceleration(at, position) * (outerKick * h);

        position += velocity * (xi * h);

        m_evaluations += 4;
        out.position = position;
        out.velocity = velocity;
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return m_evaluations; }

private:
    std::size_t m_evaluations = 0;
};

}  // namespace ysq
