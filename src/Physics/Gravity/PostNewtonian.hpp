#pragma once

#include <Math/ODE.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace ysq {

/// The 1PN (first post-Newtonian) correction to the gravitational
/// acceleration on a test particle orbiting a dominant source, in the
/// standard PPN form with gamma = beta = 1 (general relativity):
///
///     a_1PN = (GM / (c^2 r^2)) * [ (4 GM/r - v^2) n + 4 (v . n) v ]
///
/// where r and v are the position and velocity of `testParticle` relative to
/// `source`, n is the unit vector from source to testParticle, and GM uses
/// only source's mass. This is the correction that produces perihelion
/// precession; src/Physics/README.md has the derivation and the analytic
/// precession-per-orbit formula it is validated against.
///
/// **Scope: two bodies, one of them a test particle.** This is exact in the
/// limit source's mass dominates, which is the regime the analytic
/// precession formula itself assumes (Mercury around the Sun, not two
/// comparable masses). A full N-body correction is the Einstein-Infeld-
/// Hoffmann equations, which add cross terms between every pair and are not
/// implemented here; extending to that is future work, not a limitation of
/// this formula being wrong for its stated regime.
///
/// The caller adds this to newtonianAcceleration(testParticle.position, {source})
/// to get the full post-Newtonian acceleration; the two rungs of the ladder
/// compose rather than replacing one another.
[[nodiscard]] Acceleration3 postNewtonianCorrection(const Body& testParticle,
                                                    const Body& source);

/// The relativistic perihelion advance a bound orbit accumulates over one
/// full orbit, in radians: `6 pi GM / (c^2 a (1 - e^2))`, the same closed
/// form `postNewtonianCorrection` is validated against in
/// tests/unit/physics_gravity.cpp (Mercury-like: a negligible-mass orbiting
/// body, dominant source). `gm` is the source's own gravitational parameter,
/// `semiMajorAxis` and `eccentricity` the orbiting body's own elements.
///
/// A caller not running a real integrator over `postNewtonianCorrection`
/// itself (a closed-form Kepler propagator, say) can still show this real
/// effect by rotating its own argument of periapsis at the rate this value
/// implies: divide by the orbital period, `2 pi / n` for mean motion
/// `n = sqrt(gm / a^3)`, to get radians per second rather than per orbit.
[[nodiscard]] double perihelionPrecessionPerOrbit(double gm, double semiMajorAxis,
                                                  double eccentricity);

/// Direct-summation Newtonian gravity (see `NewtonianField`, whose own
/// pairwise kernel and J2 term this reuses unchanged) plus, for whichever
/// bodies the caller names a primary for, the 1PN correction of this same
/// header against that primary -- a planet's star, a moon's own planet,
/// whichever nearby dominant source's relativistic correction actually
/// matters for that body.
///
/// **Unlike `NewtonianField`, this is velocity-dependent** (the 1PN term
/// needs velocity, not only position), so it is a full `OdeSystem` over
/// `PhaseState<NBodyState>` for an explicit stepper
/// (`Rk4Stepper<PhaseState<NBodyState>>`), not an `AccelerationField` a
/// symplectic stepper (`VelocityVerletStepper` and friends) could take.
/// Trading the symplectic energy-error-stays-bounded guarantee for the 1PN
/// term is the real, documented cost of turning this on; see
/// `src/Physics/README.md`'s gravity ladder section.
class RelativisticNBodySystem {
public:
    /// `primaryIndex[i]` is which body (by index into `bodies`, the same
    /// order this and every `NBodyState` this is evaluated on share) body
    /// `i`'s own 1PN correction is computed against. A negative value, or
    /// an index equal to `i` itself, means "no relativistic correction for
    /// this body" -- its own primary (nothing dominant nearby it), or any
    /// body the caller does not want one for.
    RelativisticNBodySystem(std::span<const Body> bodies, std::vector<int> primaryIndex,
                            Length softening = Length::zero());

    [[nodiscard]] PhaseState<NBodyState> operator()(
        double time, const PhaseState<NBodyState>& state) const;

private:
    NewtonianField m_newtonian;
    std::vector<double> m_primaryGravitationalParameters;  // 0 wherever primaryIndex[i] < 0
    std::vector<int> m_primaryIndex;
};

/// The time-derivative of `relativisticAccelerationTerm`, at the same
/// relative position `r` and relative velocity `v` from a source of
/// gravitational parameter `gm`. Needed alongside the acceleration itself
/// by `Physics/Mechanics/Hermite.hpp`'s scheduler, which predicts and
/// corrects a body's own state from both.
///
/// The relative *acceleration* this differentiates through (`v`'s own time
/// derivative) is the two-body Newtonian relative acceleration, `-gm r /
/// |r|^3` -- the same test-particle-around-a-dominant-source scope
/// `relativisticAccelerationTerm`'s own formula already assumes, so this
/// pulls in no more of the full N-body picture than that formula already
/// does.
///
/// Hand-derived (product/quotient rule on every term); verified against a
/// finite-difference of `relativisticAccelerationTerm` in
/// tests/unit/physics_gravity.cpp rather than trusted on the algebra
/// alone -- exactly the kind of closed-form derivative that is easy to get
/// subtly wrong.
[[nodiscard]] Vec3 relativisticJerkTerm(const Vec3& r, const Vec3& v, double gm);

/// `RelativisticNBodySystem`'s own (acceleration, jerk) counterpart:
/// direct-summation Newtonian jerk (`NewtonianJerkField`) plus, for
/// whichever bodies the caller names a primary for, this header's own
/// `relativisticJerkTerm` against that primary. Matches
/// `IndividualJerkField`.
class RelativisticNBodyJerkSystem {
public:
    RelativisticNBodyJerkSystem(std::span<const Body> bodies, std::vector<int> primaryIndex,
                                Length softening = Length::zero());

    [[nodiscard]] std::pair<Vec3, Vec3> operator()(std::size_t bodyIndex,
                                                   const NBodyState& positions,
                                                   const NBodyState& velocities) const;

private:
    NewtonianJerkField m_newtonian;
    std::vector<double> m_primaryGravitationalParameters;
    std::vector<int> m_primaryIndex;
};

}  // namespace ysq
