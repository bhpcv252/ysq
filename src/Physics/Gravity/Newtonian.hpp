#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>

#include <span>
#include <vector>

namespace ysq {

namespace dim {

/// The Newtonian gravitational constant's dimension, L^3 / (M T^2).
using GravitationalConstant = Div<Raise<Length, 3>, Mul<Mass, Raise<Time, 2>>>;

}  // namespace dim

using GravitationalConstant = Quantity<dim::GravitationalConstant>;

namespace constants {

/// The Newtonian gravitational constant. Measured, CODATA 2018/2022, relative
/// standard uncertainty about 2.2e-5.
///
/// This lives here rather than in Units/Constants.hpp on purpose: it is not
/// one of the seven constants that define the SI, it parameterizes one
/// specific interaction, and src/Units/README.md is explicit that it belongs
/// with the gravity that uses it. Wherever a source's mass parameter GM is already
/// known directly (Constants.hpp's nominalSolarMassParameter, for instance),
/// prefer that: it does not carry this uncertainty, because it is what an
/// orbit actually measures, and G only re-enters when recovering it from a
/// mass in kilograms.
inline constexpr GravitationalConstant G{6.67430e-11};

}  // namespace constants

/// Newtonian gravity: F = G m1 m2 / r^2, toward the source. The weak-field,
/// slow-motion limit of general relativity, and the rung of the gravity
/// ladder that drives dynamical many-body systems; see src/Physics/README.md's
/// "The gravity ladder" section.
///
/// Every function here that takes a softening length uses Plummer softening,
/// a = GM (r_j - r_i) / (|r_j - r_i|^2 + softening^2)^(3/2): the same, exact
/// Newtonian force at separations well above the softening length, and finite
/// rather than singular as two bodies approach each other. A direct-summation
/// N-body integration needs it; see src/Physics/README.md's "Softening"
/// section.
///
/// **Oblateness.** Every function here also adds a source's J2 term when its
/// `Body::j2` is nonzero: not a second law, the same integral of Newton's
/// law over a non-point mass distribution, evaluated to its first
/// non-trivial (quadrupole) order rather than assumed to be a point. Zero
/// `j2` reproduces the plain point-mass term exactly, with no branch; see
/// src/Physics/README.md for the derivation.

[[nodiscard]] Force3 newtonianForce(const Body& on, const Body& from);

/// The acceleration at `at` due to every body in `sources`, summed directly.
[[nodiscard]] Acceleration3 newtonianAcceleration(const Length3& at,
                                                  std::span<const Body> sources,
                                                  Length softening = Length::zero());

/// Direct-sum acceleration on every body in `bodies` due to every other body
/// in it. O(n^2); see BarnesHut.hpp for the O(n log n) approximation.
[[nodiscard]] std::vector<Acceleration3>
newtonianAccelerations(std::span<const Body> bodies, Length softening = Length::zero());

/// Total gravitational potential energy of the system, summed once per pair
/// and with the same softening the acceleration was computed with: the two
/// have to agree for a conservation check to mean anything, since it is the
/// softened system, not the true 1/r one, that was actually integrated.
[[nodiscard]] Energy newtonianPotentialEnergy(std::span<const Body> bodies,
                                              Length softening = Length::zero());

/// A raw (unit-stripped) direct-summation acceleration field over N bodies'
/// positions, for use with Math's steppers: VelocityVerletStepper<NBodyState>
/// and friends take this directly as the AccelerationField. See
/// Mechanics/Dynamics.hpp for NBodyState and why the integrator boundary
/// carries no units.
class NewtonianField {
public:
    explicit NewtonianField(std::span<const Body> bodies,
                            Length softening = Length::zero());

    [[nodiscard]] NBodyState operator()(double time, const NBodyState& positions) const;

private:
    std::vector<double> m_gravitationalParameters;  // G * mass, per body, m^3/s^2
    // (3/2) * j2 * G * mass * equatorialRadius^2, per body; zero wherever
    // that body's j2 is zero, so the oblateness term drops out on its own.
    std::vector<double> m_j2Coefficients;
    std::vector<Vec3> m_spinAxes;  // per body, meaningless where j2Coefficient is zero
    double m_softeningSquared;
};

}  // namespace ysq
