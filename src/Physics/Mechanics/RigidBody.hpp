#pragma once

#include <Physics/Body.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>

#include <span>

namespace ysq {

/// General rigid-body rotation: gravity-gradient torque from a body's own
/// inertia asymmetry, and Euler's rotation equation, carried as an
/// inertial-frame angular momentum rather than a body-frame angular
/// velocity, the same primitive role Body::momentum already plays for
/// translation. See src/Physics/README.md for the derivation.
///
/// General for any body with a nonzero principalMomentsOfInertia: not tied
/// to Earth, and drawing on the identical inertia asymmetry
/// Gravity/Newtonian.hpp's J2 term already reads, not a second, independent
/// formula.

/// The gravity-gradient torque an external mass (`perturberMass`, at
/// `perturberPosition`) exerts on `oblateBody`'s own inertia asymmetry,
/// about its center of mass, in the inertial frame. Standard result (e.g.
/// Hughes, "Spacecraft Attitude Dynamics"): tau = (3 GM / r^3) rHat x
/// (I . rHat), evaluated in oblateBody's own body frame (where its
/// principalMomentsOfInertia is diagonal) and rotated back by its current
/// orientation. Zero whenever principalMomentsOfInertia is zero, without a
/// branch: I . rHat is then zero, and so is its cross product with rHat.
[[nodiscard]] Torque3 gravityGradientTorque(const Body& oblateBody,
                                            const Length3& perturberPosition,
                                            Mass perturberMass);

/// Summed over every body in `perturbers`. Callers exclude oblateBody itself
/// from `perturbers`, the same convention Gravity/Newtonian.hpp's `sources`
/// spans use.
[[nodiscard]] Torque3 gravityGradientTorque(const Body& oblateBody,
                                            std::span<const Body> perturbers);

/// Advances `body`'s orientation and angularMomentum by one fixed step
/// `dt`, under the gravity-gradient torque from `perturbers` (which move
/// during the step no more than Gravity/Newtonian.hpp's own bodies do
/// within one fixed step of the same size, the same order of approximation
/// already accepted for a fixed-step integration), and renormalizes the
/// resulting orientation: RK4 does not preserve the unit-quaternion
/// constraint exactly, the standard reason quaternion integration
/// renormalizes after each accepted step. A no-op, exactly, when
/// `body.principalMomentsOfInertia` is zero: rotation is not modeled for
/// that body, and nothing here decides which bodies that applies to, only
/// what a body that opts in needs.
void stepRigidBody(Body& body, std::span<const Body> perturbers, double dt);

}  // namespace ysq
