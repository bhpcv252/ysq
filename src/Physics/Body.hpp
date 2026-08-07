#pragma once

#include <Math/Quaternion.hpp>
#include <Units/Constants.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

/// Matter state: what a physical body is made of and where it is.
///
/// The dynamical variable is momentum, not velocity. In the Newtonian limit
/// velocity is just momentum over mass, but momentum is the primitive that
/// stays correct once something moves fast enough that Mechanics/Kinematics's
/// relativistic momentum, p = gamma m v, applies: velocity alone cannot
/// reconstruct that, while momentum already is that quantity.
///
/// `radius`, `j2`, `principalMomentsOfInertia`, `orientation` and
/// `angularMomentum` are all general physical properties a body may or may
/// not need, the same status `charge` already has: most theories ignore
/// them, but they describe the body, not any one scenario. Every one of them
/// defaults to a value that reproduces a plain point mass exactly, so every
/// existing body in this engine is unaffected.
struct Body {
    Mass mass{};
    ElectricCharge charge{};
    Length3 position{};
    Momentum3 momentum{};

    /// Physical extent. Zero is a point, today's implicit assumption
    /// everywhere else in this engine, made explicit rather than replaced.
    Length radius{};

    /// Gravitational oblateness (quadrupole) coefficient. Zero is a sphere:
    /// Gravity/Newtonian.hpp's field reduces to the plain point-mass term
    /// wherever this is zero, on the source side. Dimensionless by
    /// convention (J2 is normalized by mass * radius^2), so no Quantity
    /// wrapper.
    double j2 = 0.0;

    /// Ixx, Iyy, Izz about this body's own principal axes, in the frame
    /// `orientation` rotates into the inertial frame. Zero means rotation is
    /// not modeled for this body at all; see Mechanics/RigidBody.hpp.
    MomentOfInertia3 principalMomentsOfInertia{};

    /// Attitude: rotates a vector from this body's frame (whose +Z is its
    /// J2/polar axis, by convention) into the inertial frame `position` and
    /// `momentum` are already expressed in.
    Quat orientation = Quat::identity();

    /// The rotational dynamical variable, the same role `momentum` plays for
    /// translation: Mechanics/RigidBody.hpp integrates this, not an angular
    /// velocity directly, for the same reason Kinematics integrates momentum
    /// rather than velocity.
    AngularMomentum3 angularMomentum{};

    /// Non-relativistic velocity, p / m. Exact only while v << c; see
    /// Mechanics/Kinematics.hpp for the relativistic relation between the two.
    [[nodiscard]] constexpr Velocity3 velocity() const noexcept {
        return momentum / mass;
    }
};

}  // namespace ysq
