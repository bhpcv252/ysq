#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

namespace dim {

/// Force per unit speed: the same dimension `Spring.hpp`'s
/// `DampingCoefficient` has, declared independently here rather than
/// depending on that header for a `using` alias, the same relationship
/// `Units/Temperature.hpp`'s `ThermalDiffusivity` has to
/// `Units/Fluids.hpp`'s `KinematicViscosity`.
using LinearDragCoefficient = Div<Force, Velocity>;

}  // namespace dim

using LinearDragCoefficient = Quantity<dim::LinearDragCoefficient>;

/// Stokes (linear) drag, `F = -b v_rel`: the regime a small, slow object in
/// a viscous medium is actually in, where drag scales with speed rather
/// than speed squared. `mediumVelocity` defaults to a stationary medium; a
/// nonzero value is how a caller adds wind, a current, or any other bulk
/// flow the body moves relative to.
[[nodiscard]] inline Force3
linearDragForce(const Body& on, LinearDragCoefficient coefficient,
                const Velocity3& mediumVelocity = Velocity3::zero()) {
    return -coefficient * (on.velocity() - mediumVelocity);
}

/// Quadratic drag, `F = -(1/2) rho Cd A |v_rel| v_rel`: the regime most
/// solid objects moving through air or water at everyday speed are
/// actually in (high Reynolds number, where drag scales with speed
/// squared rather than speed, unlike `linearDragForce` above).
/// `dragCoefficient` is the dimensionless shape factor `Cd` (about 0.47 for
/// a sphere, 1.05 for a flat plate face-on); `crossSectionalArea` is the
/// body's own projected area facing the flow.
[[nodiscard]] inline Force3
quadraticDragForce(const Body& on, Density mediumDensity, double dragCoefficient,
                   Area crossSectionalArea,
                   const Velocity3& mediumVelocity = Velocity3::zero()) {
    const Velocity3 relativeVelocity = on.velocity() - mediumVelocity;
    const Speed relativeSpeed = length(relativeVelocity);
    if (isNearZero(relativeSpeed)) {
        return Force3::zero();
    }

    const Vec3 direction = normalized(relativeVelocity.value());
    const Force magnitude = mediumDensity * crossSectionalArea * relativeSpeed *
                            relativeSpeed * (0.5 * dragCoefficient);
    return -magnitude * direction;
}

}  // namespace ysq
