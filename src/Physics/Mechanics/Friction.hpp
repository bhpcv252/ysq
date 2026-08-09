#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Units/Force.hpp>
#include <Units/Velocity.hpp>

#include <algorithm>

namespace ysq {

/// Coulomb friction between a body and a surface, split into the same two
/// regimes real friction has. Both take `contactNormal` (unit vector,
/// pointing away from the surface into the body -- the same convention
/// `Collision.hpp`'s `Contact::normal` uses) and a normal-force magnitude
/// the caller already knows, from a collision impulse, a resting contact,
/// or simply `m g cos(theta)` on an incline: this module has no opinion on
/// where that number comes from, only on what Coulomb's law does with it
/// once known.

/// Kinetic friction: `F = -mu_k N tHat`, opposing whichever direction `on`
/// is actually sliding relative to `surfaceVelocity` (a stationary surface
/// by default), projected onto the plane perpendicular to `contactNormal`
/// -- friction never has a component along the normal itself. Zero if
/// there is no tangential relative motion (nothing to oppose); reach for
/// `staticFrictionForce` there instead.
[[nodiscard]] inline Force3
kineticFrictionForce(const Body& on, const Vec3& contactNormal,
                     Force normalForceMagnitude, double kineticFrictionCoefficient,
                     const Velocity3& surfaceVelocity = Velocity3::zero()) {
    const Vec3 relativeVelocity = (on.velocity() - surfaceVelocity).value();
    const Vec3 tangentialVelocity =
        relativeVelocity - contactNormal * dot(relativeVelocity, contactNormal);
    const double tangentialSpeed = length(tangentialVelocity);
    if (isNearZero(tangentialSpeed)) {
        return Force3::zero();
    }

    const Vec3 tangentDirection = tangentialVelocity / tangentialSpeed;
    return -(normalForceMagnitude * kineticFrictionCoefficient) * tangentDirection;
}

/// Static friction: holds `on` in place against a `drivingForce` (gravity's
/// component along an incline, an applied push) as long as the tangential
/// part of that force is within the Coulomb limit `mu_s N`. Returns exactly
/// the negation of the driving force's tangential component while under
/// that limit (equilibrium: nothing slides), and the capped value `mu_s N`
/// opposing it once the limit is exceeded -- the point past which the body
/// actually starts to slide and `kineticFrictionForce` above is the
/// correct law instead.
[[nodiscard]] inline Force3 staticFrictionForce(const Vec3& contactNormal,
                                                Force normalForceMagnitude,
                                                double staticFrictionCoefficient,
                                                const Force3& drivingForce) {
    const Vec3 driving = drivingForce.value();
    const Vec3 tangentialDriving = driving - contactNormal * dot(driving, contactNormal);
    const double tangentialMagnitude = length(tangentialDriving);
    if (isNearZero(tangentialMagnitude)) {
        return Force3::zero();
    }

    const Vec3 direction = tangentialDriving / tangentialMagnitude;
    const double maxStatic = normalForceMagnitude.value() * staticFrictionCoefficient;
    const double appliedMagnitude = std::min(tangentialMagnitude, maxStatic);

    return Force3{-direction * appliedMagnitude};
}

}  // namespace ysq
