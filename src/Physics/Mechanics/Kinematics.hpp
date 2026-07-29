#pragma once

#include <Math/Calculus.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>
#include <Units/Constants.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <cstddef>

namespace ysq {

/// Special-relativistic kinematics: worldlines, proper time, and the
/// four-velocity. Everything here reduces to the familiar Newtonian relation
/// at v << c, which is what "Newtonian = low-v limit" in the module layout
/// means: there is one set of relations, not two.

/// The Lorentz factor, 1 / sqrt(1 - (v/c)^2).
///
/// Undefined, and NaN, at or above the speed of light. Nothing here clamps
/// that away: a clamped answer would be silently wrong, where NaN at least
/// propagates and is loud.
[[nodiscard]] inline Dimensionless lorentzFactor(Speed v) {
    const Dimensionless beta = v / constants::speedOfLight;
    return Dimensionless{1.0} / sqrt(Dimensionless{1.0} - beta * beta);
}

[[nodiscard]] inline Dimensionless lorentzFactor(const Velocity3& v) {
    return lorentzFactor(length(v));
}

/// The four-velocity u^mu = (gamma c, gamma v), components (t, x, y, z).
///
/// Normalized so that the Minkowski inner product u . u = -c^2, in the
/// (-,+,+,+) signature used throughout Physics/Spacetime.
[[nodiscard]] inline Velocity4 fourVelocity(const Velocity3& v) {
    const Dimensionless gamma = lorentzFactor(v);
    const Speed timeComponent = gamma * constants::speedOfLight;
    const Velocity3 spaceComponent = gamma * v;
    return Velocity4{Vec4{timeComponent.value(), spaceComponent.value().x,
                          spaceComponent.value().y, spaceComponent.value().z}};
}

/// dTau / dt, the rate proper time accumulates against coordinate time for an
/// object moving at speed v. 1 at v = 0, falling to 0 as v approaches c.
[[nodiscard]] inline Dimensionless properTimeRate(Speed v) {
    return Dimensionless{1.0} / lorentzFactor(v);
}

/// Proper time elapsed along a worldline whose speed is given by `speedAt`,
/// over the coordinate-time interval [from, to]: the integral of dt / gamma.
///
/// Quadrature over a callable rather than a closed-form step, because a
/// worldline's speed history is data, not a formula, in general. Units cross
/// the boundary once, at the call into Simpson's rule, and back on the way
/// out; see src/Physics/README.md's "Units cross the boundary once" section.
template <class SpeedAt>
[[nodiscard]] Time properTimeElapsed(SpeedAt&& speedAt, Time from, Time to,
                                     std::size_t intervals = 1000) {
    const auto integrand = [&](double t) {
        return properTimeRate(speedAt(Time{t})).value();
    };
    return Time{simpson(integrand, from.value(), to.value(), intervals)};
}

/// Relativistic velocity addition: the velocity `u` seen in the original
/// frame, expressed in a frame moving at `frameVelocity` relative to it.
///
/// The general (non-collinear) formula, splitting u into components parallel
/// and perpendicular to the boost direction:
///
///     u'_par  = (u_par - v) / (1 - u.v/c^2)
///     u'_perp = u_perp / (gamma_v (1 - u.v/c^2))
///
/// which is what Frame's Galilean transform approximates at v << c, where the
/// denominator is 1 and gamma_v is 1.
[[nodiscard]] inline Velocity3 relativisticVelocityAdd(const Velocity3& u,
                                                       const Velocity3& frameVelocity) {
    const Speed vMag = length(frameVelocity);
    if (isNearZero(vMag)) {
        return u;
    }

    const Vec3 vHat = normalized(frameVelocity.value());
    const Speed uParallelMag = Speed{dot(u.value(), vHat)};
    const Velocity3 uParallel = uParallelMag * vHat;
    const Velocity3 uPerp = u - uParallel;

    const Dimensionless gammaV = lorentzFactor(vMag);
    const Dimensionless denominator =
        Dimensionless{1.0} - dot(u, frameVelocity) / raised<2>(constants::speedOfLight);

    const Velocity3 numerator = (uParallel - frameVelocity) + uPerp / gammaV.value();
    return numerator / denominator.value();
}

}  // namespace ysq
