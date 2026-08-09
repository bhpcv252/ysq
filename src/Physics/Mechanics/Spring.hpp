#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

namespace dim {

/// Force per unit extension: Hooke's law, `F = -k x`, needs this so `k`
/// times a length comes out as a force.
using SpringConstant = Div<Force, Length>;

/// Force per unit speed: the damping term's own coefficient, `F = -c v`.
using DampingCoefficient = Div<Force, Velocity>;

}  // namespace dim

using SpringConstant = Quantity<dim::SpringConstant>;
using DampingCoefficient = Quantity<dim::DampingCoefficient>;

/// Hooke's law, with optional linear damping along the spring's own axis: a
/// general, reusable elastic-force model, not tied to any one scenario (a
/// suspension, a tether, a mass-spring cloth all assemble one from bodies
/// and anchors).
///
/// `F = -k (|r| - restLength) rHat - c (v . rHat) rHat`, where `r` is the
/// separation (from a fixed anchor, or between two bodies in the two-body
/// overload below) and `rHat` its own unit direction: positive `stiffness`
/// pulls `on` toward the anchor when stretched past `restLength` and pushes
/// it away when compressed short of it; positive `damping` opposes whichever
/// component of velocity lies along that same axis, however stretched or
/// compressed the spring currently is.

/// The force on `on` from a spring anchored at a fixed point `anchor` -- a
/// point in the inertial frame, not itself a simulated body, so it
/// contributes no velocity of its own to the damping term.
[[nodiscard]] inline Force3
springForce(const Body& on, const Length3& anchor, SpringConstant stiffness,
            Length restLength, DampingCoefficient damping = DampingCoefficient{0.0}) {
    const Length3 separation = on.position - anchor;
    const Length distance = length(separation);
    if (isNearZero(distance)) {
        return Force3::zero();
    }
    const Vec3 direction = normalized(separation.value());

    const Force stretch = stiffness * (distance - restLength);
    const Speed alongAxis{dot(on.velocity().value(), direction)};
    const Force dampingForce = damping * alongAxis;

    return -(stretch + dampingForce) * direction;
}

/// The force `a` feels from a spring connecting it to `b`: the same law,
/// with `b`'s own position and velocity standing in for the fixed anchor
/// above, so the relative velocity along the spring's axis includes both
/// bodies' motion. `b`'s own force is exactly the negation of this result
/// (Newton's third law), which a caller applies to `b` directly rather than
/// calling this again with the arguments swapped.
[[nodiscard]] inline Force3
springForce(const Body& a, const Body& b, SpringConstant stiffness, Length restLength,
            DampingCoefficient damping = DampingCoefficient{0.0}) {
    const Length3 separation = a.position - b.position;
    const Length distance = length(separation);
    if (isNearZero(distance)) {
        return Force3::zero();
    }
    const Vec3 direction = normalized(separation.value());

    const Force stretch = stiffness * (distance - restLength);
    const Velocity3 relativeVelocity = a.velocity() - b.velocity();
    const Speed alongAxis{dot(relativeVelocity.value(), direction)};
    const Force dampingForce = damping * alongAxis;

    return -(stretch + dampingForce) * direction;
}

}  // namespace ysq
