#pragma once

#include <Math/Geometry/Primitives.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Units/Length.hpp>

#include <optional>

namespace ysq {

/// One point of contact between two overlapping shapes: where they touch,
/// the axis pushing them apart, and how far they already overlap.
///
/// `normal` points away from the first shape passed to whichever
/// `detectCollision` overload produced this contact, toward the second: for
/// two spheres, from the first body's center toward the second's; for a
/// sphere against a box, from the sphere's center toward the box. Both
/// `resolveCollision` overloads below are written against that one
/// convention.
struct Contact {
    Length3 point;
    Vec3 normal;
    Length penetrationDepth;
};

/// Sphere-sphere overlap, using each body's own `radius`. `nullopt` if they
/// do not overlap.
[[nodiscard]] std::optional<Contact> detectCollision(const Body& a, const Body& b);

/// Sphere-box overlap: `sphere`'s own `radius` against a fixed,
/// axis-aligned box in the same (meters) frame `sphere.position` is
/// already in -- a wall, a boundary, anything not itself simulated as a
/// `Body`. `nullopt` if they do not overlap.
[[nodiscard]] std::optional<Contact> detectCollision(const Body& sphere,
                                                     const AABB3<double>& box);

/// Resolves `contact` between two bodies by applying an instantaneous
/// impulse to both bodies' momentum along `contact.normal`: the standard
/// point-mass impulse formula, `j = -(1 + restitution) v_rel.n / (1/m_a +
/// 1/m_b)`. `restitution` of 0 is perfectly inelastic (the bodies end up
/// moving at the same velocity along the normal); 1 is perfectly elastic
/// (kinetic energy along the normal is conserved exactly). A no-op if the
/// bodies are already separating along the normal, so calling this on a
/// contact from a previous step that has since resolved itself does
/// nothing rather than pulling them back together.
///
/// Does not move either body apart to resolve the penetration itself --
/// `contact.penetrationDepth` is there for a caller that wants to, by
/// whatever positional-correction scheme its own integrator prefers.
void resolveCollision(Body& a, Body& b, const Contact& contact, double restitution = 1.0);

/// The same resolution against an immovable object (infinite mass, the
/// limit `1/m_b -> 0` of the two-body formula above): only `body`'s own
/// momentum changes.
void resolveCollision(Body& body, const Contact& contact, double restitution = 1.0);

}  // namespace ysq
