#include <Physics/Mechanics/Collision.hpp>

#include <Math/Geometry/Queries.hpp>
#include <Math/Scalar.hpp>
#include <Units/Force.hpp>
#include <Units/Mass.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

std::optional<Contact> detectCollision(const Body& a, const Body& b) {
    const Length3 separation = b.position - a.position;
    const Length distance = length(separation);
    const Length radiusSum = a.radius + b.radius;
    if (distance >= radiusSum) {
        return std::nullopt;
    }

    // Coincident centers have no well-defined separating axis; +x is an
    // arbitrary but harmless choice for a configuration this degenerate.
    const Vec3 normal =
        isNearZero(distance) ? Vec3::unitX() : normalized(separation.value());
    const Length penetration = radiusSum - distance;
    const Length3 point = a.position + Length3{normal * a.radius.value()};

    return Contact{point, normal, penetration};
}

std::optional<Contact> detectCollision(const Body& sphere, const AABB3<double>& box) {
    const Vec3 spherePosition = sphere.position.value();
    const Vec3 closest = closestPoint(box, spherePosition);
    const Vec3 sphereToClosest = closest - spherePosition;
    const double distance = length(sphereToClosest);
    const double radius = sphere.radius.value();
    if (distance >= radius) {
        return std::nullopt;
    }

    // The sphere's center already sits inside the box, so every axis
    // clamps to it and there is no well-defined nearest surface point to
    // aim the normal at; +y is an arbitrary but harmless choice for a case
    // this degenerate (a body that has fully entered a wall) rather than
    // leaving the normal undefined.
    const Vec3 normal = isNearZero(distance) ? Vec3::unitY() : sphereToClosest / distance;
    const Length penetration{radius - distance};

    return Contact{Length3{closest}, normal, penetration};
}

void resolveCollision(Body& a, Body& b, const Contact& contact, double restitution) {
    const Velocity3 relativeVelocity = b.velocity() - a.velocity();
    const double velocityAlongNormal = dot(relativeVelocity.value(), contact.normal);
    if (velocityAlongNormal >= 0.0) {
        return;
    }

    const double inverseMassSum = 1.0 / a.mass.value() + 1.0 / b.mass.value();
    const double impulseMagnitude =
        -(1.0 + restitution) * velocityAlongNormal / inverseMassSum;
    const Momentum3 impulse{contact.normal * impulseMagnitude};

    a.momentum -= impulse;
    b.momentum += impulse;
}

void resolveCollision(Body& body, const Contact& contact, double restitution) {
    const double approachSpeed = dot(body.velocity().value(), contact.normal);
    if (approachSpeed <= 0.0) {
        return;
    }

    const double impulseMagnitude =
        (1.0 + restitution) * approachSpeed * body.mass.value();
    body.momentum -= Momentum3{contact.normal * impulseMagnitude};
}

}  // namespace ysq
