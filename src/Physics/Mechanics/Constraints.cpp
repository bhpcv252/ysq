#include <Physics/Mechanics/Constraints.hpp>

#include <Math/Scalar.hpp>
#include <Units/Force.hpp>
#include <Units/Mass.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

void solveDistanceConstraint(Body& a, Body& b, Length targetDistance, Time dt,
                             double baumgarteFactor) {
    const Length3 separation = b.position - a.position;
    const Length distance = length(separation);
    if (isNearZero(distance)) {
        return;
    }
    const Vec3 normal = normalized(separation.value());

    const Velocity3 relativeVelocity = b.velocity() - a.velocity();
    const Speed velocityAlongNormal{dot(relativeVelocity.value(), normal)};
    const Length separationError = distance - targetDistance;
    const Speed bias = (separationError / dt) * baumgarteFactor;

    const double inverseMassSum = 1.0 / a.mass.value() + 1.0 / b.mass.value();
    const double lambda = -(velocityAlongNormal.value() + bias.value()) / inverseMassSum;
    const Momentum3 impulse{normal * lambda};

    a.momentum -= impulse;
    b.momentum += impulse;
}

void solveDistanceConstraint(Body& body, const Length3& anchor, Length targetDistance,
                             Time dt, double baumgarteFactor) {
    // The infinite-mass limit of the two-body form above, with `body` in
    // the role of `a` and the stationary `anchor` in the role of `b`:
    // separation and relative velocity both reduce to `body`'s own state
    // relative to a fixed point, and `1/m_b -> 0` drops the anchor's own
    // (nonexistent) momentum change.
    const Length3 separation = anchor - body.position;
    const Length distance = length(separation);
    if (isNearZero(distance)) {
        return;
    }
    const Vec3 normal = normalized(separation.value());

    const Speed velocityAlongNormal{-dot(body.velocity().value(), normal)};
    const Length separationError = distance - targetDistance;
    const Speed bias = (separationError / dt) * baumgarteFactor;

    const double lambda =
        -(velocityAlongNormal.value() + bias.value()) * body.mass.value();
    body.momentum -= Momentum3{normal * lambda};
}

void solvePointConstraint(Body& body, const Length3& anchor, Time dt,
                          double baumgarteFactor) {
    const Length3 positionError = body.position - anchor;
    const Velocity3 bias = (positionError / dt) * baumgarteFactor;
    const Velocity3 desiredVelocity = -bias;
    const Velocity3 velocityCorrection = desiredVelocity - body.velocity();

    body.momentum += Momentum3{velocityCorrection.value() * body.mass.value()};
}

}  // namespace ysq
