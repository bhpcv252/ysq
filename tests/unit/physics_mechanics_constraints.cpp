#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Mechanics/Constraints.hpp>
#include <Units/Acceleration.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Time.hpp>
#include <Units/Velocity.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Body;
using ysq::Length;
using ysq::Length3;
using ysq::Time;
using ysq::Vec3;

}  // namespace

TEST(PhysicsMechanicsConstraints, DistanceConstraintRemovesRadialApproachVelocity) {
    Body a{};
    a.mass = ysq::Mass{1.0};
    a.position = Length3{Vec3{0.0, 0.0, 0.0}};

    Body b{};
    b.mass = ysq::Mass{1.0};
    b.position = Length3{Vec3{2.0, 0.0, 0.0}};          // already at the target distance
    b.momentum = ysq::Momentum3{Vec3{-1.0, 0.0, 0.0}};  // approaching a

    ysq::solveDistanceConstraint(a, b, Length{2.0}, Time{0.01}, 0.0);

    const double radialVelocity =
        ysq::dot((b.velocity() - a.velocity()).value(), Vec3{1.0, 0.0, 0.0});
    EXPECT_NEAR(radialVelocity, 0.0, 1e-9);
}

TEST(PhysicsMechanicsConstraints, DistanceConstraintConservesTotalMomentum) {
    Body a{};
    a.mass = ysq::Mass{2.0};
    a.position = Length3{Vec3{0.0, 0.0, 0.0}};
    a.momentum = ysq::Momentum3{Vec3{1.0, 1.0, 0.0}};

    Body b{};
    b.mass = ysq::Mass{3.0};
    b.position = Length3{Vec3{2.0, 0.0, 0.0}};
    b.momentum = ysq::Momentum3{Vec3{-1.0, 0.5, 0.0}};

    const ysq::Momentum3 totalBefore = a.momentum + b.momentum;
    ysq::solveDistanceConstraint(a, b, Length{2.5}, Time{0.01}, 0.2);
    const ysq::Momentum3 totalAfter = a.momentum + b.momentum;

    EXPECT_QUANTITY_VEC_NEAR(totalAfter, totalBefore, ysq::Momentum{1e-9});
}

TEST(PhysicsMechanicsConstraints,
     DistanceConstraintWithBaumgarteStabilizationClosesAGap) {
    Body a{};
    a.mass = ysq::Mass{1.0};
    a.position = Length3{Vec3{0.0, 0.0, 0.0}};

    Body b{};
    b.mass = ysq::Mass{1.0};
    b.position = Length3{Vec3{3.0, 0.0, 0.0}};  // 1 m past the target distance of 2

    ysq::solveDistanceConstraint(a, b, Length{2.0}, Time{0.1}, 0.2);

    // Baumgarte stabilization should introduce a closing velocity: b moving
    // toward a, a moving toward b.
    EXPECT_LT(b.velocity().value().x, 0.0);
    EXPECT_GT(a.velocity().value().x, 0.0);
}

TEST(PhysicsMechanicsConstraints,
     DistanceConstraintWithZeroBaumgarteLeavesPositionErrorAlone) {
    // With baumgarteFactor = 0, no positional correction is introduced: a
    // pair already at rest relative to each other, even with a position
    // error, stays at rest.
    Body a{};
    a.mass = ysq::Mass{1.0};
    a.position = Length3{Vec3{0.0, 0.0, 0.0}};

    Body b{};
    b.mass = ysq::Mass{1.0};
    b.position = Length3{Vec3{3.0, 0.0, 0.0}};

    ysq::solveDistanceConstraint(a, b, Length{2.0}, Time{0.1}, 0.0);

    EXPECT_QUANTITY_VEC_NEAR(a.momentum, ysq::Momentum3::zero(), ysq::Momentum{1e-12});
    EXPECT_QUANTITY_VEC_NEAR(b.momentum, ysq::Momentum3::zero(), ysq::Momentum{1e-12});
}

TEST(PhysicsMechanicsConstraints, AnchoredDistanceConstraintRemovesRadialVelocity) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{2.0, 0.0, 0.0}};
    // Purely radial (away from the anchor) velocity.
    body.momentum = ysq::Momentum3{Vec3{1.0, 0.0, 0.0}};

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    ysq::solveDistanceConstraint(body, anchor, Length{2.0}, Time{0.01}, 0.0);

    EXPECT_NEAR(body.velocity().value().x, 0.0, 1e-9);
}

TEST(PhysicsMechanicsConstraints, AnchoredDistanceConstraintPreservesTangentialVelocity) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{2.0, 0.0, 0.0}};
    // Purely tangential velocity (a pendulum bob swinging): should be untouched.
    body.momentum = ysq::Momentum3{Vec3{0.0, 3.0, 0.0}};

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    ysq::solveDistanceConstraint(body, anchor, Length{2.0}, Time{0.01}, 0.0);

    EXPECT_NEAR(body.velocity().value().y, 3.0, 1e-9);
}

TEST(PhysicsMechanicsConstraints, PendulumStaysNearItsRodLengthOverManySteps) {
    // A simple pendulum: gravity plus a distance constraint to a fixed
    // pivot, integrated with plain symplectic (semi-implicit) Euler.
    // Physical correctness of *dynamics* isn't the point here (there is no
    // proper constraint-aware integrator involved); what this checks is
    // that repeatedly applying the constraint keeps the rod length from
    // drifting very far over many steps, which is what Baumgarte
    // stabilization exists for.
    Body bob{};
    bob.mass = ysq::Mass{1.0};
    bob.position = Length3{Vec3{2.0, 0.0, 0.0}};  // horizontal, rod length 2

    const Length3 pivot{Vec3{0.0, 0.0, 0.0}};
    const Length rodLength{2.0};
    const Time dt{0.01};
    const ysq::Acceleration3 gravity{Vec3{0.0, -9.8, 0.0}};

    for (int step = 0; step < 2000; ++step) {
        bob.momentum += ysq::Momentum3{gravity.value() * bob.mass.value() * dt.value()};
        ysq::solveDistanceConstraint(bob, pivot, rodLength, dt, 0.2);
        bob.position += Length3{bob.velocity().value() * dt.value()};
    }

    const double finalDistance = ysq::length(bob.position - pivot).value();
    EXPECT_NEAR(finalDistance, rodLength.value(), 0.05);
}

TEST(PhysicsMechanicsConstraints, PointConstraintFullyCancelsVelocityAtTheAnchor) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{0.0, 0.0, 0.0}};  // already at the anchor
    body.momentum = ysq::Momentum3{Vec3{3.0, -2.0, 1.0}};

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    ysq::solvePointConstraint(body, anchor, Time{0.01}, 0.2);

    EXPECT_QUANTITY_VEC_NEAR(body.momentum, ysq::Momentum3::zero(), ysq::Momentum{1e-9});
}

TEST(PhysicsMechanicsConstraints, PointConstraintPullsAnOffsetBodyTowardTheAnchor) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{1.0, 0.0, 0.0}};  // offset from the anchor

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    ysq::solvePointConstraint(body, anchor, Time{0.1}, 0.2);

    // Baumgarte bias should introduce velocity pulling the body back toward
    // the anchor (in -x).
    EXPECT_LT(body.velocity().value().x, 0.0);
}

TEST(PhysicsMechanicsConstraints, PointConstraintConstrainsAllThreeAxesAtOnce) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{1.0, 2.0, -1.0}};

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    ysq::solvePointConstraint(body, anchor, Time{0.1}, 0.2);

    EXPECT_LT(body.velocity().value().x, 0.0);
    EXPECT_LT(body.velocity().value().y, 0.0);
    EXPECT_GT(body.velocity().value().z, 0.0);
}
