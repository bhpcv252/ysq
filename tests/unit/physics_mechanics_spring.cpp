#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Mechanics/Spring.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Body;
using ysq::DampingCoefficient;
using ysq::Force;
using ysq::Force3;
using ysq::Length;
using ysq::Length3;
using ysq::SpringConstant;
using ysq::Vec3;

}  // namespace

TEST(PhysicsMechanicsSpring, StretchedSpringPullsTowardTheAnchor) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{5.0, 0.0, 0.0}};

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    const SpringConstant k{2.0};
    const Length restLength{3.0};

    // Stretched by 2 m past rest length, so |F| = k * 2 = 4 N, toward the
    // anchor (in the -x direction, since the body sits at +x).
    const Force3 force = ysq::springForce(body, anchor, k, restLength);
    EXPECT_QUANTITY_VEC_NEAR(force, (Force3{Vec3{-4.0, 0.0, 0.0}}), Force{1e-9});
}

TEST(PhysicsMechanicsSpring, CompressedSpringPushesAwayFromTheAnchor) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{1.0, 0.0, 0.0}};

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    const SpringConstant k{2.0};
    const Length restLength{3.0};

    // Compressed by 2 m short of rest length, so the force pushes outward
    // (+x, away from the anchor) with the same magnitude as the stretched case.
    const Force3 force = ysq::springForce(body, anchor, k, restLength);
    EXPECT_QUANTITY_VEC_NEAR(force, (Force3{Vec3{4.0, 0.0, 0.0}}), Force{1e-9});
}

TEST(PhysicsMechanicsSpring, AtRestLengthTheForceIsZero) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{3.0, 0.0, 0.0}};

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    const Force3 force = ysq::springForce(body, anchor, SpringConstant{5.0}, Length{3.0});
    EXPECT_QUANTITY_VEC_NEAR(force, Force3::zero(), Force{1e-9});
}

TEST(PhysicsMechanicsSpring, DampingOpposesVelocityAlongTheSpringAxis) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{3.0, 0.0, 0.0}};  // at rest length: no stretch force
    body.momentum = ysq::Momentum3{Vec3{1.0, 0.0, 0.0}};  // moving away from the anchor

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    const Force3 force = ysq::springForce(body, anchor, SpringConstant{5.0}, Length{3.0},
                                          DampingCoefficient{2.0});

    // Purely damping (no stretch): opposes the +x velocity, so the force is
    // in -x, magnitude c * v = 2 * 1 = 2 N.
    EXPECT_QUANTITY_VEC_NEAR(force, (Force3{Vec3{-2.0, 0.0, 0.0}}), Force{1e-9});
}

TEST(PhysicsMechanicsSpring, DampingOnlyRespondsToTheAxialVelocityComponent) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.position = Length3{Vec3{3.0, 0.0, 0.0}};
    // Purely perpendicular velocity: no component along the spring axis.
    body.momentum = ysq::Momentum3{Vec3{0.0, 1.0, 0.0}};

    const Length3 anchor{Vec3{0.0, 0.0, 0.0}};
    const Force3 force = ysq::springForce(body, anchor, SpringConstant{5.0}, Length{3.0},
                                          DampingCoefficient{2.0});
    EXPECT_QUANTITY_VEC_NEAR(force, Force3::zero(), Force{1e-9});
}

TEST(PhysicsMechanicsSpring, TwoBodyFormIsEqualAndOppositeOnBothEnds) {
    Body a{};
    a.mass = ysq::Mass{1.0};
    a.position = Length3{Vec3{0.0, 0.0, 0.0}};
    a.momentum = ysq::Momentum3{Vec3{1.0, 0.0, 0.0}};

    Body b{};
    b.mass = ysq::Mass{2.0};
    b.position = Length3{Vec3{5.0, 0.0, 0.0}};
    b.momentum = ysq::Momentum3{Vec3{-1.0, 0.0, 0.0}};

    const SpringConstant k{3.0};
    const Length restLength{2.0};
    const DampingCoefficient c{1.0};

    const Force3 forceOnA = ysq::springForce(a, b, k, restLength, c);
    const Force3 forceOnB = ysq::springForce(b, a, k, restLength, c);

    EXPECT_QUANTITY_VEC_NEAR(forceOnA, -forceOnB, Force{1e-9});
}

TEST(PhysicsMechanicsSpring, TwoBodyDampingUsesTheRelativeVelocity) {
    Body a{};
    a.mass = ysq::Mass{1.0};
    a.position = Length3{Vec3{0.0, 0.0, 0.0}};
    a.momentum = ysq::Momentum3{Vec3{1.0, 0.0, 0.0}};  // moving toward b

    Body b{};
    b.mass = ysq::Mass{1.0};
    b.position = Length3{Vec3{2.0, 0.0, 0.0}};
    b.momentum = ysq::Momentum3{Vec3{1.0, 0.0, 0.0}};  // moving at the same velocity as a

    // Equal velocities: zero relative velocity, so damping contributes nothing,
    // regardless of either body's own absolute motion.
    const Force3 force = ysq::springForce(a, b, SpringConstant{5.0}, Length{2.0},
                                          DampingCoefficient{10.0});
    EXPECT_QUANTITY_VEC_NEAR(force, Force3::zero(), Force{1e-9});
}
