#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Mechanics/Friction.hpp>
#include <Units/Force.hpp>
#include <Units/Mass.hpp>
#include <Units/Velocity.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Body;
using ysq::Force;
using ysq::Force3;
using ysq::Vec3;
using ysq::Velocity3;

}  // namespace

TEST(PhysicsMechanicsFriction, KineticFrictionOpposesTheSlidingDirection) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.momentum = ysq::Momentum3{Vec3{3.0, 0.0, 0.0}};  // sliding in +x

    const Vec3 normal{0.0, 1.0, 0.0};  // surface normal is +y
    const Force3 friction = ysq::kineticFrictionForce(body, normal, Force{10.0}, 0.3);

    EXPECT_LT(friction.value().x, 0.0);
    EXPECT_NEAR(friction.value().y, 0.0, 1e-12);
    EXPECT_NEAR(friction.value().z, 0.0, 1e-12);
}

TEST(PhysicsMechanicsFriction, KineticFrictionMagnitudeIsCoefficientTimesNormalForce) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.momentum = ysq::Momentum3{Vec3{5.0, 0.0, 0.0}};

    const Vec3 normal{0.0, 1.0, 0.0};
    const Force3 friction = ysq::kineticFrictionForce(body, normal, Force{10.0}, 0.3);

    EXPECT_NEAR(ysq::length(friction).value(), 3.0, 1e-9);  // 0.3 * 10
}

TEST(PhysicsMechanicsFriction, KineticFrictionIgnoresTheNormalVelocityComponent) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    // Purely normal motion (bouncing straight up/down), no sliding at all.
    body.momentum = ysq::Momentum3{Vec3{0.0, 5.0, 0.0}};

    const Vec3 normal{0.0, 1.0, 0.0};
    const Force3 friction = ysq::kineticFrictionForce(body, normal, Force{10.0}, 0.3);
    EXPECT_QUANTITY_VEC_NEAR(friction, Force3::zero(), Force{1e-12});
}

TEST(PhysicsMechanicsFriction, KineticFrictionUsesVelocityRelativeToTheSurface) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.momentum = ysq::Momentum3{Vec3{5.0, 0.0, 0.0}};

    const Vec3 normal{0.0, 1.0, 0.0};
    // Surface moving at the same tangential velocity as the body: no relative sliding.
    const Velocity3 surfaceVelocity{Vec3{5.0, 0.0, 0.0}};
    const Force3 friction =
        ysq::kineticFrictionForce(body, normal, Force{10.0}, 0.3, surfaceVelocity);
    EXPECT_QUANTITY_VEC_NEAR(friction, Force3::zero(), Force{1e-9});
}

TEST(PhysicsMechanicsFriction, StaticFrictionFullyCancelsADrivingForceWithinTheLimit) {
    const Vec3 normal{0.0, 1.0, 0.0};
    const Force3 drivingForce{Vec3{2.0, 0.0, 0.0}};  // well under mu_s * N = 0.5 * 10 = 5

    const Force3 friction =
        ysq::staticFrictionForce(normal, Force{10.0}, 0.5, drivingForce);
    EXPECT_QUANTITY_VEC_NEAR(friction, -drivingForce, Force{1e-9});
}

TEST(PhysicsMechanicsFriction, StaticFrictionCapsAtTheCoulombLimitWhenExceeded) {
    const Vec3 normal{0.0, 1.0, 0.0};
    const Force3 drivingForce{Vec3{20.0, 0.0, 0.0}};  // exceeds mu_s * N = 0.5 * 10 = 5

    const Force3 friction =
        ysq::staticFrictionForce(normal, Force{10.0}, 0.5, drivingForce);
    EXPECT_NEAR(ysq::length(friction).value(), 5.0, 1e-9);
    EXPECT_LT(friction.value().x, 0.0);  // still opposes the driving force
}

TEST(PhysicsMechanicsFriction, StaticFrictionIgnoresTheNormalComponentOfTheDrivingForce) {
    const Vec3 normal{0.0, 1.0, 0.0};
    // Purely normal driving force (e.g. gravity straight down on flat
    // ground): nothing tangential to resist.
    const Force3 drivingForce{Vec3{0.0, -10.0, 0.0}};

    const Force3 friction =
        ysq::staticFrictionForce(normal, Force{10.0}, 0.5, drivingForce);
    EXPECT_QUANTITY_VEC_NEAR(friction, Force3::zero(), Force{1e-12});
}

TEST(PhysicsMechanicsFriction, StaticFrictionIsZeroForNoDrivingForce) {
    const Vec3 normal{0.0, 1.0, 0.0};
    const Force3 friction =
        ysq::staticFrictionForce(normal, Force{10.0}, 0.5, Force3::zero());
    EXPECT_QUANTITY_VEC_NEAR(friction, Force3::zero(), Force{1e-12});
}
