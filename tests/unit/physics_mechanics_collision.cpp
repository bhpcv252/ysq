#include <Math/Geometry/Primitives.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Mechanics/Collision.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Velocity.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::AABB3;
using ysq::Body;
using ysq::Contact;
using ysq::Length3;
using ysq::Vec3;

Body makeSphere(double x, double radius, double mass = 1.0, double velocityX = 0.0) {
    Body body{};
    body.mass = ysq::Mass{mass};
    body.radius = ysq::Length{radius};
    body.position = Length3{Vec3{x, 0.0, 0.0}};
    body.momentum = ysq::Momentum3{Vec3{velocityX * mass, 0.0, 0.0}};
    return body;
}

}  // namespace

TEST(PhysicsMechanicsCollision, OverlappingSpheresProduceAContact) {
    const Body a = makeSphere(0.0, 1.0);
    const Body b = makeSphere(1.5, 1.0);  // centers 1.5 apart, radii sum to 2

    const std::optional<Contact> contact = ysq::detectCollision(a, b);
    ASSERT_TRUE(contact.has_value());
    EXPECT_NEAR(contact->penetrationDepth.value(), 0.5, 1e-9);
    EXPECT_NEAR(contact->normal.x, 1.0, 1e-9);  // b is in the +x direction from a
}

TEST(PhysicsMechanicsCollision, SeparatedSpheresProduceNoContact) {
    const Body a = makeSphere(0.0, 1.0);
    const Body b = makeSphere(5.0, 1.0);

    EXPECT_FALSE(ysq::detectCollision(a, b).has_value());
}

TEST(PhysicsMechanicsCollision, ExactlyTouchingSpheresProduceNoContact) {
    // Touching exactly at distance == radius sum is the boundary, not an
    // overlap: a strict >= comparison in the detector treats it as a miss.
    const Body a = makeSphere(0.0, 1.0);
    const Body b = makeSphere(2.0, 1.0);

    EXPECT_FALSE(ysq::detectCollision(a, b).has_value());
}

TEST(PhysicsMechanicsCollision, SphereFarFromABoxProducesNoContact) {
    const Body sphere = makeSphere(0.0, 1.0);
    const AABB3<double> box{Vec3{2.0, -5.0, -5.0}, Vec3{5.0, 5.0, 5.0}};

    // Sphere at x=0 with radius 1 reaches x=1; the box starts at x=2, so
    // they do not overlap.
    EXPECT_FALSE(ysq::detectCollision(sphere, box).has_value());
}

TEST(PhysicsMechanicsCollision, SphereOverlappingABoxProducesAContact) {
    const Body sphere = makeSphere(1.5, 1.0);
    const AABB3<double> box{Vec3{2.0, -5.0, -5.0}, Vec3{5.0, 5.0, 5.0}};

    const std::optional<Contact> contact = ysq::detectCollision(sphere, box);
    ASSERT_TRUE(contact.has_value());
    EXPECT_NEAR(contact->penetrationDepth.value(), 0.5, 1e-9);
    EXPECT_NEAR(contact->normal.x, 1.0, 1e-9);  // the box is in the +x direction
}

TEST(PhysicsMechanicsCollision, TwoBodyElasticCollisionConservesMomentumAndEnergy) {
    Body a = makeSphere(0.0, 1.0, 1.0, 2.0);   // moving toward b at +2
    Body b = makeSphere(1.9, 1.0, 1.0, -2.0);  // moving toward a at -2, overlapping

    const std::optional<Contact> contact = ysq::detectCollision(a, b);
    ASSERT_TRUE(contact.has_value());

    const ysq::Momentum3 totalBefore = a.momentum + b.momentum;
    const double kineticBefore =
        0.5 * a.mass.value() * ysq::lengthSquared(a.velocity().value()) +
        0.5 * b.mass.value() * ysq::lengthSquared(b.velocity().value());

    ysq::resolveCollision(a, b, *contact, 1.0);

    const ysq::Momentum3 totalAfter = a.momentum + b.momentum;
    const double kineticAfter =
        0.5 * a.mass.value() * ysq::lengthSquared(a.velocity().value()) +
        0.5 * b.mass.value() * ysq::lengthSquared(b.velocity().value());

    EXPECT_QUANTITY_VEC_NEAR(totalAfter, totalBefore, ysq::Momentum{1e-9});
    EXPECT_NEAR(kineticAfter, kineticBefore, 1e-9);

    // Equal masses, perfectly elastic, head-on: velocities exchange exactly.
    EXPECT_NEAR(a.velocity().value().x, -2.0, 1e-9);
    EXPECT_NEAR(b.velocity().value().x, 2.0, 1e-9);
}

TEST(PhysicsMechanicsCollision,
     TwoBodyPerfectlyInelasticCollisionEndsWithASharedVelocity) {
    Body a = makeSphere(0.0, 1.0, 1.0, 2.0);
    Body b = makeSphere(1.9, 1.0, 3.0, 0.0);

    const std::optional<Contact> contact = ysq::detectCollision(a, b);
    ASSERT_TRUE(contact.has_value());

    ysq::resolveCollision(a, b, *contact, 0.0);

    // With restitution 0, both bodies end up with the same velocity along
    // the normal, matching a perfectly inelastic collision.
    EXPECT_NEAR(a.velocity().value().x, b.velocity().value().x, 1e-9);
}

TEST(PhysicsMechanicsCollision, TwoBodyResolutionIsANoOpWhenAlreadySeparating) {
    Body a = makeSphere(0.0, 1.0, 1.0, -1.0);  // moving away from b
    Body b = makeSphere(1.9, 1.0, 1.0, 1.0);   // moving away from a

    const std::optional<Contact> contact = ysq::detectCollision(a, b);
    ASSERT_TRUE(contact.has_value());

    const ysq::Momentum3 momentumABefore = a.momentum;
    const ysq::Momentum3 momentumBBefore = b.momentum;
    ysq::resolveCollision(a, b, *contact, 1.0);

    EXPECT_QUANTITY_VEC_NEAR(a.momentum, momentumABefore, ysq::Momentum{1e-12});
    EXPECT_QUANTITY_VEC_NEAR(b.momentum, momentumBBefore, ysq::Momentum{1e-12});
}

TEST(PhysicsMechanicsCollision, SingleBodyElasticBounceReversesTheNormalVelocityExactly) {
    Body body = makeSphere(1.5, 1.0, 2.0, 1.0);  // moving toward the box at +x
    const AABB3<double> box{Vec3{2.0, -5.0, -5.0}, Vec3{5.0, 5.0, 5.0}};

    const std::optional<Contact> contact = ysq::detectCollision(body, box);
    ASSERT_TRUE(contact.has_value());

    ysq::resolveCollision(body, *contact, 1.0);
    EXPECT_NEAR(body.velocity().value().x, -1.0, 1e-9);
}

TEST(PhysicsMechanicsCollision, SingleBodyInelasticBounceStopsTheNormalVelocity) {
    Body body = makeSphere(1.5, 1.0, 2.0, 1.0);
    const AABB3<double> box{Vec3{2.0, -5.0, -5.0}, Vec3{5.0, 5.0, 5.0}};

    const std::optional<Contact> contact = ysq::detectCollision(body, box);
    ASSERT_TRUE(contact.has_value());

    ysq::resolveCollision(body, *contact, 0.0);
    EXPECT_NEAR(body.velocity().value().x, 0.0, 1e-9);
}

TEST(PhysicsMechanicsCollision, SingleBodyResolutionIsANoOpWhenAlreadyMovingAway) {
    Body body = makeSphere(1.5, 1.0, 2.0, -1.0);  // moving away from the box
    const AABB3<double> box{Vec3{2.0, -5.0, -5.0}, Vec3{5.0, 5.0, 5.0}};

    const std::optional<Contact> contact = ysq::detectCollision(body, box);
    ASSERT_TRUE(contact.has_value());

    const ysq::Momentum3 before = body.momentum;
    ysq::resolveCollision(body, *contact, 1.0);
    EXPECT_QUANTITY_VEC_NEAR(body.momentum, before, ysq::Momentum{1e-12});
}
