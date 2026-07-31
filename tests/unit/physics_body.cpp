#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
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
using ysq::Vec3;

TEST(PhysicsBody, DefaultConstructedIsAllZero) {
    constexpr Body body{};
    EXPECT_EQ(body.mass, ysq::Mass::zero());
    EXPECT_EQ(body.position, ysq::Length3{});
    EXPECT_EQ(body.momentum, ysq::Momentum3{});
}

TEST(PhysicsBody, DefaultConstructedIsAPlainPointMass) {
    constexpr Body body{};
    EXPECT_EQ(body.radius, ysq::Length{});
    EXPECT_EQ(body.j2, 0.0);
    EXPECT_EQ(body.principalMomentsOfInertia, ysq::MomentOfInertia3{});
    EXPECT_EQ(body.orientation, ysq::Quat::identity());
    EXPECT_EQ(body.angularMomentum, ysq::AngularMomentum3{});
}

TEST(PhysicsBody, VelocityIsMomentumOverMass) {
    Body body{};
    body.mass = ysq::Mass{2.0};
    body.momentum = ysq::Momentum3{Vec3{4.0, 6.0, 8.0}};

    const ysq::Velocity3 velocity = body.velocity();
    EXPECT_VEC_APPROX(velocity.value(), (Vec3{2.0, 3.0, 4.0}));
}

}  // namespace
