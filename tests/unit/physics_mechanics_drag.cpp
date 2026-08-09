#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Mechanics/Drag.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Area;
using ysq::Body;
using ysq::Density;
using ysq::Force;
using ysq::Force3;
using ysq::LinearDragCoefficient;
using ysq::Vec3;
using ysq::Velocity3;

}  // namespace

TEST(PhysicsMechanicsDrag, LinearDragOpposesVelocity) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.momentum = ysq::Momentum3{Vec3{3.0, 0.0, 0.0}};

    const Force3 force = ysq::linearDragForce(body, LinearDragCoefficient{2.0});
    EXPECT_QUANTITY_VEC_NEAR(force, (Force3{Vec3{-6.0, 0.0, 0.0}}), Force{1e-9});
}

TEST(PhysicsMechanicsDrag, LinearDragIsZeroAtRest) {
    Body body{};
    body.mass = ysq::Mass{1.0};

    const Force3 force = ysq::linearDragForce(body, LinearDragCoefficient{2.0});
    EXPECT_QUANTITY_VEC_NEAR(force, Force3::zero(), Force{1e-12});
}

TEST(PhysicsMechanicsDrag, LinearDragUsesVelocityRelativeToTheMedium) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.momentum = ysq::Momentum3{Vec3{5.0, 0.0, 0.0}};

    // Moving at the same velocity as the medium: no relative motion.
    const Velocity3 mediumVelocity{Vec3{5.0, 0.0, 0.0}};
    const Force3 force =
        ysq::linearDragForce(body, LinearDragCoefficient{2.0}, mediumVelocity);
    EXPECT_QUANTITY_VEC_NEAR(force, Force3::zero(), Force{1e-9});
}

TEST(PhysicsMechanicsDrag, QuadraticDragScalesWithTheSquareOfSpeed) {
    Body slow{};
    slow.mass = ysq::Mass{1.0};
    slow.momentum = ysq::Momentum3{Vec3{2.0, 0.0, 0.0}};

    Body fast{};
    fast.mass = ysq::Mass{1.0};
    fast.momentum = ysq::Momentum3{Vec3{4.0, 0.0, 0.0}};  // twice the speed

    const Density density{1.2};
    const double dragCoefficient = 0.5;
    const Area area{0.1};

    const Force3 slowForce =
        ysq::quadraticDragForce(slow, density, dragCoefficient, area);
    const Force3 fastForce =
        ysq::quadraticDragForce(fast, density, dragCoefficient, area);

    // Doubling the speed should quadruple the magnitude.
    EXPECT_NEAR(ysq::length(fastForce).value(), 4.0 * ysq::length(slowForce).value(),
                1e-9);
}

TEST(PhysicsMechanicsDrag, QuadraticDragOpposesTheDirectionOfMotion) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.momentum = ysq::Momentum3{Vec3{0.0, 3.0, 0.0}};

    const Force3 force = ysq::quadraticDragForce(body, Density{1.0}, 1.0, Area{1.0});
    EXPECT_LT(force.value().y, 0.0);
    EXPECT_NEAR(force.value().x, 0.0, 1e-12);
    EXPECT_NEAR(force.value().z, 0.0, 1e-12);
}

TEST(PhysicsMechanicsDrag, QuadraticDragIsZeroAtRest) {
    Body body{};
    body.mass = ysq::Mass{1.0};

    const Force3 force = ysq::quadraticDragForce(body, Density{1.2}, 0.5, Area{0.1});
    EXPECT_QUANTITY_VEC_NEAR(force, Force3::zero(), Force{1e-12});
}

TEST(PhysicsMechanicsDrag, QuadraticDragMatchesTheClosedFormMagnitude) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.momentum = ysq::Momentum3{Vec3{10.0, 0.0, 0.0}};

    const Density density{1.225};         // sea-level air
    const double dragCoefficient = 0.47;  // a sphere
    const Area area{0.05};

    const Force3 force = ysq::quadraticDragForce(body, density, dragCoefficient, area);
    const double expectedMagnitude =
        0.5 * density.value() * dragCoefficient * area.value() * 10.0 * 10.0;
    EXPECT_NEAR(ysq::length(force).value(), expectedMagnitude, 1e-9);
}
