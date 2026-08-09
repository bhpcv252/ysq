#include <Math/Vector3.hpp>
#include <Physics/Optics/Aberration.hpp>
#include <Units/Constants.hpp>
#include <Units/Velocity.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ysq::Speed;
using ysq::Vec3;
using ysq::Velocity3;

}  // namespace

TEST(PhysicsOpticsAberration, ZeroBoostLeavesTheDirectionUnchanged) {
    const Vec3 direction{0.6, 0.8, 0.0};
    const Vec3 result = ysq::aberratedDirection(direction, Velocity3::zero());

    EXPECT_NEAR(result.x, direction.x, 1e-9);
    EXPECT_NEAR(result.y, direction.y, 1e-9);
    EXPECT_NEAR(result.z, direction.z, 1e-9);
}

TEST(PhysicsOpticsAberration, ResultIsAlwaysAUnitVector) {
    const Vec3 direction{0.3, 0.4, std::sqrt(1.0 - 0.09 - 0.16)};
    const Velocity3 boost{Vec3{0.5 * ysq::constants::speedOfLight.value(), 0.0, 0.0}};

    const Vec3 result = ysq::aberratedDirection(direction, boost);
    EXPECT_NEAR(ysq::length(result), 1.0, 1e-9);
}

TEST(PhysicsOpticsAberration, VectorFormMatchesTheClosedFormAlongTheBoostAxis) {
    const Speed boostSpeed{0.6 * ysq::constants::speedOfLight.value()};
    const Velocity3 boost{Vec3{boostSpeed.value(), 0.0, 0.0}};

    // Forward-pointing photon (cos(theta) = 1): aberration cannot change
    // "straight ahead" into anything else.
    const Vec3 forward = ysq::aberratedDirection(Vec3{1.0, 0.0, 0.0}, boost);
    EXPECT_NEAR(forward.x, ysq::aberratedCosine(1.0, boostSpeed), 1e-9);

    // Backward-pointing photon (cos(theta) = -1): same, "straight behind".
    const Vec3 backward = ysq::aberratedDirection(Vec3{-1.0, 0.0, 0.0}, boost);
    EXPECT_NEAR(backward.x, ysq::aberratedCosine(-1.0, boostSpeed), 1e-9);
}

TEST(PhysicsOpticsAberration, VectorFormsAxialComponentMatchesTheClosedFormOffAxis) {
    const Speed boostSpeed{0.7 * ysq::constants::speedOfLight.value()};
    const Velocity3 boost{Vec3{boostSpeed.value(), 0.0, 0.0}};

    // A direction at 60 degrees to the boost axis: cos(theta) = 0.5.
    const Vec3 direction{0.5, std::sqrt(0.75), 0.0};
    const Vec3 result = ysq::aberratedDirection(direction, boost);

    EXPECT_NEAR(result.x, ysq::aberratedCosine(0.5, boostSpeed), 1e-9);
}

TEST(PhysicsOpticsAberration,
     RelativisticBeamingTiltsAPerpendicularRaysTravelDirectionBackward) {
    // aberratedCosine tracks a photon's own travel direction, not the
    // apparent direction to its source (the two are opposite). A photon
    // traveling perpendicular to the boost in the original frame
    // (cos(theta) = 0) tips its travel direction toward the frame's own
    // "aft" (cos(theta') < 0) in a frame moving fast relative to the
    // original one -- exactly why an isotropic field's apparent *sources*
    // bunch toward "forward" instead (the headlight effect, and the
    // physical origin of the CMB dipole): a source is the negation of the
    // direction its light travels.
    const Speed fastBoost{0.9 * ysq::constants::speedOfLight.value()};
    EXPECT_LT(ysq::aberratedCosine(0.0, fastBoost), 0.0);
}

TEST(PhysicsOpticsAberration, NoBoostLeavesTheClosedFormAngleUnchanged) {
    for (double cosTheta : {-1.0, -0.3, 0.0, 0.4, 1.0}) {
        EXPECT_NEAR(ysq::aberratedCosine(cosTheta, Speed{0.0}), cosTheta, 1e-12);
    }
}
