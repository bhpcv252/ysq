#include <Applications/Helper/Pole.hpp>

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ysq::applications::poleRotation;
using ysq::radians;

ysq::Vec3 expectedPoleDirection(double rightAscension, double declination) {
    return ysq::Vec3{std::cos(declination) * std::cos(rightAscension),
                     std::cos(declination) * std::sin(rightAscension),
                     std::sin(declination)};
}

}  // namespace

TEST(ApplicationsHelperPole, RotatesLocalZToThePublishedDirectionForARangeOfPoles) {
    for (double raDeg = 0.0; raDeg < 360.0; raDeg += 37.0) {
        for (double decDeg = -80.0; decDeg <= 80.0; decDeg += 20.0) {
            const double ra = radians(raDeg);
            const double dec = radians(decDeg);
            const ysq::Vec3 rotated = rotate(poleRotation(ra, dec), ysq::Vec3::unitZ());
            const ysq::Vec3 expected = expectedPoleDirection(ra, dec);
            EXPECT_NEAR(rotated.x, expected.x, 1e-12) << "RA=" << raDeg << " Dec=" << decDeg;
            EXPECT_NEAR(rotated.y, expected.y, 1e-12) << "RA=" << raDeg << " Dec=" << decDeg;
            EXPECT_NEAR(rotated.z, expected.z, 1e-12) << "RA=" << raDeg << " Dec=" << decDeg;
        }
    }
}

TEST(ApplicationsHelperPole, NorthCelestialPoleIsAnIdentityRotationOnZ) {
    // RA is undefined at Dec = 90 deg but the rotation must still be
    // well-formed and leave +Z pointing at +Z regardless of which RA value
    // is passed in.
    for (double raDeg : {0.0, 90.0, 200.0}) {
        const ysq::Vec3 rotated =
            rotate(poleRotation(radians(raDeg), radians(90.0)), ysq::Vec3::unitZ());
        EXPECT_NEAR(rotated.x, 0.0, 1e-9);
        EXPECT_NEAR(rotated.y, 0.0, 1e-9);
        EXPECT_NEAR(rotated.z, 1.0, 1e-9);
    }
}

TEST(ApplicationsHelperPole, ZeroRightAscensionAndDeclinationPointsAlongPlusX) {
    const ysq::Vec3 rotated = rotate(poleRotation(0.0, 0.0), ysq::Vec3::unitZ());
    EXPECT_NEAR(rotated.x, 1.0, 1e-12);
    EXPECT_NEAR(rotated.y, 0.0, 1e-12);
    EXPECT_NEAR(rotated.z, 0.0, 1e-12);
}

TEST(ApplicationsHelperPole, NinetyDegreeRightAscensionAtZeroDeclinationPointsAlongPlusY) {
    const ysq::Vec3 rotated = rotate(poleRotation(radians(90.0), 0.0), ysq::Vec3::unitZ());
    EXPECT_NEAR(rotated.x, 0.0, 1e-12);
    EXPECT_NEAR(rotated.y, 1.0, 1e-12);
    EXPECT_NEAR(rotated.z, 0.0, 1e-12);
}

TEST(ApplicationsHelperPole, RotationIsOrthonormal) {
    // A pole rotation must not scale or shear the frame it's applied to:
    // it is meant to be composed with an orbital state vector, and any
    // distortion there would corrupt distances and speeds.
    const ysq::Quat q = poleRotation(radians(123.0), radians(-37.0));
    const ysq::Vec3 x = rotate(q, ysq::Vec3::unitX());
    const ysq::Vec3 y = rotate(q, ysq::Vec3::unitY());
    const ysq::Vec3 z = rotate(q, ysq::Vec3::unitZ());

    EXPECT_NEAR(length(x), 1.0, 1e-12);
    EXPECT_NEAR(length(y), 1.0, 1e-12);
    EXPECT_NEAR(length(z), 1.0, 1e-12);
    EXPECT_NEAR(dot(x, y), 0.0, 1e-12);
    EXPECT_NEAR(dot(y, z), 0.0, 1e-12);
    EXPECT_NEAR(dot(x, z), 0.0, 1e-12);
}
