#include <Math/Vector3.hpp>
#include <Physics/Optics/Illumination.hpp>
#include <Physics/Optics/RefractiveMedium.hpp>

#include <gtest/gtest.h>

#include <array>

namespace {

using ysq::OpaqueOccluder;
using ysq::RefractingOccluder;
using ysq::Vec3;

// Earth-like occluder, sitting between a Sun-like source and a Moon-like
// target, with a real-scale atmosphere.
RefractingOccluder makeEarthLikeOccluder() {
    RefractingOccluder occluder{};
    occluder.center = Vec3{0.0, 0.0, 0.0};
    occluder.opaqueRadius = 6.371e6;
    occluder.medium = ysq::RefractiveMedium{6.371e6, 2.9e-4, 8434.5};
    occluder.surfaceNumberDensity = 2.6868e25;
    occluder.scatteringScaleHeight = 8434.5;
    return occluder;
}

const std::array<double, 3> kRgbWavelengths{630.0e-9, 532.0e-9, 465.0e-9};

TEST(OpticsIllumination, ClearLineOfSightIsFullTransmission) {
    const Vec3 sourceCenter{1.496e11, 0.0, 0.0};
    const double sourceRadius = 6.957e8;
    const Vec3 target{-3.844e8, 5.0e8, 0.0};  // well off to the side, not shadowed

    const ysq::IlluminationResult result = ysq::illuminate(
        sourceCenter, sourceRadius, {}, nullptr, target, kRgbWavelengths, 4, 800);

    EXPECT_NEAR(result.transmission.x, 1.0, 1e-9);
    EXPECT_NEAR(result.transmission.y, 1.0, 1e-9);
    EXPECT_NEAR(result.transmission.z, 1.0, 1e-9);
    EXPECT_NEAR(result.geometricVisibility, 1.0, 1e-9);
}

TEST(OpticsIllumination, BlockedByAnUnrelatedOpaqueBodyIsFullShadow) {
    const Vec3 sourceCenter{1.496e11, 0.0, 0.0};
    const double sourceRadius = 6.957e8;
    const Vec3 target{-1.0e9, 0.0, 0.0};

    const OpaqueOccluder blocker{Vec3{0.0, 0.0, 0.0}, 5.0e8};
    const std::array<OpaqueOccluder, 1> occluders{blocker};

    const ysq::IlluminationResult result = ysq::illuminate(
        sourceCenter, sourceRadius, occluders, nullptr, target, kRgbWavelengths, 4, 800);

    EXPECT_NEAR(result.transmission.x, 0.0, 1e-9);
    EXPECT_NEAR(result.transmission.y, 0.0, 1e-9);
    EXPECT_NEAR(result.transmission.z, 0.0, 1e-9);
    EXPECT_NEAR(result.geometricVisibility, 0.0, 1e-9);
}

TEST(OpticsIllumination, DeepInAnAtmosphericOccludersShadowSomeReddenedLightGetsThrough) {
    const Vec3 sourceCenter{1.496e11, 0.0, 0.0};  // Sun-like, 1 AU away
    const double sourceRadius = 6.957e8;
    const RefractingOccluder earth = makeEarthLikeOccluder();
    const Vec3 target{-3.844e8, 0.0, 0.0};  // Moon-like, directly opposite the source

    const ysq::IlluminationResult result = ysq::illuminate(
        sourceCenter, sourceRadius, {}, &earth, target, kRgbWavelengths, 4, 800);

    // Not fully dark: some light gets through via the bent path.
    EXPECT_GT(result.transmission.x + result.transmission.y + result.transmission.z, 0.0);
    // Not full daylight either: this is deep in the geometric shadow.
    EXPECT_LT(result.transmission.x, 1.0);
    EXPECT_NEAR(result.geometricVisibility, 0.0, 1e-9);
    // Reddened: red's cross-section is far smaller than blue's, so far more
    // red survives the same grazing path.
    EXPECT_GT(result.transmission.x, result.transmission.z);
}

}  // namespace
