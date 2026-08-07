#include <Math/Vector3.hpp>
#include <Physics/Optics/Illumination.hpp>
#include <Physics/Optics/RefractiveMedium.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

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

TEST(OpticsIllumination, DiscOcclusionFractionIsFullyLitWhenTheOccluderDoesNotOverlapTheSource) {
    const Vec3 point{0.0, 0.0, 0.0};
    const Vec3 sourceCenter{100.0, 0.0, 0.0};
    const double sourceRadius = 10.0;
    // Off to the side far enough that the two discs, as seen from point,
    // do not overlap at all.
    const Vec3 occluderCenter{10.0, 5.0, 0.0};
    const double occluderRadius = 0.3;

    EXPECT_NEAR(ysq::discOcclusionFraction(point, sourceCenter, sourceRadius, occluderCenter,
                                    occluderRadius),
               1.0, 1e-9);
}

TEST(OpticsIllumination, DiscOcclusionFractionIsFullyLitWhenTheOccluderIsFartherThanTheSource) {
    const Vec3 point{0.0, 0.0, 0.0};
    const Vec3 sourceCenter{10.0, 0.0, 0.0};
    // Directly behind the source, from point's own perspective: cannot be
    // sitting between them.
    const Vec3 occluderCenter{100.0, 0.0, 0.0};

    EXPECT_NEAR(ysq::discOcclusionFraction(point, sourceCenter, 10.0, occluderCenter, 50.0), 1.0,
               1e-9);
}

TEST(OpticsIllumination, DiscOcclusionFractionIsZeroForATotalEclipse) {
    const Vec3 point{0.0, 0.0, 0.0};
    const Vec3 sourceCenter{100.0, 0.0, 0.0};
    const double sourceRadius = 10.0;
    // Directly in front, angularly bigger than the source: total eclipse.
    const Vec3 occluderCenter{10.0, 0.0, 0.0};
    const double occluderRadius = 2.0;

    EXPECT_NEAR(ysq::discOcclusionFraction(point, sourceCenter, sourceRadius, occluderCenter,
                                    occluderRadius),
               0.0, 1e-9);
}

TEST(OpticsIllumination, DiscOcclusionFractionMatchesTheClosedFormForAnAnnularEclipse) {
    // Directly in front (separation = 0), angularly smaller than the
    // source: an annular eclipse, blocking only the fraction of the
    // source's own disc area the occluder's own angular disc covers.
    const Vec3 point{0.0, 0.0, 0.0};
    const Vec3 sourceCenter{100.0, 0.0, 0.0};
    const double sourceRadius = 10.0;
    const Vec3 occluderCenter{10.0, 0.0, 0.0};
    const double occluderRadius = 0.5;

    const double alphaSource = std::atan(sourceRadius / 100.0);
    const double alphaOccluder = std::atan(occluderRadius / 10.0);
    const double expected =
        1.0 - (alphaOccluder * alphaOccluder) / (alphaSource * alphaSource);

    EXPECT_NEAR(ysq::discOcclusionFraction(point, sourceCenter, sourceRadius, occluderCenter,
                                    occluderRadius),
               expected, 1e-9);
    // Same case computed independently ahead of time, so this pins the
    // actual number down, not just internal self-consistency with the
    // formula it is supposed to be checking.
    EXPECT_NEAR(ysq::discOcclusionFraction(point, sourceCenter, sourceRadius, occluderCenter,
                                    occluderRadius),
               0.7487536309827298, 1e-9);
}

TEST(OpticsIllumination, DiscOcclusionFractionMatchesAnIndependentlyComputedPartialCase) {
    const Vec3 point{0.0, 0.0, 0.0};
    const Vec3 sourceCenter{100.0, 0.0, 0.0};
    const double sourceRadius = 10.0;
    // Offset so the occluder's own angular radius equals the source's own,
    // and the angular separation between them equals that same angle too
    // -- partial overlap by construction, not edge-on or total.
    const Vec3 occluderCenter{10.0, 1.0000000000000002, 0.0};
    const double occluderRadius = 1.0000000000000002;

    const double result = ysq::discOcclusionFraction(point, sourceCenter, sourceRadius,
                                               occluderCenter, occluderRadius);
    EXPECT_GT(result, 0.0);
    EXPECT_LT(result, 1.0);
    EXPECT_NEAR(result, 0.6122809621777536, 1e-9);
}

TEST(OpticsIllumination, DiscOcclusionFractionIsNearZeroForARealTotalLunarEclipse) {
    // Real geometry: Sun, Earth's own real opaque radius as the occluder,
    // and the Moon directly opposite the Sun at its own real distance --
    // Earth's real umbra is wide enough there (about 9200 km across
    // against the Moon's own ~3474 km) to cover the Moon entirely, a real
    // total lunar eclipse, not a contrived number.
    const Vec3 point{-3.844e8, 0.0, 0.0};       // the Moon
    const Vec3 sourceCenter{1.496e11, 0.0, 0.0};  // the Sun
    const double sourceRadius = 6.957e8;
    const Vec3 occluderCenter{0.0, 0.0, 0.0};  // the Earth
    const double occluderRadius = 6.371e6;

    EXPECT_NEAR(
        ysq::discOcclusionFraction(point, sourceCenter, sourceRadius, occluderCenter, occluderRadius),
        0.0, 1e-6);
}

}  // namespace
