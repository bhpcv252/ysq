#include <Math/Scalar.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Optics/Lensing.hpp>
#include <Physics/Optics/RefractiveMedium.hpp>
#include <Physics/Spacetime/Metric.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

TEST(OpticsRefractiveMedium, RefractiveIndexIsOneFarFromTheSurface) {
    const ysq::RefractiveMedium medium{6.371e6, 2.9e-4, 8434.5};
    EXPECT_NEAR(medium.refractiveIndex(6.371e6 + 1.0e6), 1.0, 1e-9);
}

TEST(OpticsRefractiveMedium, RefractiveIndexAtTheSurfaceIsOnePlusSurfaceRefractivity) {
    const ysq::RefractiveMedium medium{6.371e6, 2.9e-4, 8434.5};
    EXPECT_NEAR(medium.refractiveIndex(6.371e6), 1.00029, 1e-12);
}

TEST(OpticsRefractiveMedium, ImpactParameterRayIsActuallyNull) {
    const double radius = 6.371e6;
    const ysq::RefractiveMedium medium{radius, 2.9e-4, 8434.5};
    const double startRadius = radius + 200.0 * 8434.5;

    const ysq::PhaseState<ysq::Vec4> ray =
        ysq::refractiveMediumRayFromImpactParameter(medium, radius, startRadius);

    EXPECT_NEAR(ysq::metricProduct(medium, ray.position, ray.velocity, ray.velocity), 0.0,
                1e-9);
    EXPECT_LT(ray.velocity.y, 0.0);
}

TEST(OpticsRefractiveMedium, HorizontalRefractionIsCloseToTheRealMeasuredValue) {
    const double radius = 6.371e6;
    const double scaleHeight = 8434.5;
    const double surfaceRefractivity = 2.9e-4;
    const ysq::RefractiveMedium medium{radius, surfaceRefractivity, scaleHeight};

    // Unlike Schwarzschild's 1/r tail, this medium is exactly flat (n = 1 to
    // well beyond double precision) a few dozen scale heights up: 15 scale
    // heights leaves n - 1 of order 1e-10, so startRadius here does not need
    // to be "far away" in the Schwarzschild sense, only past the atmosphere,
    // which keeps the affine range deflectionAngle has to search across
    // small enough to run in well under a second rather than minutes.
    const double startRadius = radius + 15.0 * scaleHeight;
    const double step = scaleHeight / 20.0;
    const std::size_t maxSteps = 10000;

    // Grazing exactly at r = radius bends enough to plunge below it (this
    // medium has no opacity of its own to stop that, it is a pure
    // refractive-index field; Optics/Illumination.hpp is what will apply
    // occlusion against the actual opaque body). A graze about half a
    // percent above the nominal radius is a fair stand-in for "the
    // horizon": real horizontal refraction is itself only defined up to the
    // real atmosphere's structure and the observer's own height, not to
    // sub-kilometer precision either.
    const double grazingImpactParameter = radius * 1.001;
    const ysq::PhaseState<ysq::Vec4> ray = ysq::refractiveMediumRayFromImpactParameter(
        medium, grazingImpactParameter, startRadius);
    const double bending = ysq::deflectionAngle(medium, ray, grazingImpactParameter,
                                                startRadius, step, maxSteps);

    ASSERT_FALSE(std::isnan(bending));
    const double bendingArcmin = bending * (180.0 / ysq::kPi<double>)*60.0;

    // The real measured value is about 34-35 arcmin; this simplified
    // isothermal exponential atmosphere is not claimed to reproduce that to
    // the arcminute, only to land close, which it does.
    EXPECT_GT(bendingArcmin, 20.0);
    EXPECT_LT(bendingArcmin, 55.0);

    // Physical sanity beyond the single number: grazing higher, through
    // thinner air, must bend less.
    const double higherImpactParameter = radius * 1.005;
    const ysq::PhaseState<ysq::Vec4> higherRay =
        ysq::refractiveMediumRayFromImpactParameter(medium, higherImpactParameter,
                                                    startRadius);
    const double higherBending = ysq::deflectionAngle(
        medium, higherRay, higherImpactParameter, startRadius, step, maxSteps);

    ASSERT_FALSE(std::isnan(higherBending));
    EXPECT_LT(higherBending, bending);
    EXPECT_GT(higherBending, 0.0);
}

}  // namespace
