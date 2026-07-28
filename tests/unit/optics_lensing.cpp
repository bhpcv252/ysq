#include <Math/Vector4.hpp>
#include <Physics/Optics/Lensing.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Units/Constants.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

TEST(OpticsLensing, ImpactParameterRayIsActuallyNull) {
    const ysq::Schwarzschild schwarzschild{ysq::GravitationalParameter{5.0e14}};
    const double rs = schwarzschild.schwarzschildRadius();
    const double startRadius = 1.0e4 * rs;
    const double impactParameter = 50.0 * rs;

    const ysq::PhaseState<ysq::Vec4> ray = ysq::schwarzschildRayFromImpactParameter(
        schwarzschild, impactParameter, startRadius);

    EXPECT_NEAR(
        ysq::metricProduct(schwarzschild, ray.position, ray.velocity, ray.velocity), 0.0,
        1e-3);
    // Moving inward.
    EXPECT_LT(ray.velocity.y, 0.0);
}

TEST(OpticsLensing, WeakFieldFormulaIsTwoSchwarzschildRadiiOverImpactParameter) {
    EXPECT_NEAR(ysq::weakFieldDeflectionAngle(10.0, 1000.0), 20.0 / 1000.0, 1e-15);
}

TEST(OpticsLensing, DeflectionShrinksAsImpactParameterGrows) {
    const ysq::GravitationalParameter gm{5.0e14};
    const ysq::Schwarzschild schwarzschild{gm};
    const double rs = schwarzschild.schwarzschildRadius();

    // startRadius scaled to each impact parameter, not shared, so "far from
    // the source" holds for both: dr/dlambda is order 1 in magnitude near
    // r0 either way, so one step size and step budget scaled to startRadius
    // alone covers the round trip (affine range of roughly 2 * startRadius).
    const auto deflectionAt = [&](double impactParameter) {
        const double startRadius = 20.0 * impactParameter;
        const double step = startRadius * 1.0e-3;
        const ysq::PhaseState<ysq::Vec4> ray = ysq::schwarzschildRayFromImpactParameter(
            schwarzschild, impactParameter, startRadius);
        return ysq::deflectionAngle(schwarzschild, ray, impactParameter, startRadius,
                                    step, 5000);
    };

    const double closeDeflection = deflectionAt(20.0 * rs);
    const double farDeflection = deflectionAt(200.0 * rs);

    ASSERT_FALSE(std::isnan(closeDeflection));
    ASSERT_FALSE(std::isnan(farDeflection));
    EXPECT_GT(closeDeflection, 0.0);
    EXPECT_GT(farDeflection, 0.0);
    EXPECT_GT(closeDeflection, farDeflection);
}

}  // namespace
