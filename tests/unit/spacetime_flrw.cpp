#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Spacetime/FLRW.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ysq::Vec4;

constexpr double kPi = ysq::kPi<double>;

TEST(SpacetimeFLRW, MatterDominatedScaleFactorIsOneAtTheReferenceTime) {
    const ysq::MatterDominatedFLRW cosmology{1.0e17, 0.0};
    const ysq::MetricTensor<double> g = cosmology.components(Vec4{1.0e17, 1.0, 1.0, 0.0});
    // Flat (k = 0): a(T0)^2 = 1 makes g_rr exactly 1, same as Minkowski's
    // spatial part.
    EXPECT_NEAR(g(1, 1), 1.0, 1e-9);
}

TEST(SpacetimeFLRW, MatterDominatedScaleFactorGrowsAsTToTheTwoThirds) {
    const double referenceTime = 1.0e17;
    const ysq::MatterDominatedFLRW cosmology{referenceTime, 0.0};

    const double laterTime = 8.0 * referenceTime;
    const ysq::MetricTensor<double> g =
        cosmology.components(Vec4{laterTime, 1.0, kPi / 2.0, 0.0});

    // g_theta_theta = a^2 r^2, r = 1, so this is a^2 directly:
    // a = 8^(2/3) = (8^(1/3))^2 = 4, a^2 = 16.
    EXPECT_NEAR(g(2, 2), 16.0, 1e-6);
}

TEST(SpacetimeFLRW, RadiationDominatedScaleFactorGrowsAsTToTheOneHalf) {
    const double referenceTime = 1.0e17;
    const ysq::RadiationDominatedFLRW cosmology{referenceTime, 0.0};

    const double laterTime = 4.0 * referenceTime;
    const ysq::MetricTensor<double> g =
        cosmology.components(Vec4{laterTime, 1.0, kPi / 2.0, 0.0});

    EXPECT_NEAR(g(2, 2), 4.0, 1e-6);  // a = sqrt(4) = 2, a^2 = 4
}

TEST(SpacetimeFLRW, LambdaDominatedScaleFactorGrowsExponentially) {
    const double hubbleRate = 2.0e-18;
    const double referenceTime = 0.0;
    const ysq::LambdaDominatedFLRW cosmology{hubbleRate, referenceTime, 0.0};

    const double doublingTime = std::log(2.0) / hubbleRate;
    const ysq::MetricTensor<double> g =
        cosmology.components(Vec4{doublingTime, 1.0, kPi / 2.0, 0.0});

    EXPECT_NEAR(g(2, 2), 4.0, 1e-6);  // a doubled, a^2 quadruples
}

TEST(SpacetimeFLRW, PositiveCurvatureShrinksTheRadialComponent) {
    const ysq::MatterDominatedFLRW flat{1.0e17, 0.0};
    const ysq::MatterDominatedFLRW closed{1.0e17, 0.5};

    const Vec4 at{1.0e17, 0.5, kPi / 2.0, 0.0};
    const double flatGrr = flat.components(at)(1, 1);
    const double closedGrr = closed.components(at)(1, 1);

    // g_rr = a^2 / (1 - k r^2): positive k makes the denominator smaller
    // than 1, so g_rr grows relative to the flat case at the same r.
    EXPECT_GT(closedGrr, flatGrr);
}

}  // namespace
