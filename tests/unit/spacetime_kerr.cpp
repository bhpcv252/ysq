#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Spacetime/Kerr.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Units/Constants.hpp>
#include <Units/Length.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cstddef>

namespace {

using ysq::Vec4;

constexpr double kPi = ysq::kPi<double>;

TEST(SpacetimeKerr, AccessorsMatchTheConstructorArguments) {
    const ysq::GravitationalParameter gm{6.0e14};
    const ysq::Length spin{1.0e5};
    const ysq::Kerr kerr{gm, spin};

    const ysq::Schwarzschild schwarzschild{gm};
    EXPECT_NEAR(kerr.schwarzschildRadius(), schwarzschild.schwarzschildRadius(),
                schwarzschild.schwarzschildRadius() * 1e-12);
    EXPECT_NEAR(kerr.spin(), spin.value(), spin.value() * 1e-12);
}

TEST(SpacetimeKerr, ReducesExactlyToSchwarzschildAtZeroSpin) {
    const ysq::GravitationalParameter gm{6.0e14};
    const ysq::Kerr kerr{gm, ysq::Length::zero()};
    const ysq::Schwarzschild schwarzschild{gm};

    for (const double r : {2.0e6, 5.0e6, 5.0e8}) {
        for (const double polar : {0.3, kPi / 2.0, 2.5}) {
            const Vec4 at{0.0, r, polar, 0.9};
            const ysq::MetricTensor<double> gKerr = kerr.components(at);
            const ysq::MetricTensor<double> gSchwarzschild = schwarzschild.components(at);

            for (std::size_t flat = 0; flat < gKerr.size(); ++flat) {
                EXPECT_NEAR(gKerr[flat], gSchwarzschild[flat],
                            std::abs(gSchwarzschild[flat]) * 1e-9 + 1e-9)
                    << "r=" << r << " polar=" << polar << " component=" << flat;
            }
        }
    }
}

TEST(SpacetimeKerr, HasAFrameDraggingCrossTermWhenSpinning) {
    const ysq::GravitationalParameter gm{6.0e14};
    const ysq::Kerr kerr{gm, ysq::Length{1.0e5}};

    const ysq::MetricTensor<double> g = kerr.components(Vec4{0.0, 5.0e6, kPi / 2.0, 0.0});
    EXPECT_NE(g(0, 3), 0.0);
    EXPECT_NEAR(g(0, 3), g(3, 0), 1e-30);
}

TEST(SpacetimeKerr, ChristoffelSymbolsAreSymmetricInTheLowerIndices) {
    const ysq::Kerr kerr{ysq::GravitationalParameter{6.0e14}, ysq::Length{1.0e5}};
    const Vec4 at{0.0, 5.0e6, 1.1, 0.4};
    const ysq::ChristoffelSymbols<double> gamma = ysq::christoffelSymbols(kerr, at);

    for (std::size_t lambda = 0; lambda < 4; ++lambda) {
        for (std::size_t mu = 0; mu < 4; ++mu) {
            for (std::size_t nu = 0; nu < 4; ++nu) {
                EXPECT_NEAR(gamma(lambda, mu, nu), gamma(lambda, nu, mu), 1e-6);
            }
        }
    }
}

}  // namespace
