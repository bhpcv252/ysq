#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Units/Constants.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

using ysq::Vec4;

constexpr double kPi = ysq::kPi<double>;

TEST(SpacetimeSchwarzschild, SchwarzschildRadiusIsTwoGMOverCSquared) {
    const ysq::GravitationalParameter gm{4.0e14};
    const ysq::Schwarzschild schwarzschild{gm};

    const double c = ysq::constants::speedOfLight.value();
    EXPECT_NEAR(schwarzschild.schwarzschildRadius(), 2.0 * gm.value() / (c * c),
                schwarzschild.schwarzschildRadius() * 1e-12);
}

TEST(SpacetimeSchwarzschild, ComponentsMatchTheClosedForm) {
    const ysq::GravitationalParameter gm{4.0e14};
    const ysq::Schwarzschild schwarzschild{gm};
    const double rs = schwarzschild.schwarzschildRadius();

    const double r = 10.0 * rs;
    const double polar = kPi / 3.0;
    const ysq::MetricTensor<double> g =
        schwarzschild.components(Vec4{0.0, r, polar, 0.7});

    const double factor = 1.0 - rs / r;
    EXPECT_NEAR(g(0, 0), -factor, 1e-12);
    EXPECT_NEAR(g(1, 1), 1.0 / factor, 1e-12);
    EXPECT_NEAR(g(2, 2), r * r, r * r * 1e-12);
    EXPECT_NEAR(g(3, 3), r * r * std::sin(polar) * std::sin(polar), r * r * 1e-12);

    // Off-diagonal terms are all zero: spherical symmetry, no rotation.
    for (std::size_t mu = 0; mu < 4; ++mu) {
        for (std::size_t nu = 0; nu < 4; ++nu) {
            if (mu != nu) {
                EXPECT_APPROX(g(mu, nu), 0.0);
            }
        }
    }
}

TEST(SpacetimeSchwarzschild, ReducesToMinkowskiFarFromTheSource) {
    const ysq::GravitationalParameter gm{1.0e14};
    const ysq::Schwarzschild schwarzschild{gm};
    const double rs = schwarzschild.schwarzschildRadius();

    const double r = 1.0e9 * rs;
    const ysq::MetricTensor<double> g =
        schwarzschild.components(Vec4{0.0, r, kPi / 2.0, 0.0});

    EXPECT_NEAR(g(0, 0), -1.0, 1e-6);
    EXPECT_NEAR(g(1, 1), 1.0, 1e-6);
}

TEST(SpacetimeSchwarzschild, RadialChristoffelMatchesTheDirectlyDerivedClosedForm) {
    // Gamma^r_TT = -(1/2) g^rr d_r g_TT, since g is diagonal and g_TT does
    // not depend on T. With g_TT = -(1 - r_s/r) and g^rr = (1 - r_s/r), that
    // works out to (1/2) r_s (1 - r_s/r) / r^2 exactly, not just in the weak
    // field; src/Physics/README.md has the full derivation, and this is what
    // reduces to Newtonian gravity, GM/r^2, as r_s/r -> 0.
    const ysq::GravitationalParameter gm{5.0e14};
    const ysq::Schwarzschild schwarzschild{gm};
    const double rs = schwarzschild.schwarzschildRadius();

    for (const double rOverRs : {3.0, 10.0, 1.0e6}) {
        const double r = rOverRs * rs;
        const ysq::ChristoffelSymbols<double> gamma =
            ysq::christoffelSymbols(schwarzschild, Vec4{0.0, r, kPi / 2.0, 0.0});

        const double expected = 0.5 * rs * (1.0 - rs / r) / (r * r);
        EXPECT_NEAR(gamma(1, 0, 0), expected, std::abs(expected) * 1e-6)
            << "r = " << rOverRs << " * r_s";
    }
}

}  // namespace
