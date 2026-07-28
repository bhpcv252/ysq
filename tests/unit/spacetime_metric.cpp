#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Minkowski.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Units/Constants.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cstddef>

namespace {

using ysq::Vec4;

// --- Minkowski, the base case -----------------------------------------------

TEST(SpacetimeMetric, MinkowskiComponentsAreTheFlatDiagonalEverywhere) {
    const ysq::Minkowski flat;
    for (const Vec4 at : {Vec4{0.0, 0.0, 0.0, 0.0}, Vec4{5.0, -3.0, 2.0, 1.0}}) {
        const ysq::MetricTensor<double> g = flat.components(at);
        for (std::size_t mu = 0; mu < 4; ++mu) {
            for (std::size_t nu = 0; nu < 4; ++nu) {
                const double expected = (mu != nu) ? 0.0 : ((mu == 0) ? -1.0 : 1.0);
                EXPECT_APPROX(g(mu, nu), expected);
            }
        }
    }
}

TEST(SpacetimeMetric, MinkowskiChristoffelSymbolsAreExactlyZero) {
    const ysq::Minkowski flat;
    const ysq::ChristoffelSymbols<double> gamma =
        ysq::christoffelSymbols(flat, Vec4{3.0, 1.0, -2.0, 4.0});

    for (std::size_t i = 0; i < gamma.size(); ++i) {
        EXPECT_APPROX(gamma[i], 0.0);
    }
}

// --- Structural properties, true of any metric ------------------------------

TEST(SpacetimeMetric, ChristoffelSymbolsAreSymmetricInTheLowerIndices) {
    const ysq::Schwarzschild schwarzschild{ysq::GravitationalParameter{1.0e15}};
    const Vec4 at{0.0, 5.0e7, 1.2, 0.4};
    const ysq::ChristoffelSymbols<double> gamma =
        ysq::christoffelSymbols(schwarzschild, at);

    for (std::size_t lambda = 0; lambda < 4; ++lambda) {
        for (std::size_t mu = 0; mu < 4; ++mu) {
            for (std::size_t nu = 0; nu < 4; ++nu) {
                EXPECT_NEAR(gamma(lambda, mu, nu), gamma(lambda, nu, mu), 1e-9)
                    << "lambda=" << lambda << " mu=" << mu << " nu=" << nu;
            }
        }
    }
}

// --- Causal character --------------------------------------------------------

TEST(SpacetimeMetric, CausalCharacterMatchesTheSignOfTheSelfProduct) {
    const ysq::Minkowski flat;
    const Vec4 at{};
    const double c = ysq::constants::speedOfLight.value();

    const Vec4 atRest{c, 0.0, 0.0, 0.0};
    EXPECT_TRUE(ysq::isTimelike(flat, at, atRest));
    EXPECT_FALSE(ysq::isSpacelike(flat, at, atRest));
    EXPECT_FALSE(ysq::isNull(flat, at, atRest));

    const Vec4 lightlike{1.0, 1.0, 0.0, 0.0};
    EXPECT_TRUE(ysq::isNull(flat, at, lightlike));
    EXPECT_FALSE(ysq::isTimelike(flat, at, lightlike));

    const Vec4 spatial{0.0, 1.0, 0.0, 0.0};
    EXPECT_TRUE(ysq::isSpacelike(flat, at, spatial));
    EXPECT_FALSE(ysq::isTimelike(flat, at, spatial));
}

}  // namespace
