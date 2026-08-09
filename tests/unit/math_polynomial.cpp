#include <Math/Polynomial.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace {

using ysq::Polynomial;

::testing::AssertionResult rootsMatch(std::vector<double> actual,
                                      std::vector<double> expected,
                                      double tolerance = 1e-8) {
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << "expected " << expected.size() << " roots, got " << actual.size();
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > tolerance) {
            return ::testing::AssertionFailure() << "root " << i << ": expected "
                                                 << expected[i] << ", got " << actual[i];
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

TEST(MathPolynomial, EvaluateMatchesHandComputedResult) {
    // p(x) = 1 + 2x + 3x^2
    const Polynomial<double> p{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(p(2.0), 1.0 + 4.0 + 12.0);
}

TEST(MathPolynomial, DerivativeOfACubicIsAQuadratic) {
    // p(x) = 1 + 2x + 3x^2 + 4x^3 -> p'(x) = 2 + 6x + 12x^2
    const Polynomial<double> p{1.0, 2.0, 3.0, 4.0};
    const Polynomial<double> d = p.derivative();

    ASSERT_EQ(d.coefficients().size(), 3U);
    EXPECT_DOUBLE_EQ(d.coefficients()[0], 2.0);
    EXPECT_DOUBLE_EQ(d.coefficients()[1], 6.0);
    EXPECT_DOUBLE_EQ(d.coefficients()[2], 12.0);
}

TEST(MathPolynomial, DerivativeOfAConstantIsZero) {
    const Polynomial<double> p{5.0};
    EXPECT_DOUBLE_EQ(p.derivative()(3.0), 0.0);
}

TEST(MathPolynomial, LinearRealRootMatchesTheAlgebraicSolution) {
    // 2x - 4 = 0 -> x = 2
    EXPECT_NEAR(*ysq::linearRealRoot(2.0, -4.0), 2.0, 1e-12);
    EXPECT_FALSE(ysq::linearRealRoot(0.0, 1.0).has_value());
}

TEST(MathPolynomial, QuadraticRealRootsMatchesTwoDistinctRoots) {
    // (x - 1)(x - 2) = x^2 - 3x + 2
    EXPECT_TRUE(rootsMatch(ysq::quadraticRealRoots(1.0, -3.0, 2.0), {1.0, 2.0}));
}

TEST(MathPolynomial, QuadraticRealRootsHandlesARepeatedRoot) {
    // (x - 3)^2 = x^2 - 6x + 9: a repeated root is reported once, not twice.
    EXPECT_TRUE(rootsMatch(ysq::quadraticRealRoots(1.0, -6.0, 9.0), {3.0}, 1e-9));
}

TEST(MathPolynomial, QuadraticRealRootsIsEmptyForAComplexConjugatePair) {
    // x^2 + 1 = 0 has no real roots.
    EXPECT_TRUE(ysq::quadraticRealRoots(1.0, 0.0, 1.0).empty());
}

TEST(MathPolynomial, QuadraticRealRootsStaysAccurateWhenBDominatesTheDiscriminant) {
    // A case chosen so naive (-b +- sqrt(disc))/(2a) would cancel badly for
    // the smaller root: a=1, b=1e8, c=1 -> roots near -1e-8 and -1e8.
    const std::vector<double> roots = ysq::quadraticRealRoots(1.0, 1.0e8, 1.0);
    ASSERT_EQ(roots.size(), 2U);
    // Verify by substitution rather than against a precomputed literal, since
    // the point of the test is that the returned roots actually satisfy the
    // equation to high relative precision.
    for (double root : roots) {
        const double residual = root * root + 1.0e8 * root + 1.0;
        EXPECT_NEAR(residual / (1.0e8 * std::abs(root) + 1.0), 0.0, 1e-9);
    }
}

TEST(MathPolynomial, CubicRealRootsFindsThreeDistinctIntegerRoots) {
    // (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
    EXPECT_TRUE(rootsMatch(ysq::cubicRealRoots(1.0, -6.0, 11.0, -6.0), {1.0, 2.0, 3.0}));
}

TEST(MathPolynomial, CubicRealRootsFindsThreeDistinctRootsIncludingNegatives) {
    // (x + 2)(x - 1)(x + 5) = x^3 + 6x^2 + 3x - 10
    EXPECT_TRUE(rootsMatch(ysq::cubicRealRoots(1.0, 6.0, 3.0, -10.0), {-5.0, -2.0, 1.0}));
}

TEST(MathPolynomial, CubicRealRootsFindsOneRealRootForAComplexConjugatePairCase) {
    // (x - 1)(x^2 + 1) = x^3 - x^2 + x - 1, real root at 1 only.
    EXPECT_TRUE(rootsMatch(ysq::cubicRealRoots(1.0, -1.0, 1.0, -1.0), {1.0}));
}

TEST(MathPolynomial, CubicRealRootsHandlesARepeatedRoot) {
    // (x - 2)^2 (x - 5) = x^3 - 9x^2 + 24x - 20: the double root is reported
    // once, so this is the boundary case with two returned values, not three.
    EXPECT_TRUE(
        rootsMatch(ysq::cubicRealRoots(1.0, -9.0, 24.0, -20.0), {2.0, 5.0}, 1e-6));
}

TEST(MathPolynomial, QuarticRealRootsFindsFourDistinctIntegerRoots) {
    // (x - 1)(x - 2)(x - 3)(x - 4) = x^4 - 10x^3 + 35x^2 - 50x + 24
    EXPECT_TRUE(rootsMatch(ysq::quarticRealRoots(1.0, -10.0, 35.0, -50.0, 24.0),
                           {1.0, 2.0, 3.0, 4.0}));
}

TEST(MathPolynomial, QuarticRealRootsHandlesTheBiquadraticCase) {
    // (x^2 - 1)(x^2 - 4) = x^4 - 5x^2 + 4, no odd-degree terms.
    EXPECT_TRUE(rootsMatch(ysq::quarticRealRoots(1.0, 0.0, -5.0, 0.0, 4.0),
                           {-2.0, -1.0, 1.0, 2.0}));
}

TEST(MathPolynomial, QuarticRealRootsFindsTwoRealRootsWhenTwoAreComplex) {
    // (x - 1)(x + 1)(x^2 + 1) = x^4 - 1, real roots at +-1 only.
    EXPECT_TRUE(rootsMatch(ysq::quarticRealRoots(1.0, 0.0, 0.0, 0.0, -1.0), {-1.0, 1.0}));
}

TEST(MathPolynomial, RealRootsDispatchesToTheClosedFormForADegreeThreePolynomial) {
    // Ascending coefficients: -6 + 11x - 6x^2 + x^3, same cubic as above.
    const Polynomial<double> p{-6.0, 11.0, -6.0, 1.0};
    EXPECT_TRUE(rootsMatch(ysq::realRoots(p), {1.0, 2.0, 3.0}));
}

TEST(MathPolynomial, RealRootsHandlesADegreeFivePolynomialByDeflation) {
    // (x - 1)(x - 2)(x - 3)(x - 4)(x - 5)
    // = x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120
    const Polynomial<double> p{-120.0, 274.0, -225.0, 85.0, -15.0, 1.0};
    EXPECT_TRUE(rootsMatch(ysq::realRoots(p), {1.0, 2.0, 3.0, 4.0, 5.0}, 1e-6));
}

TEST(MathPolynomial, RealRootsTrimsTrailingZeroCoefficientsToFindTheEffectiveDegree) {
    // Stored as degree 4 (zero-padded) but effectively (x-1)(x-2), degree 2.
    const Polynomial<double> p{2.0, -3.0, 1.0, 0.0, 0.0};
    EXPECT_TRUE(rootsMatch(ysq::realRoots(p), {1.0, 2.0}));
}

TEST(MathPolynomial, DeflatedQuotientSatisfiesTheOriginalPolynomialAtEveryFoundRoot) {
    // A cross-check independent of the closed forms: every root realRoots
    // returns must make the original polynomial (not just the deflated one)
    // evaluate to (nearly) zero.
    const Polynomial<double> p{-120.0, 274.0, -225.0, 85.0, -15.0, 1.0};
    for (double root : ysq::realRoots(p)) {
        EXPECT_NEAR(p(root), 0.0, 1e-6);
    }
}
