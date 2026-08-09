#include <Math/RootFinding.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

// f(x) = x^2 - 2, root at sqrt(2).
double residual(double x) {
    return x * x - 2.0;
}

double derivative(double x) {
    return 2.0 * x;
}

constexpr double kSqrt2 = 1.4142135623730951;

}  // namespace

TEST(MathRootFinding, NewtonRaphsonWithDerivativeFindsSqrtTwo) {
    const double root = ysq::newtonRaphson(residual, derivative, 1.0);
    EXPECT_NEAR(root, kSqrt2, 1e-10);
}

TEST(MathRootFinding, NewtonRaphsonWithoutDerivativeFindsSqrtTwo) {
    const double root = ysq::newtonRaphson(residual, 1.0);
    EXPECT_NEAR(root, kSqrt2, 1e-8);
}

TEST(MathRootFinding, NewtonRaphsonConvergesInAFewIterationsForASmoothFunction) {
    const double root = ysq::newtonRaphson(residual, derivative, 1.0, 1e-14, 10);
    EXPECT_NEAR(root, kSqrt2, 1e-12);
}

TEST(MathRootFinding, SecantFindsSqrtTwoWithoutADerivative) {
    const double root = ysq::secant(residual, 1.0, 2.0);
    EXPECT_NEAR(root, kSqrt2, 1e-10);
}

TEST(MathRootFinding, BisectionFindsSqrtTwoInABracket) {
    const double root = ysq::bisection(residual, 0.0, 2.0);
    EXPECT_NEAR(root, kSqrt2, 1e-9);
}

TEST(MathRootFinding, BisectionHalvesTheBracketEveryIteration) {
    // After n iterations the remaining bracket is (upper - lower) / 2^n, so a
    // handful of iterations only guarantees a coarse tolerance.
    const double root = ysq::bisection(residual, 0.0, 2.0, 1e-9, 4);
    EXPECT_NEAR(root, kSqrt2, 0.2);
}

TEST(MathRootFinding, NewtonRaphsonSolvesKeplersEquationLikeTheGravityModuleDoes) {
    // The same transcendental equation Physics/Gravity/Kepler.cpp solves:
    // M = E - e sin(E), for a representative bound orbit.
    const double meanAnomaly = 1.0;
    const double eccentricity = 0.5;

    const auto f = [&](double eccentricAnomaly) {
        return eccentricAnomaly - eccentricity * std::sin(eccentricAnomaly) - meanAnomaly;
    };
    const auto fPrime = [&](double eccentricAnomaly) {
        return 1.0 - eccentricity * std::cos(eccentricAnomaly);
    };

    const double initialGuess = meanAnomaly + eccentricity * std::sin(meanAnomaly);
    const double eccentricAnomaly =
        ysq::newtonRaphson(f, fPrime, initialGuess, 1e-14, 50);

    EXPECT_NEAR(f(eccentricAnomaly), 0.0, 1e-13);
}
