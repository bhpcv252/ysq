#include <Math/Dual.hpp>

#include <Math/Format.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <format>
#include <limits>

namespace {

using ysq::Dual;
using ysq::DualD;

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// Checks a derivative against its closed form. Everything here is exact to a
/// few ulps, so the tolerance is tight on purpose: a chain rule with a wrong
/// factor would still land inside a loose one.
#define EXPECT_DERIVATIVE(expression, at, analytic)                                      \
    EXPECT_NEAR(ysq::derivative([](auto x) { return expression; }, at), analytic,        \
                64.0 * kEps * std::abs(analytic) + 1e-14)

// --- Seeding ----------------------------------------------------------------

TEST(MathDual, AVariableCarriesUnitDerivativeAndAConstantCarriesNone) {
    EXPECT_EQ(DualD::variable(3.0).value, 3.0);
    EXPECT_EQ(DualD::variable(3.0).derivative, 1.0);
    EXPECT_EQ(DualD::constant(3.0).derivative, 0.0);

    // The implicit conversion from the scalar makes a literal a constant,
    // which is what lets a formula be written the ordinary way.
    const DualD fromScalar = 3.0;
    EXPECT_TRUE(identical(fromScalar, DualD::constant(3.0)));
    EXPECT_EQ(DualD{}.value, 0.0);
    EXPECT_EQ(DualD{}.derivative, 0.0);
}

// --- The differentiation rules ----------------------------------------------

TEST(MathDual, SumAndDifferenceRules) {
    EXPECT_DERIVATIVE(x + x, 2.0, 2.0);
    EXPECT_DERIVATIVE(x - x, 2.0, 0.0);
    EXPECT_DERIVATIVE(x + 5.0, 2.0, 1.0);
    EXPECT_DERIVATIVE(5.0 - x, 2.0, -1.0);
    EXPECT_DERIVATIVE(-x, 2.0, -1.0);
}

TEST(MathDual, ProductRule) {
    // d(x^2) = 2x
    EXPECT_DERIVATIVE(x * x, 3.0, 6.0);
    // d(x^3) = 3x^2
    EXPECT_DERIVATIVE(x * x * x, 3.0, 27.0);
    // d(x sin x) = sin x + x cos x
    EXPECT_DERIVATIVE(x * sin(x), 1.3, std::sin(1.3) + 1.3 * std::cos(1.3));
    EXPECT_DERIVATIVE(x * 4.0, 3.0, 4.0);
}

TEST(MathDual, QuotientRule) {
    // d(1/x) = -1/x^2
    EXPECT_DERIVATIVE(1.0 / x, 2.0, -0.25);
    // d(x/(x+1)) = 1/(x+1)^2
    EXPECT_DERIVATIVE(x / (x + 1.0), 3.0, 1.0 / 16.0);
    // d(sin x / x) = (x cos x - sin x)/x^2
    EXPECT_DERIVATIVE(sin(x) / x, 1.3,
                      (1.3 * std::cos(1.3) - std::sin(1.3)) / (1.3 * 1.3));
}

TEST(MathDual, ChainRuleThroughNestedCalls) {
    // d(sin(x^2)) = 2x cos(x^2)
    EXPECT_DERIVATIVE(sin(x * x), 1.3, 2.0 * 1.3 * std::cos(1.69));
    // d(exp(sin x)) = exp(sin x) cos x
    EXPECT_DERIVATIVE(exp(sin(x)), 0.7, std::exp(std::sin(0.7)) * std::cos(0.7));
    // Three deep.
    EXPECT_DERIVATIVE(log(cos(x * x)), 0.5, -2.0 * 0.5 * std::tan(0.25));
}

// --- The function library ---------------------------------------------------

TEST(MathDual, AlgebraicFunctions) {
    EXPECT_DERIVATIVE(sqrt(x), 4.0, 1.0 / 4.0);
    EXPECT_DERIVATIVE(exp(x), 1.5, std::exp(1.5));
    EXPECT_DERIVATIVE(log(x), 2.5, 1.0 / 2.5);
    EXPECT_DERIVATIVE(abs(x), 2.5, 1.0);
    EXPECT_DERIVATIVE(abs(x), -2.5, -1.0);
}

TEST(MathDual, TrigonometricFunctions) {
    EXPECT_DERIVATIVE(sin(x), 0.7, std::cos(0.7));
    EXPECT_DERIVATIVE(cos(x), 0.7, -std::sin(0.7));
    EXPECT_DERIVATIVE(tan(x), 0.7, 1.0 / (std::cos(0.7) * std::cos(0.7)));
    EXPECT_DERIVATIVE(asin(x), 0.4, 1.0 / std::sqrt(1.0 - 0.16));
    EXPECT_DERIVATIVE(acos(x), 0.4, -1.0 / std::sqrt(1.0 - 0.16));
    EXPECT_DERIVATIVE(atan(x), 0.4, 1.0 / 1.16);
}

TEST(MathDual, HyperbolicFunctions) {
    EXPECT_DERIVATIVE(sinh(x), 0.7, std::cosh(0.7));
    EXPECT_DERIVATIVE(cosh(x), 0.7, std::sinh(0.7));
    EXPECT_DERIVATIVE(tanh(x), 0.7, 1.0 - std::tanh(0.7) * std::tanh(0.7));
    EXPECT_DERIVATIVE(asinh(x), 0.7, 1.0 / std::sqrt(0.49 + 1.0));
    EXPECT_DERIVATIVE(acosh(x), 1.7, 1.0 / std::sqrt(1.7 * 1.7 - 1.0));
    EXPECT_DERIVATIVE(atanh(x), 0.4, 1.0 / (1.0 - 0.16));
}

TEST(MathDual, PowerInAllThreeForms) {
    // Dual base, constant exponent: no logarithm, so a negative base is fine.
    EXPECT_DERIVATIVE(pow(x, 3.0), 2.0, 3.0 * 4.0);
    EXPECT_DERIVATIVE(pow(x, 3.0), -2.0, 3.0 * 4.0);
    EXPECT_DERIVATIVE(pow(x, 0.5), 4.0, 0.5 / 2.0);

    // Constant base, dual exponent: d(2^x) = 2^x ln 2
    EXPECT_DERIVATIVE(pow(2.0, x), 3.0, 8.0 * std::log(2.0));

    // Both dual: d(x^x) = x^x (ln x + 1)
    EXPECT_DERIVATIVE(pow(x, x), 2.0, 4.0 * (std::log(2.0) + 1.0));
}

TEST(MathDual, TwoArgumentFunctionsDifferentiateInBothArguments) {
    // atan2 of a point going round the unit circle is the angle itself, so its
    // derivative with respect to that angle is exactly one.
    const double t = 0.9;
    const DualD x{std::cos(t), -std::sin(t)};
    const DualD y{std::sin(t), std::cos(t)};
    EXPECT_NEAR(atan2(y, x).value, t, 1e-15);
    EXPECT_NEAR(atan2(y, x).derivative, 1.0, 1e-15);

    // hypot(3t, 4t) is 5t.
    const DualD a{3.0 * t, 3.0};
    const DualD b{4.0 * t, 4.0};
    EXPECT_NEAR(hypot(a, b).value, 5.0 * t, 1e-15);
    EXPECT_NEAR(hypot(a, b).derivative, 5.0, 1e-15);
}

// --- Accuracy ---------------------------------------------------------------

TEST(MathDual, IsExactWhereAFiniteDifferenceIsNot) {
    // The point of the whole header. A central difference has to trade
    // truncation against cancellation and lands around 1e-11 at best; the dual
    // number evaluates the chain rule directly and is right to the last ulp.
    const double at = 1.3;
    const auto f = [](auto x) { return exp(sin(x * x)); };
    const auto plain = [](double x) { return std::exp(std::sin(x * x)); };

    const double analytic = std::exp(std::sin(at * at)) * std::cos(at * at) * 2.0 * at;

    const double automatic = ysq::derivative(f, at);

    const double step = 1e-5;
    const double finite = (plain(at + step) - plain(at - step)) / (2.0 * step);

    const double automaticError = std::abs(automatic - analytic);
    const double finiteError = std::abs(finite - analytic);

    EXPECT_LT(automaticError, 1e-15) << "automatic differentiation is exact";
    EXPECT_GT(finiteError, 1e-12) << "and the finite difference is not";
    EXPECT_LT(automaticError * 100.0, finiteError)
        << std::format("automatic {:.3e} vs finite {:.3e}", automaticError, finiteError);
}

// --- Higher derivatives -----------------------------------------------------

TEST(MathDual, SecondDerivativesComeFromNesting) {
    EXPECT_NEAR(ysq::secondDerivative([](auto x) { return x * x * x; }, 2.0), 12.0,
                1e-13);
    EXPECT_NEAR(ysq::secondDerivative([](auto x) { return sin(x); }, 0.7), -std::sin(0.7),
                1e-14);
    EXPECT_NEAR(ysq::secondDerivative([](auto x) { return exp(x); }, 1.1), std::exp(1.1),
                1e-13);
    // d2/dx2 log(x) = -1/x^2
    EXPECT_NEAR(ysq::secondDerivative([](auto x) { return log(x); }, 2.0), -0.25, 1e-14);
}

TEST(MathDual, NestingAlsoGivesMixedPartials) {
    // f(x, y) = x^2 y^3, so d2f/dxdy = 6 x y^2. The inner level is seeded in x
    // and the outer in y, and the mixed partial falls out of the doubly nested
    // tangent without any special support for it.
    using Inner = Dual<double>;
    using Outer = Dual<Inner>;

    const double x0 = 1.5;
    const double y0 = 2.0;
    const Outer x{Inner{x0, 1.0}, Inner{0.0, 0.0}};
    const Outer y{Inner{y0, 0.0}, Inner{1.0, 0.0}};

    const Outer f = x * x * y * y * y;

    EXPECT_NEAR(f.value.value, x0 * x0 * y0 * y0 * y0, 1e-14);
    EXPECT_NEAR(f.value.derivative, 2.0 * x0 * y0 * y0 * y0, 1e-14);
    EXPECT_NEAR(f.derivative.value, 3.0 * x0 * x0 * y0 * y0, 1e-14);
    EXPECT_NEAR(f.derivative.derivative, 6.0 * x0 * y0 * y0, 1e-14);
}

// --- Comparison semantics ---------------------------------------------------

TEST(MathDual, ComparisonsLookAtTheValueOnly) {
    const DualD a{1.0, 5.0};
    const DualD b{1.0, -3.0};
    const DualD c{2.0, 5.0};

    // Same value, different derivative: equal, and not interchangeable.
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(identical(a, b));
    EXPECT_TRUE(identical(a, a));

    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(a < c);
    EXPECT_TRUE(c > a);

    // Which is what makes the generic helpers behave: they are asking about
    // magnitude, and a derivative should not change their answer.
    EXPECT_EQ(ysq::clamp(c, DualD{0.0}, DualD{1.5}).value, 1.5);
    EXPECT_EQ(ysq::sign(DualD{-2.0, 9.0}).value, -1.0);
}

TEST(MathDual, ValueOfStripsEveryLayer) {
    EXPECT_EQ(ysq::valueOf(3.0), 3.0);
    EXPECT_EQ(ysq::valueOf(DualD{3.0, 1.0}), 3.0);
    EXPECT_EQ(ysq::valueOf(Dual<Dual<double>>{DualD{3.0, 1.0}, DualD{1.0, 0.0}}), 3.0);

    static_assert(!ysq::isDual<double>);
    static_assert(ysq::isDual<DualD>);
    static_assert(ysq::isDual<Dual<Dual<double>>>);
}

// --- Composition with the containers ----------------------------------------

TEST(MathDual, ComposesWithVectorsForDerivativesAlongACurve) {
    // This composition is the reason Vector is templated on its scalar at all.
    // v(t) = (t, t^2, 3), so dot(v, v) = t^2 + t^4 + 9 with derivative
    // 2t + 4t^3, and |v| differentiates through the square root.
    const double t = 1.4;
    const ysq::Vector3<DualD> v{DualD{t, 1.0}, DualD{t * t, 2.0 * t}, DualD{3.0, 0.0}};

    const double expectedDotDerivative = 2.0 * t + 4.0 * t * t * t;
    EXPECT_NEAR(dot(v, v).value, t * t + t * t * t * t + 9.0, 1e-13);
    EXPECT_NEAR(dot(v, v).derivative, expectedDotDerivative, 1e-13);

    const double norm = std::sqrt(t * t + t * t * t * t + 9.0);
    EXPECT_NEAR(length(v).value, norm, 1e-13);
    EXPECT_NEAR(length(v).derivative, expectedDotDerivative / (2.0 * norm), 1e-13);

    // A unit vector has constant length, so the derivative of its squared
    // length has to be exactly zero however the algebra got there.
    const ysq::Vector3<DualD> unit = normalized(v);
    EXPECT_NEAR(lengthSquared(unit).value, 1.0, 1e-14);
    EXPECT_NEAR(lengthSquared(unit).derivative, 0.0, 1e-14);
}

// --- Formatting -------------------------------------------------------------

TEST(MathDual, FormattingReadsAsAValuePlusATangent) {
    EXPECT_EQ(std::format("{}", DualD{1.0, 2.0}), "1 + 2eps");
    EXPECT_EQ(std::format("{}", DualD{1.0, -2.0}), "1 - 2eps");
    EXPECT_EQ(std::format("{:.2f}", DualD{1.5, -0.25}), "1.50 - 0.25eps");
}

// --- Single precision -------------------------------------------------------

TEST(MathDual, DerivativesHoldAtSinglePrecision) {
    using ysq::DualF;
    constexpr float tol = 1e-5f;

    const DualF x = DualF::variable(2.0f);
    EXPECT_NEAR((x * x).derivative, 4.0f, tol);
    EXPECT_NEAR(sqrt(x).derivative, 1.0f / (2.0f * std::sqrt(2.0f)), tol);
    EXPECT_NEAR(sin(x).derivative, std::cos(2.0f), tol);
    EXPECT_NEAR(ysq::derivative([](auto v) { return exp(v); }, 1.0f), std::exp(1.0f),
                1e-4f);
}

}  // namespace
