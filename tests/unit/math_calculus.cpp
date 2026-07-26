#include <Math/Calculus.hpp>

#include <Math/Dual.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <vector>

namespace {

using ysq::Mat3;
using ysq::Vec3;

constexpr double kPi = ysq::kPi<double>;

/// Counts how many times an integrand is evaluated, so a test can say that
/// adaptive refinement actually costs less rather than merely converging.
template <class F>
class Counted {
public:
    explicit Counted(F f) : m_f(f) {}

    double operator()(double x) const {
        ++m_calls;
        return m_f(x);
    }

    [[nodiscard]] std::size_t calls() const { return m_calls; }

private:
    F m_f;
    mutable std::size_t m_calls = 0;
};

/// The observed order of a method, from how its error shrinks when the step is
/// halved. log2 of the error ratio is the exponent.
double observedOrder(double coarseError, double fineError) {
    return std::log2(coarseError / fineError);
}

// --- Finite differences vs the closed form ----------------------------------

TEST(MathCalculus, FiniteDifferencesApproximateTheDerivative) {
    const auto f = [](double x) { return std::exp(std::sin(x)); };
    const auto analytic = [](double x) {
        return std::exp(std::sin(x)) * std::cos(x);
    };

    for (const double at : {0.3, 1.0, 2.5, -1.2}) {
        EXPECT_NEAR(ysq::forwardDifference(f, at), analytic(at), 1e-7)
            << "forward, at " << at;
        EXPECT_NEAR(ysq::backwardDifference(f, at), analytic(at), 1e-7);
        // A central difference at its optimal step is worth about eleven
        // digits, and Richardson about thirteen. Neither reaches the fifteen
        // that the dual-number path does, which is the whole argument for
        // preferring it.
        EXPECT_NEAR(ysq::centralDifference(f, at), analytic(at), 1e-9);
        EXPECT_NEAR(ysq::richardsonDerivative(f, at), analytic(at), 1e-11);
    }
}

TEST(MathCalculus, TheStencilsHitTheirAdvertisedOrder) {
    // Measured, not asserted by name. The step range is chosen to stay well
    // clear of the rounding floor at the fine end and of the large-step regime
    // at the coarse end, so the error is truncation throughout and the ratio
    // means what it says.
    const auto f = [](double x) { return std::exp(std::sin(x)); };
    const double at = 0.7;
    const double analytic = std::exp(std::sin(at)) * std::cos(at);

    const double coarseForward =
        std::abs(ysq::forwardDifference(f, at, 1e-3) - analytic);
    const double fineForward =
        std::abs(ysq::forwardDifference(f, at, 5e-4) - analytic);
    EXPECT_NEAR(observedOrder(coarseForward, fineForward), 1.0, 0.1);

    const double coarseCentral =
        std::abs(ysq::centralDifference(f, at, 1e-2) - analytic);
    const double fineCentral =
        std::abs(ysq::centralDifference(f, at, 5e-3) - analytic);
    EXPECT_NEAR(observedOrder(coarseCentral, fineCentral), 2.0, 0.1);

    const auto secondAnalytic =
        std::exp(std::sin(at)) * (std::cos(at) * std::cos(at) - std::sin(at));
    const double coarseSecond =
        std::abs(ysq::secondCentralDifference(f, at, 1e-2) - secondAnalytic);
    const double fineSecond =
        std::abs(ysq::secondCentralDifference(f, at, 5e-3) - secondAnalytic);
    EXPECT_NEAR(observedOrder(coarseSecond, fineSecond), 2.0, 0.1);
}

TEST(MathCalculus, SecondDifferenceApproximatesTheSecondDerivative) {
    const auto f = [](double x) { return x * x * x; };
    EXPECT_NEAR(ysq::secondCentralDifference(f, 2.0), 12.0, 1e-4);

    const auto sine = [](double x) { return std::sin(x); };
    EXPECT_NEAR(ysq::secondCentralDifference(sine, 0.7), -std::sin(0.7), 1e-5);
}

TEST(MathCalculus, RichardsonBeatsThePlainCentralDifference) {
    const auto f = [](double x) { return std::exp(std::sin(x)); };
    const double at = 0.7;
    const double analytic = std::exp(std::sin(at)) * std::cos(at);

    const double central = std::abs(ysq::centralDifference(f, at) - analytic);
    const double extrapolated =
        std::abs(ysq::richardsonDerivative(f, at) - analytic);

    EXPECT_LT(extrapolated, central)
        << std::format("richardson {:.3e} vs central {:.3e}", extrapolated,
                       central);
    EXPECT_LT(extrapolated, 1e-11);
}

TEST(MathCalculus, TheExactDerivativeBeatsEveryFiniteDifference) {
    // Which is why `derivative` is the dual-number one and the approximations
    // are all named for what they are.
    const auto f = [](auto x) { return exp(sin(x)); };
    const auto plain = [](double x) { return std::exp(std::sin(x)); };
    const double at = 0.7;
    const double analytic = std::exp(std::sin(at)) * std::cos(at);

    EXPECT_NEAR(ysq::derivative(f, at), analytic, 1e-15);
    EXPECT_GT(std::abs(ysq::centralDifference(plain, at) - analytic), 1e-14);
}

// --- Vector calculus --------------------------------------------------------

TEST(MathCalculus, GradientOfASquaredLengthIsTwiceThePosition) {
    const auto field = [](const auto& v) { return dot(v, v); };
    const Vec3 at{1.0, -2.0, 3.5};

    EXPECT_VEC_NEAR(ysq::gradient(field, at), at * 2.0, 1e-14);
    EXPECT_VEC_NEAR(ysq::numericalGradient(field, at), at * 2.0, 1e-7);
}

TEST(MathCalculus, GradientMatchesAKnownScalarField) {
    // f(x, y, z) = x^2 y + sin(z), so grad f = (2xy, x^2, cos z).
    const auto field = [](const auto& v) { return v.x * v.x * v.y + sin(v.z); };
    const Vec3 at{1.5, -0.5, 0.8};
    const Vec3 analytic{2.0 * at.x * at.y, at.x * at.x, std::cos(at.z)};

    EXPECT_VEC_NEAR(ysq::gradient(field, at), analytic, 1e-14);
    EXPECT_VEC_NEAR(ysq::numericalGradient(field, at), analytic, 1e-8);
}

TEST(MathCalculus, JacobianOfAKnownMap) {
    // f(v) = (x^2, x y, z^3), so the Jacobian is
    //   [2x   0   0 ]
    //   [ y   x   0 ]
    //   [ 0   0  3z^2]
    const auto flow = [](const auto& v) {
        using V = std::remove_cvref_t<decltype(v)>;
        return V{v.x * v.x, v.x * v.y, v.z * v.z * v.z};
    };
    const Vec3 at{1.5, -0.5, 2.0};

    const Mat3 analytic =
        Mat3::fromRows({2.0 * at.x, 0.0, 0.0}, {at.y, at.x, 0.0},
                       {0.0, 0.0, 3.0 * at.z * at.z});

    EXPECT_MAT_NEAR(ysq::jacobian(flow, at), analytic, 1e-13);
    EXPECT_MAT_NEAR(ysq::numericalJacobian(flow, at), analytic, 1e-7);
}

TEST(MathCalculus, HessianOfAQuadraticFormIsConstantAndSymmetric) {
    // f(v) = x^2 + 3 y^2 + 2 x z, so the Hessian is a constant
    //   [2 0 2]
    //   [0 6 0]
    //   [2 0 0]
    const auto field = [](const auto& v) {
        return v.x * v.x + v.y * v.y * 3.0 + v.x * v.z * 2.0;
    };
    const Mat3 analytic = Mat3::fromRows({2.0, 0.0, 2.0}, {0.0, 6.0, 0.0},
                                         {2.0, 0.0, 0.0});

    for (const Vec3& at : {Vec3{0.0, 0.0, 0.0}, Vec3{1.5, -2.0, 3.0}}) {
        const Mat3 computed = ysq::hessian(field, at);
        EXPECT_MAT_NEAR(computed, analytic, 1e-13);
        // Both triangles are computed independently, so the symmetry is a
        // result rather than a construction.
        EXPECT_MAT_NEAR(transpose(computed), computed, 1e-13);
    }
}

TEST(MathCalculus, HessianOfANonQuadraticFieldMatchesTheFiniteDifference) {
    // f(v) = exp(x y) + sin(z)
    const auto field = [](const auto& v) { return exp(v.x * v.y) + sin(v.z); };
    const Vec3 at{0.6, 0.8, 1.1};

    const Mat3 exact = ysq::hessian(field, at);
    const Mat3 approximate = ysq::numericalHessian(field, at);

    EXPECT_MAT_NEAR(exact, approximate, 1e-4);
    EXPECT_MAT_NEAR(transpose(exact), exact, 1e-13);

    // A couple of entries against the closed form, so the two are not simply
    // agreeing on the same mistake.
    const double e = std::exp(at.x * at.y);
    EXPECT_NEAR(exact(0, 0), at.y * at.y * e, 1e-12);
    EXPECT_NEAR(exact(0, 1), e * (1.0 + at.x * at.y), 1e-12);
    EXPECT_NEAR(exact(2, 2), -std::sin(at.z), 1e-12);
}

// --- Quadrature -------------------------------------------------------------

TEST(MathCalculus, TrapezoidIsExactOnALine) {
    const auto line = [](double x) { return 3.0 * x + 1.0; };
    // Integral over [0, 2] is 3*2 + 2 = 8.
    EXPECT_NEAR(ysq::trapezoid(line, 0.0, 2.0, 1), 8.0, 1e-13);
    EXPECT_NEAR(ysq::trapezoid(line, 0.0, 2.0, 16), 8.0, 1e-13);

    EXPECT_EQ(ysq::trapezoid(line, 0.0, 2.0, 0), 0.0);
}

TEST(MathCalculus, SimpsonIsExactOnACubic) {
    // One degree better than the quadratic it is derived from, which is the
    // thing worth knowing about Simpson's rule.
    const auto cubic = [](double x) {
        return 2.0 * x * x * x - x * x + 3.0 * x - 1.0;
    };
    // Integral over [0, 2]: 8 - 8/3 + 6 - 2.
    const double exact = 8.0 - 8.0 / 3.0 + 6.0 - 2.0;

    EXPECT_NEAR(ysq::simpson(cubic, 0.0, 2.0, 2), exact, 1e-13);
    EXPECT_NEAR(ysq::simpson(cubic, 0.0, 2.0, 10), exact, 1e-13);

    // An odd count is rounded up rather than silently mishandled.
    EXPECT_NEAR(ysq::simpson(cubic, 0.0, 2.0, 3), exact, 1e-13);

    // But not on a quartic.
    const auto quartic = [](double x) { return x * x * x * x; };
    EXPECT_GT(std::abs(ysq::simpson(quartic, 0.0, 2.0, 2) - 32.0 / 5.0), 1e-3);
}

TEST(MathCalculus, CompositeRulesHitTheirAdvertisedOrder) {
    const auto f = [](double x) { return std::exp(std::sin(x)); };
    constexpr double lower = 0.0;
    constexpr double upper = 2.0;
    const double reference = ysq::romberg(f, lower, upper, 20, 1e-15);

    const double coarseTrapezoid =
        std::abs(ysq::trapezoid(f, lower, upper, 16) - reference);
    const double fineTrapezoid =
        std::abs(ysq::trapezoid(f, lower, upper, 32) - reference);
    EXPECT_NEAR(observedOrder(coarseTrapezoid, fineTrapezoid), 2.0, 0.1);

    const double coarseSimpson =
        std::abs(ysq::simpson(f, lower, upper, 8) - reference);
    const double fineSimpson =
        std::abs(ysq::simpson(f, lower, upper, 16) - reference);
    EXPECT_NEAR(observedOrder(coarseSimpson, fineSimpson), 4.0, 0.1);
}

TEST(MathCalculus, GaussLegendreIsExactToDegreeTwiceItsOrderMinusOne) {
    // The sharpest statement available about a quadrature rule, and the one
    // that catches a mistyped node or weight: nothing close to the right table
    // integrates every polynomial up to that degree exactly.
    const auto monomial = [](std::size_t power) {
        return [power](double x) { return std::pow(x, static_cast<double>(power)); };
    };
    const auto exactOverUnitInterval = [](std::size_t power) {
        return 1.0 / static_cast<double>(power + 1);
    };

    for (std::size_t power = 0; power <= 3; ++power) {
        EXPECT_NEAR(ysq::gaussLegendre<2>(monomial(power), 0.0, 1.0),
                    exactOverUnitInterval(power), 1e-14)
            << "2-point rule, degree " << power;
    }
    for (std::size_t power = 0; power <= 5; ++power) {
        EXPECT_NEAR(ysq::gaussLegendre<3>(monomial(power), 0.0, 1.0),
                    exactOverUnitInterval(power), 1e-14);
    }
    for (std::size_t power = 0; power <= 7; ++power) {
        EXPECT_NEAR(ysq::gaussLegendre<4>(monomial(power), 0.0, 1.0),
                    exactOverUnitInterval(power), 1e-14);
    }
    for (std::size_t power = 0; power <= 9; ++power) {
        EXPECT_NEAR(ysq::gaussLegendre<5>(monomial(power), 0.0, 1.0),
                    exactOverUnitInterval(power), 1e-14);
    }

    // And one degree past its reach, it stops being exact.
    EXPECT_GT(std::abs(ysq::gaussLegendre<2>(monomial(4), 0.0, 1.0) -
                       exactOverUnitInterval(4)),
              1e-4);
    EXPECT_GT(std::abs(ysq::gaussLegendre<3>(monomial(6), 0.0, 1.0) -
                       exactOverUnitInterval(6)),
              1e-5);
}

TEST(MathCalculus, GaussLegendreHandlesAGeneralIntervalAndASmoothIntegrand) {
    // Integral of sin over [0, pi] is exactly 2.
    EXPECT_NEAR(ysq::gaussLegendre<5>([](double x) { return std::sin(x); }, 0.0,
                                      kPi),
                2.0, 1e-6);
    // Five evaluations, against the several dozen a composite rule would need
    // for the same accuracy.
    EXPECT_GT(std::abs(ysq::simpson([](double x) { return std::sin(x); }, 0.0, kPi,
                                    4) -
                       2.0),
              1e-6);
}

TEST(MathCalculus, RombergConvergesFastOnASmoothIntegrand) {
    EXPECT_NEAR(ysq::romberg([](double x) { return std::sin(x); }, 0.0, kPi), 2.0,
                1e-12);
    EXPECT_NEAR(ysq::romberg([](double x) { return std::exp(x); }, 0.0, 1.0),
                std::exp(1.0) - 1.0, 1e-12);
    EXPECT_NEAR(ysq::romberg([](double x) { return 1.0 / x; }, 1.0, 2.0),
                std::log(2.0), 1e-12);
}

TEST(MathCalculus, AdaptiveSimpsonPutsItsEffortWhereTheIntegrandNeedsIt) {
    // A narrow peak. A composite rule has to use its finest spacing across the
    // whole interval; the adaptive one refines only where the error estimate
    // fails, which is the entire reason it exists.
    const auto peaked = [](double x) {
        const double scaled = 20000.0 * (x - 0.5);
        return 1.0 / (1.0 + scaled * scaled);
    };
    const double exact = (std::atan(10000.0) - std::atan(-10000.0)) / 20000.0;

    const Counted adaptive{peaked};
    const double adaptiveResult = ysq::adaptiveSimpson(adaptive, 0.0, 1.0, 1e-10);
    EXPECT_NEAR(adaptiveResult, exact, 1e-9) << "it meets the tolerance it was given";

    // The comparison that means something is at equal cost. Composite Simpson
    // given the same evaluation budget has to spread it evenly, most of it
    // across the flat part where nothing is happening, so it resolves the peak
    // with whatever is left.
    const Counted composite{peaked};
    const double compositeResult =
        ysq::simpson(composite, 0.0, 1.0, adaptive.calls());
    EXPECT_LE(composite.calls(), adaptive.calls() + 2) << "same budget";

    EXPECT_LT(std::abs(adaptiveResult - exact), std::abs(compositeResult - exact))
        << std::format("adaptive {:.3e} vs composite {:.3e} over {} evaluations",
                       std::abs(adaptiveResult - exact),
                       std::abs(compositeResult - exact), adaptive.calls());
}

TEST(MathCalculus, AdaptiveSimpsonAgreesWithTheOtherRulesOnEasyIntegrands) {
    const auto f = [](double x) { return std::exp(std::sin(x)); };
    const double reference = ysq::romberg(f, 0.0, 2.0, 20, 1e-15);

    EXPECT_NEAR(ysq::adaptiveSimpson(f, 0.0, 2.0, 1e-12), reference, 1e-10);
    EXPECT_NEAR(ysq::gaussLegendre<5>(f, 0.0, 2.0), reference, 1e-5);
}

TEST(MathCalculus, QuadratureRespectsOrientationAndAdditivity) {
    const auto f = [](double x) { return x * x + 1.0; };

    // Swapping the limits flips the sign.
    EXPECT_NEAR(ysq::simpson(f, 0.0, 2.0, 10), -ysq::simpson(f, 2.0, 0.0, 10),
                1e-13);
    EXPECT_NEAR(ysq::gaussLegendre<4>(f, 0.0, 2.0),
                -ysq::gaussLegendre<4>(f, 2.0, 0.0), 1e-13);

    // And splitting the interval adds up.
    EXPECT_NEAR(ysq::simpson(f, 0.0, 1.0, 10) + ysq::simpson(f, 1.0, 3.0, 10),
                ysq::simpson(f, 0.0, 3.0, 60), 1e-10);
}

// --- Non-finite integrands --------------------------------------------------

TEST(MathCalculus, ANonFiniteIntegrandPropagatesRatherThanRunningForever) {
    // Every comparison against a NaN is false, so a non-finite estimate can
    // never satisfy an error test. Adaptive Simpson used to respond by
    // recursing to full depth in every branch rather than in a few, which at
    // the default depth is 2^40 calls: not a slow answer, no answer at all.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto poisoned = [nan](double x) { return x < 0.5 ? 1.0 : nan; };

    EXPECT_TRUE(std::isnan(ysq::adaptiveSimpson(poisoned, 0.0, 1.0, 1e-10)));
    EXPECT_TRUE(std::isnan(ysq::romberg(poisoned, 0.0, 1.0)));
    EXPECT_TRUE(std::isnan(ysq::simpson(poisoned, 0.0, 1.0, 16)));
    EXPECT_TRUE(std::isnan(ysq::trapezoid(poisoned, 0.0, 1.0, 16)));
    EXPECT_TRUE(std::isnan(ysq::gaussLegendre<5>(poisoned, 0.0, 1.0)));

    // An infinite integrand is the same case: an improper integral is not
    // something a Simpson rule can do, so it says so instead of grinding.
    const double inf = std::numeric_limits<double>::infinity();
    const auto unbounded = [inf](double x) { return x <= 0.0 ? inf : 1.0 / x; };
    EXPECT_FALSE(std::isfinite(ysq::adaptiveSimpson(unbounded, 0.0, 1.0, 1e-10)));

    // And a well-behaved integrand is unaffected by the guard.
    EXPECT_NEAR(ysq::adaptiveSimpson([](double x) { return x * x; }, 0.0, 1.0, 1e-12),
                1.0 / 3.0, 1e-12);
}

// --- Single precision -------------------------------------------------------

TEST(MathCalculus, WorksAtSinglePrecision) {
    const auto f = [](float x) { return x * x; };
    EXPECT_NEAR(ysq::centralDifference(f, 2.0f), 4.0f, 1e-2f);
    EXPECT_NEAR(ysq::simpson(f, 0.0f, 1.0f, 8), 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(ysq::gaussLegendre<3>(f, 0.0f, 1.0f), 1.0f / 3.0f, 1e-6f);

    const auto field = [](const auto& v) { return dot(v, v); };
    const ysq::Vec3f at{1.0f, -2.0f, 3.0f};
    EXPECT_VEC_NEAR(ysq::gradient(field, at), at * 2.0f, 1e-5f);
}

}  // namespace
