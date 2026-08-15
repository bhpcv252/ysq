#include <Math/SpecialFunctions.hpp>

#include <Compute/CPU/CpuBackend.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

TEST(MathSpecialFunctions, ErfIsZeroAtTheOrigin) {
    EXPECT_NEAR(ysq::erf(0.0), 0.0, 1e-15);
}

TEST(MathSpecialFunctions, ErfApproachesOneForALargeArgument) {
    EXPECT_NEAR(ysq::erf(4.0), 1.0, 1e-7);
}

TEST(MathSpecialFunctions, ErfMatchesATabulatedValue) {
    // erf(1) = 0.8427007929497149...
    EXPECT_NEAR(ysq::erf(1.0), 0.8427007929497149, 1e-12);
}

TEST(MathSpecialFunctions, ErfPlusErfcIsOne) {
    for (const double x : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
        EXPECT_NEAR(ysq::erf(x) + ysq::erfc(x), 1.0, 1e-12);
    }
}

TEST(MathSpecialFunctions, GammaOfAPositiveIntegerIsFactorial) {
    // Gamma(n) = (n - 1)! for a positive integer n.
    EXPECT_NEAR(ysq::gamma(1.0), 1.0, 1e-12);
    EXPECT_NEAR(ysq::gamma(2.0), 1.0, 1e-12);
    EXPECT_NEAR(ysq::gamma(5.0), 24.0, 1e-10);
    EXPECT_NEAR(ysq::gamma(7.0), 720.0, 1e-9);
}

TEST(MathSpecialFunctions, GammaOfOneHalfIsSqrtPi) {
    EXPECT_NEAR(ysq::gamma(0.5), std::sqrt(ysq::kPi<double>), 1e-12);
}

TEST(MathSpecialFunctions, LogGammaMatchesTheLogOfGammaWhereGammaDoesNotOverflow) {
    for (const double x : {1.5, 3.0, 10.0}) {
        EXPECT_NEAR(ysq::logGamma(x), std::log(ysq::gamma(x)), 1e-9);
    }
}

TEST(MathSpecialFunctions, LegendreP0IsOneEverywhere) {
    for (const double x : {-1.0, -0.3, 0.0, 0.7, 1.0}) {
        EXPECT_NEAR(ysq::legendreP(0U, x), 1.0, 1e-14);
    }
}

TEST(MathSpecialFunctions, LegendreP1IsX) {
    for (const double x : {-1.0, -0.3, 0.0, 0.7, 1.0}) {
        EXPECT_NEAR(ysq::legendreP(1U, x), x, 1e-14);
    }
}

TEST(MathSpecialFunctions, LegendreP2MatchesItsClosedForm) {
    // P_2(x) = (3x^2 - 1) / 2.
    for (const double x : {-0.8, -0.2, 0.4, 0.9}) {
        const double expected = (3.0 * x * x - 1.0) / 2.0;
        EXPECT_NEAR(ysq::legendreP(2U, x), expected, 1e-13);
    }
}

TEST(MathSpecialFunctions, AssociatedLegendreP11MatchesItsClosedForm) {
    // P_1^1(x) = -sqrt(1 - x^2), with the Condon-Shortley phase included.
    for (const double x : {-0.9, -0.1, 0.3, 0.6}) {
        const double expected = -std::sqrt(1.0 - x * x);
        EXPECT_NEAR(ysq::legendreP(1U, 1U, x), expected, 1e-13);
    }
}

TEST(MathSpecialFunctions, AssociatedLegendreP21MatchesItsClosedForm) {
    // P_2^1(x) = -3x sqrt(1 - x^2).
    for (const double x : {-0.9, -0.1, 0.3, 0.6}) {
        const double expected = -3.0 * x * std::sqrt(1.0 - x * x);
        EXPECT_NEAR(ysq::legendreP(2U, 1U, x), expected, 1e-12);
    }
}

TEST(MathSpecialFunctions, AssociatedLegendreP22MatchesItsClosedForm) {
    // P_2^2(x) = 3(1 - x^2).
    for (const double x : {-0.9, -0.1, 0.3, 0.6}) {
        const double expected = 3.0 * (1.0 - x * x);
        EXPECT_NEAR(ysq::legendreP(2U, 2U, x), expected, 1e-12);
    }
}

TEST(MathSpecialFunctions, LegendrePWithZeroOrderMatchesTheOrdinaryOverload) {
    for (const double x : {-0.7, 0.1, 0.5}) {
        EXPECT_NEAR(ysq::legendreP(3U, 0U, x), ysq::legendreP(3U, x), 1e-13);
    }
}

TEST(MathSpecialFunctions, BesselJ0IsOneAtTheOrigin) {
    // The A&S rational approximation is fit to about 1e-8 relative
    // accuracy, not to machine precision, so the tolerance reflects the
    // method's own accuracy rather than double's.
    EXPECT_NEAR(ysq::besselJ0(0.0), 1.0, 1e-8);
}

TEST(MathSpecialFunctions, BesselJ1IsZeroAtTheOrigin) {
    EXPECT_NEAR(ysq::besselJ1(0.0), 0.0, 1e-14);
}

TEST(MathSpecialFunctions, BesselJ0MatchesItsKnownFirstZero) {
    EXPECT_NEAR(ysq::besselJ0(2.4048255577), 0.0, 1e-8);
}

TEST(MathSpecialFunctions, BesselJ1MatchesItsKnownFirstZero) {
    EXPECT_NEAR(ysq::besselJ1(3.8317059702), 0.0, 1e-8);
}

TEST(MathSpecialFunctions, BesselJMatchesTheClosedFormOrdersZeroAndOne) {
    for (const double x : {0.5, 2.0, 6.0, 9.5}) {
        EXPECT_NEAR(ysq::besselJ(0U, x), ysq::besselJ0(x), 1e-13);
        EXPECT_NEAR(ysq::besselJ(1U, x), ysq::besselJ1(x), 1e-13);
    }
}

TEST(MathSpecialFunctions, BesselJSatisfiesItsThreeTermRecurrence) {
    // J_{n-1}(x) + J_{n+1}(x) = (2n/x) J_n(x), checked at orders 2 through
    // 4 against x = 1.5 (exercises besselJ's downward, Miller's-algorithm
    // branch, ax <= n) and x = 9.5 (exercises its upward-recurrence
    // branch, ax > n).
    for (const double x : {1.5, 9.5}) {
        for (unsigned n = 2; n <= 4; ++n) {
            const double lhs = ysq::besselJ(n - 1, x) + ysq::besselJ(n + 1, x);
            const double rhs = (2.0 * static_cast<double>(n) / x) * ysq::besselJ(n, x);
            EXPECT_NEAR(lhs, rhs, 1e-7);
        }
    }
}

TEST(MathSpecialFunctions, BesselYSatisfiesItsThreeTermRecurrence) {
    for (const double x : {1.5, 9.5}) {
        for (unsigned n = 2; n <= 4; ++n) {
            const double lhs = ysq::besselY(n - 1, x) + ysq::besselY(n + 1, x);
            const double rhs = (2.0 * static_cast<double>(n) / x) * ysq::besselY(n, x);
            EXPECT_NEAR(lhs, rhs, 1e-9);
        }
    }
}

TEST(MathSpecialFunctions, BesselJAndYSatisfyTheirWronskianIdentity) {
    // J_n(x) Y_{n+1}(x) - J_{n+1}(x) Y_n(x) = -2 / (pi x) (Abramowitz &
    // Stegun 9.1.16): an identity linking the two families together,
    // independent of how either was computed internally.
    for (const double x : {0.8, 3.0, 9.5}) {
        for (unsigned n = 0; n <= 3; ++n) {
            const double lhs = ysq::besselJ(n, x) * ysq::besselY(n + 1, x) -
                               ysq::besselJ(n + 1, x) * ysq::besselY(n, x);
            const double rhs = -2.0 / (ysq::kPi<double> * x);
            EXPECT_NEAR(lhs, rhs, 1e-6);
        }
    }
}

TEST(MathSpecialFunctions, BatchedErfMatchesTheScalarOverloadElementwise) {
    const std::vector<double> x{0.0, 0.5, 1.0, -1.0, 2.0};
    std::vector<double> result(x.size());

    ysq::erf<double>(x, result);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(result[i], ysq::erf(x[i]), 1e-15) << "element " << i;
    }
}

TEST(MathSpecialFunctions, BatchedErfcMatchesTheScalarOverloadElementwise) {
    const std::vector<double> x{0.0, 0.5, 1.0, -1.0, 2.0};
    std::vector<double> result(x.size());

    ysq::erfc<double>(x, result);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(result[i], ysq::erfc(x[i]), 1e-15) << "element " << i;
    }
}

TEST(MathSpecialFunctions, BatchedGammaMatchesTheScalarOverloadElementwise) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 0.5};
    std::vector<double> result(x.size());

    ysq::gamma<double>(x, result);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(result[i], ysq::gamma(x[i]), 1e-12) << "element " << i;
    }
}

TEST(MathSpecialFunctions, BatchedLogGammaMatchesTheScalarOverloadElementwise) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> result(x.size());

    ysq::logGamma<double>(x, result);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(result[i], ysq::logGamma(x[i]), 1e-12) << "element " << i;
    }
}

TEST(MathSpecialFunctions, BatchedLegendrePMatchesTheScalarOverloadElementwise) {
    const std::vector<double> x{0.0, 0.5, 1.0, -1.0};
    std::vector<double> result(x.size());

    ysq::legendreP<double>(2, 1, x, result);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(result[i], ysq::legendreP(2, 1, x[i]), 1e-12) << "element " << i;
    }
}

TEST(MathSpecialFunctions, BatchedLegendrePWithZeroOrderMatchesTheOrdinaryOverload) {
    const std::vector<double> x{0.0, 0.5, 1.0, -1.0};
    std::vector<double> result(x.size());

    ysq::legendreP<double>(3, x, result);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(result[i], ysq::legendreP(3, x[i]), 1e-12) << "element " << i;
    }
}

TEST(MathSpecialFunctions,
     BatchedErfAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // Above SpecialFunctions.hpp's own GPU dispatch threshold and T = float
    // (the only type that ever dispatches). ysq::CpuBackend is called
    // directly as an independent reference (its own agreement with every
    // GPU backend is already covered by
    // tests/integration/compute_backends_agree.cpp; this test only checks
    // that the batched erf overload's marshaling is wired correctly).
    // At the threshold (measured by benchmarks/compute_thresholds.cpp).
    constexpr std::size_t n = 131072;
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = -3.0f + 6.0f * static_cast<float>(i) / static_cast<float>(n - 1);
    }

    std::vector<float> result(n);
    ysq::erf<float>(x, result);

    const ysq::CpuBackend cpu;
    std::vector<float> resultReference(n);
    cpu.batchErf(x, resultReference);

    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        EXPECT_NEAR(result[i], resultReference[i], 1e-5f) << "element " << i;
    }
}

TEST(MathSpecialFunctions,
     BatchedGammaAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // At the threshold (measured by benchmarks/compute_thresholds.cpp).
    constexpr std::size_t n = 131072;
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = 1.0f + 0.01f * static_cast<float>(i % 200);
    }

    std::vector<float> result(n);
    ysq::gamma<float>(x, result);

    const ysq::CpuBackend cpu;
    std::vector<float> resultReference(n);
    cpu.batchGamma(x, resultReference);

    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        EXPECT_NEAR(result[i], resultReference[i], 1e-2f) << "element " << i;
    }
}

TEST(MathSpecialFunctions,
     BatchedLegendrePAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // At the threshold (measured by benchmarks/compute_thresholds.cpp).
    constexpr std::size_t n = 131072;
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(n - 1);
    }

    std::vector<float> result(n);
    ysq::legendreP<float>(4, 2, x, result);

    const ysq::CpuBackend cpu;
    std::vector<float> resultReference(n);
    cpu.batchLegendreP(4, 2, x, resultReference);

    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        EXPECT_NEAR(result[i], resultReference[i], 1e-3f) << "element " << i;
    }
}
