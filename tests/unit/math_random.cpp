#include <Math/Random.hpp>

#include <gtest/gtest.h>

TEST(MathRandom, SameSeedProducesTheSameSequence) {
    ysq::RandomEngine a = ysq::makeRandomEngine(1234);
    ysq::RandomEngine b = ysq::makeRandomEngine(1234);

    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(ysq::uniformReal(a, 0.0, 1.0), ysq::uniformReal(b, 0.0, 1.0));
    }
}

TEST(MathRandom, DifferentSeedsEventuallyDiffer) {
    ysq::RandomEngine a = ysq::makeRandomEngine(1);
    ysq::RandomEngine b = ysq::makeRandomEngine(2);

    bool sawADifference = false;
    for (int i = 0; i < 10; ++i) {
        if (ysq::uniformReal(a, 0.0, 1.0) != ysq::uniformReal(b, 0.0, 1.0)) {
            sawADifference = true;
        }
    }
    EXPECT_TRUE(sawADifference);
}

TEST(MathRandom, UniformRealStaysWithinItsRequestedBounds) {
    ysq::RandomEngine engine = ysq::makeRandomEngine(7);
    for (int i = 0; i < 10000; ++i) {
        const double x = ysq::uniformReal(engine, -3.0, 5.0);
        EXPECT_GE(x, -3.0);
        EXPECT_LT(x, 5.0);
    }
}

TEST(MathRandom, UniformIntStaysWithinItsInclusiveBounds) {
    ysq::RandomEngine engine = ysq::makeRandomEngine(9);
    for (int i = 0; i < 10000; ++i) {
        const int x = ysq::uniformInt(engine, 1, 6);
        EXPECT_GE(x, 1);
        EXPECT_LE(x, 6);
    }
}

TEST(MathRandom, NormalSampleMeanConvergesToTheRequestedMean) {
    ysq::RandomEngine engine = ysq::makeRandomEngine(11);
    double total = 0.0;
    constexpr int kSamples = 200000;
    for (int i = 0; i < kSamples; ++i) {
        total += ysq::normal(engine, 5.0, 2.0);
    }
    const double sampleMean = total / static_cast<double>(kSamples);
    EXPECT_NEAR(sampleMean, 5.0, 0.05);
}

TEST(MathRandom, NormalCdfAtTheMeanIsOneHalf) {
    EXPECT_NEAR(ysq::normalCdf(5.0, 5.0, 2.0), 0.5, 1e-12);
}

TEST(MathRandom, NormalCdfMatchesATabulatedStandardNormalValue) {
    // Phi(1) = 0.8413447460685429...
    EXPECT_NEAR(ysq::normalCdf(1.0), 0.8413447460685429, 1e-10);
}

TEST(MathRandom, NormalCdfIsMonotonicallyIncreasing) {
    EXPECT_LT(ysq::normalCdf(-1.0), ysq::normalCdf(0.0));
    EXPECT_LT(ysq::normalCdf(0.0), ysq::normalCdf(1.0));
}

TEST(MathRandom, PoissonSampleMeanConvergesToTheRequestedRate) {
    ysq::RandomEngine engine = ysq::makeRandomEngine(13);
    long total = 0;
    constexpr int kSamples = 200000;
    for (int i = 0; i < kSamples; ++i) {
        total += ysq::poisson(engine, 4.0);
    }
    const double sampleMean = static_cast<double>(total) / static_cast<double>(kSamples);
    EXPECT_NEAR(sampleMean, 4.0, 0.05);
}

TEST(MathRandom, MonteCarloIntegrateConvergesToTheKnownIntegralOfXSquared) {
    ysq::RandomEngine engine = ysq::makeRandomEngine(17);
    const auto square = [](double x) { return x * x; };
    // Integral of x^2 from 0 to 1 is 1/3.
    const double result = ysq::monteCarloIntegrate(square, 0.0, 1.0, 500000, engine);
    EXPECT_NEAR(result, 1.0 / 3.0, 0.01);
}
