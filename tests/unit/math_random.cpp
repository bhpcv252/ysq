#include <Math/Random.hpp>

#include <Compute/CPU/CpuBackend.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

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

TEST(MathRandom, ParallelUniformRealIsDeterministicForTheSameSeedAndOffset) {
    std::vector<double> a(64);
    std::vector<double> b(64);

    ysq::parallelUniformReal<double>(21, 0, a);
    ysq::parallelUniformReal<double>(21, 0, b);

    EXPECT_EQ(a, b);
}

TEST(MathRandom, ParallelUniformRealStaysWithinItsRequestedBounds) {
    std::vector<double> result(10000);

    ysq::parallelUniformReal<double>(22, 0, result, -3.0, 5.0);

    for (double x : result) {
        EXPECT_GE(x, -3.0);
        EXPECT_LT(x, 5.0);
    }
}

TEST(MathRandom, ParallelUniformRealAtNonZeroOffsetContinuesTheSameLogicalStream) {
    std::vector<double> wholeStream(20);
    ysq::parallelUniformReal<double>(23, 0, wholeStream);

    std::vector<double> firstHalf(12);
    std::vector<double> secondHalf(8);
    ysq::parallelUniformReal<double>(23, 0, firstHalf);
    ysq::parallelUniformReal<double>(23, 12, secondHalf);

    for (std::size_t i = 0; i < firstHalf.size(); ++i) {
        EXPECT_DOUBLE_EQ(firstHalf[i], wholeStream[i]) << "element " << i;
    }
    for (std::size_t i = 0; i < secondHalf.size(); ++i) {
        EXPECT_DOUBLE_EQ(secondHalf[i], wholeStream[12 + i]) << "element " << i;
    }
}

TEST(MathRandom, ParallelUniformRealSampleMeanIsNearTheMidpointOfItsRange) {
    std::vector<double> result(200000);

    ysq::parallelUniformReal<double>(24, 0, result, 10.0, 20.0);

    double total = 0.0;
    for (double x : result) {
        total += x;
    }
    EXPECT_NEAR(total / static_cast<double>(result.size()), 15.0, 0.05);
}

TEST(MathRandom, ParallelNormalIsDeterministicForTheSameSeedAndOffset) {
    std::vector<double> a(64);
    std::vector<double> b(64);

    ysq::parallelNormal<double>(25, 0, a);
    ysq::parallelNormal<double>(25, 0, b);

    EXPECT_EQ(a, b);
}

TEST(MathRandom, ParallelNormalSampleMeanAndVarianceAreNearTheRequestedMeanAndStddev) {
    std::vector<double> result(200000);

    ysq::parallelNormal<double>(26, 0, result, 5.0, 2.0);

    double total = 0.0;
    for (double x : result) {
        total += x;
    }
    const double mean = total / static_cast<double>(result.size());

    double variance = 0.0;
    for (double x : result) {
        variance += (x - mean) * (x - mean);
    }
    variance /= static_cast<double>(result.size());

    EXPECT_NEAR(mean, 5.0, 0.05);
    EXPECT_NEAR(variance, 4.0, 0.15);
}

TEST(MathRandom, ParallelMonteCarloIntegrateConvergesToTheKnownIntegralOfXSquared) {
    const auto square = [](double x) { return x * x; };
    const double result = ysq::parallelMonteCarloIntegrate(square, 0.0, 1.0, 500000, 27);
    EXPECT_NEAR(result, 1.0 / 3.0, 0.01);
}

TEST(MathRandom,
     ParallelUniformRealAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // Above Random.hpp's own GPU dispatch threshold and T = float (the only
    // type that ever dispatches). ysq::CpuBackend is called directly as an
    // independent reference (its own agreement with every GPU backend is
    // already covered by tests/integration/compute_backends_agree.cpp; this
    // test only checks that parallelUniformReal's marshaling -- including
    // the affine rescale to [lo, hi) applied after the raw [0, 1) draw --
    // is wired correctly).
    // At kRandomGpuDispatchThreshold (32768; measured by
    // benchmarks/compute_thresholds.cpp).
    constexpr std::size_t n = 32768;
    constexpr std::uint64_t seed = 555;
    constexpr std::uint64_t offset = 9;
    constexpr float lo = -2.0f;
    constexpr float hi = 3.0f;

    std::vector<float> result(n);
    ysq::parallelUniformReal<float>(seed, offset, result, lo, hi);

    const ysq::CpuBackend cpu;
    std::vector<float> raw(n);
    cpu.batchUniformReal(seed, offset, raw);

    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        const float expected = lo + raw[i] * (hi - lo);
        EXPECT_NEAR(result[i], expected, 1e-4f) << "element " << i;
    }
}

TEST(MathRandom, ParallelNormalAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // At kRandomGpuDispatchThreshold (32768; measured by
    // benchmarks/compute_thresholds.cpp).
    constexpr std::size_t n = 32768;
    constexpr std::uint64_t seed = 777;
    constexpr std::uint64_t offset = 3;
    constexpr float mean = 10.0f;
    constexpr float stddev = 2.5f;

    std::vector<float> result(n);
    ysq::parallelNormal<float>(seed, offset, result, mean, stddev);

    const ysq::CpuBackend cpu;
    std::vector<float> raw(n);
    cpu.batchNormal(seed, offset, raw);

    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        const float expected = mean + raw[i] * stddev;
        EXPECT_NEAR(result[i], expected, 1e-3f) << "element " << i;
    }
}
