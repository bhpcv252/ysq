#include <Compute/CPU/CpuBackend.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace {

using ysq::CpuBackend;

TEST(ComputeCpu, SaxpyMatchesTheDefinitionElementwise) {
    const CpuBackend backend;
    std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y{10.0f, 20.0f, 30.0f, 40.0f};
    backend.saxpy(x, y, 2.0f);
    EXPECT_FLOAT_EQ(y[0], 12.0f);
    EXPECT_FLOAT_EQ(y[1], 24.0f);
    EXPECT_FLOAT_EQ(y[2], 36.0f);
    EXPECT_FLOAT_EQ(y[3], 48.0f);
}

TEST(ComputeCpu, SaxpyOnEmptySpansDoesNothing) {
    const CpuBackend backend;
    std::vector<float> x;
    std::vector<float> y;
    backend.saxpy(x, y, 3.0f);
    EXPECT_TRUE(y.empty());
}

TEST(ComputeCpu, SumMatchesTheObviousTotalForSmallInputs) {
    const CpuBackend backend;
    const std::array<float, 5> x{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    EXPECT_FLOAT_EQ(backend.sum(x), 15.0f);
}

TEST(ComputeCpu, SumOfEmptyIsZero) {
    const CpuBackend backend;
    EXPECT_FLOAT_EQ(backend.sum(std::span<const float>{}), 0.0f);
}

// A naive float accumulator loses the small terms once the running total is
// large enough that they round away. Compensated summation is what the CPU
// backend has to get right to be trustworthy as the reference: this is a
// contrived case a naive accumulator gets visibly wrong and this one does not.
TEST(ComputeCpu, SumStaysAccurateWhenSmallTermsFollowALargeOne) {
    const CpuBackend backend;
    std::vector<float> x(100000, 1e-4f);
    x[0] = 1e4f;
    // x[0] overwrote one of the small terms, so only x.size() - 1 of them
    // remain: 1e4 + 99999 * 1e-4 = 10009.9999.
    const float expected = 1e4f + static_cast<float>(x.size() - 1) * 1e-4f;

    EXPECT_NEAR(backend.sum(x), expected, 1e-2f);

    float naive = 0.0f;
    for (const float v : x) {
        naive += v;
    }
    EXPECT_GT(std::abs(naive - expected), 1e-2f)
        << "the naive sum should actually be off here, or this test proves nothing";
}

TEST(ComputeCpu, DoublePathMatchesItsOwnDefinition) {
    const CpuBackend backend;
    std::vector<double> xd{1.5, 2.5, 3.5};
    std::vector<double> yd{0.0, 0.0, 0.0};
    backend.saxpyD(xd, yd, 2.0);
    EXPECT_DOUBLE_EQ(yd[0], 3.0);
    EXPECT_DOUBLE_EQ(yd[1], 5.0);
    EXPECT_DOUBLE_EQ(yd[2], 7.0);

    EXPECT_DOUBLE_EQ(backend.sumD(xd), 7.5);
}

TEST(ComputeCpu, KindIsCpu) {
    const CpuBackend backend;
    EXPECT_EQ(backend.kind(), ysq::ComputeBackendKind::Cpu);
}

}  // namespace
