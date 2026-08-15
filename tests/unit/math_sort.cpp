#include <Math/Sort.hpp>

#include <Compute/CPU/CpuBackend.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <vector>

TEST(MathSort, SortInPlaceSortsAKnownUnorderedSequence) {
    std::vector<double> values{5.0, 3.0, 8.0, 1.0, 9.0, 2.0};

    ysq::sortInPlace<double>(values);

    const std::vector<double> expected{1.0, 2.0, 3.0, 5.0, 8.0, 9.0};
    EXPECT_EQ(values, expected);
}

TEST(MathSort, SortInPlaceOnEmptyOrSingleElementDoesNothing) {
    std::vector<double> empty;
    ysq::sortInPlace<double>(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<double> single{7.0};
    ysq::sortInPlace<double>(single);
    EXPECT_EQ(single, (std::vector<double>{7.0}));
}

TEST(MathSort, SortedDoesNotModifyItsInput) {
    const std::vector<double> values{5.0, 3.0, 8.0, 1.0};

    const std::vector<double> result = ysq::sorted<double>(values);

    EXPECT_EQ(values, (std::vector<double>{5.0, 3.0, 8.0, 1.0}));
    EXPECT_EQ(result, (std::vector<double>{1.0, 3.0, 5.0, 8.0}));
}

TEST(MathSort, KthSmallestMatchesTheSortedOrderStatistic) {
    const std::vector<double> values{5.0, 3.0, 8.0, 1.0, 9.0, 2.0};
    const std::vector<double> expectedSorted{1.0, 2.0, 3.0, 5.0, 8.0, 9.0};

    for (std::size_t k = 0; k < values.size(); ++k) {
        EXPECT_DOUBLE_EQ(ysq::kthSmallest<double>(values, k), expectedSorted[k])
            << "k = " << k;
    }
}

TEST(MathSort, KthLargestMatchesTheSortedOrderStatisticFromTheOtherEnd) {
    const std::vector<double> values{5.0, 3.0, 8.0, 1.0, 9.0, 2.0};
    const std::vector<double> expectedSorted{1.0, 2.0, 3.0, 5.0, 8.0, 9.0};

    for (std::size_t k = 0; k < values.size(); ++k) {
        EXPECT_DOUBLE_EQ(ysq::kthLargest<double>(values, k),
                         expectedSorted[expectedSorted.size() - 1 - k])
            << "k = " << k;
    }
}

TEST(MathSort, KthSmallestAndKthLargestAgreeAtTheMedianForAnOddCount) {
    const std::vector<double> values{9.0, 1.0, 5.0, 3.0, 7.0};  // sorted: 1 3 5 7 9

    EXPECT_DOUBLE_EQ(ysq::kthSmallest<double>(values, 2), 5.0);
    EXPECT_DOUBLE_EQ(ysq::kthLargest<double>(values, 2), 5.0);
}

TEST(MathSort, SortInPlaceAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // Above Sort.hpp's own GPU dispatch threshold and T = float (the only
    // type that ever dispatches). ysq::CpuBackend is called directly as an
    // independent reference (its own agreement with every GPU backend is
    // already covered by tests/integration/compute_backends_agree.cpp;
    // this test only checks that sortInPlace's dispatch is wired
    // correctly).
    // Above kSortGpuDispatchThreshold (131072; measured by
    // benchmarks/compute_thresholds.cpp), and not a power of two.
    constexpr std::size_t n = 131073;
    std::vector<float> values(n);
    std::vector<float> reference(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float v = static_cast<float>((i * 2654435761u) % 1000007u) * 0.001f;
        values[i] = v;
        reference[i] = v;
    }

    ysq::sortInPlace<float>(values);

    const ysq::CpuBackend cpu;
    cpu.sortAscending(reference);

    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        EXPECT_FLOAT_EQ(values[i], reference[i]) << "element " << i;
    }
}
