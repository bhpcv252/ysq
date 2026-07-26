#include <Math/Statistics.hpp>

#include <Math/Scalar.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace {

const std::vector<double> kSimple{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};

// --- Summation --------------------------------------------------------------

TEST(MathStatistics, SumMatchesTheObviousAnswerForOrdinaryData) {
    EXPECT_APPROX(ysq::sum(kSimple), 40.0);
    EXPECT_EQ(ysq::sum(std::vector<double>{}), 0.0);
    EXPECT_APPROX(ysq::sum(std::vector<double>{1.5}), 1.5);
}

TEST(MathStatistics, CompensatedSummationSurvivesWhatNaiveSummationDoesNot) {
    // The classic case. Adding 1 to 1e16 changes nothing at double precision,
    // so a naive total loses it entirely and comes back with exactly zero.
    const std::vector<double> cancelling{1e16, 1.0, -1e16};
    EXPECT_EQ(ysq::naiveSum(cancelling), 0.0);
    EXPECT_APPROX(ysq::sum(cancelling), 1.0);

    // And the case Kahan gets wrong but Neumaier does not: an addend far
    // larger than the running total.
    const std::vector<double> growing{1.0, 1e100, 1.0, -1e100};
    EXPECT_EQ(ysq::naiveSum(growing), 0.0);
    EXPECT_APPROX(ysq::sum(growing), 2.0);
}

TEST(MathStatistics, CompensatedSummationHoldsUpOverManyTerms) {
    // A long run of a value with no exact binary representation. This is the
    // shape of an energy accumulator: the error in the naive total grows with
    // the number of steps, and the compensated one does not.
    constexpr std::size_t count = 1'000'000;
    const std::vector<double> tenths(count, 0.1);
    const double exact = static_cast<double>(count) * 0.1;

    const double compensatedError = std::abs(ysq::sum(tenths) - exact);
    const double naiveError = std::abs(ysq::naiveSum(tenths) - exact);

    EXPECT_LT(compensatedError, 1e-9);
    EXPECT_GT(naiveError, compensatedError * 100.0)
        << "naive error " << naiveError << " vs compensated " << compensatedError;
}

// --- Central tendency and spread --------------------------------------------

TEST(MathStatistics, MeanVarianceAndDeviationMatchTheirDefinitions) {
    EXPECT_APPROX(ysq::mean(kSimple), 5.0);
    EXPECT_APPROX(ysq::variance(kSimple), 4.0);
    EXPECT_APPROX(ysq::standardDeviation(kSimple), 2.0);

    // Bessel's correction: n / (n - 1) times the population variance.
    EXPECT_APPROX(ysq::sampleVariance(kSimple), 4.0 * 8.0 / 7.0);
    EXPECT_APPROX(ysq::sampleStandardDeviation(kSimple),
                  std::sqrt(4.0 * 8.0 / 7.0));

    EXPECT_APPROX(ysq::variance(std::vector<double>{3.0, 3.0, 3.0}), 0.0);
}

TEST(MathStatistics, VarianceSurvivesALargeOffsetThatDefeatsTheShortcut) {
    // Variance as E[x^2] - E[x]^2 is one subtraction of two nearly equal large
    // numbers, and it loses everything once the mean dwarfs the spread. The
    // two-pass form here subtracts the mean first, so it does not.
    const double offset = 1e9;
    std::vector<double> shifted;
    for (const double value : {1.0, 2.0, 3.0, 4.0, 5.0}) {
        shifted.push_back(offset + value);
    }

    EXPECT_NEAR(ysq::variance(shifted), 2.0, 1e-6);

    double sumOfSquares = 0.0;
    double plainSum = 0.0;
    for (const double value : shifted) {
        sumOfSquares += value * value;
        plainSum += value;
    }
    const double count = static_cast<double>(shifted.size());
    const double average = plainSum / count;
    const double shortcut = sumOfSquares / count - average * average;
    EXPECT_GT(std::abs(shortcut - 2.0), 1e-3)
        << "the shortcut gave " << shortcut << ", which is why it is not used";
}

TEST(MathStatistics, InsufficientDataYieldsNotANumberRatherThanZero) {
    const std::vector<double> empty;
    const std::vector<double> single{1.0};

    EXPECT_TRUE(std::isnan(ysq::mean(empty)));
    EXPECT_TRUE(std::isnan(ysq::variance(empty)));
    EXPECT_TRUE(std::isnan(ysq::minimum(empty)));
    EXPECT_TRUE(std::isnan(ysq::maximum(empty)));
    EXPECT_TRUE(std::isnan(ysq::median(empty)));

    // One sample has a mean but no sample variance: there is nothing to
    // estimate a spread from.
    EXPECT_APPROX(ysq::mean(single), 1.0);
    EXPECT_APPROX(ysq::variance(single), 0.0);
    EXPECT_TRUE(std::isnan(ysq::sampleVariance(single)));
}

TEST(MathStatistics, ExtremesAndRange) {
    EXPECT_APPROX(ysq::minimum(kSimple), 2.0);
    EXPECT_APPROX(ysq::maximum(kSimple), 9.0);
    EXPECT_APPROX(ysq::range(kSimple), 7.0);
}

// --- Order statistics -------------------------------------------------------

TEST(MathStatistics, MedianHandlesBothParities) {
    EXPECT_APPROX(ysq::median(std::vector<double>{3.0, 1.0, 2.0}), 2.0);
    EXPECT_APPROX(ysq::median(std::vector<double>{4.0, 1.0, 3.0, 2.0}), 2.5);
    EXPECT_APPROX(ysq::median(std::vector<double>{5.0}), 5.0);
    // Order of the input does not matter.
    EXPECT_APPROX(ysq::median(kSimple), 4.5);
}

TEST(MathStatistics, QuantileInterpolatesBetweenOrderStatistics) {
    const std::vector<double> ramp{0.0, 1.0, 2.0, 3.0, 4.0};

    EXPECT_APPROX(ysq::quantile(ramp, 0.0), 0.0);
    EXPECT_APPROX(ysq::quantile(ramp, 1.0), 4.0);
    EXPECT_APPROX(ysq::quantile(ramp, 0.5), 2.0);
    // The type-7 definition: position is p * (n - 1), then linear between
    // neighbours. p = 0.3 lands at index 1.2.
    EXPECT_APPROX(ysq::quantile(ramp, 0.3), 1.2);
    EXPECT_APPROX(ysq::quantile(ramp, 0.25), 1.0);

    EXPECT_TRUE(std::isnan(ysq::quantile(ramp, -0.1)));
    EXPECT_TRUE(std::isnan(ysq::quantile(ramp, 1.1)));
}

// --- Two-variable statistics ------------------------------------------------

TEST(MathStatistics, CovarianceAndCorrelationOfExactRelationships) {
    const std::vector<double> xs{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> doubled;
    std::vector<double> negated;
    for (const double x : xs) {
        doubled.push_back(2.0 * x + 7.0);
        negated.push_back(-3.0 * x + 1.0);
    }

    EXPECT_APPROX(ysq::covariance(xs, xs), ysq::variance(xs));
    EXPECT_APPROX(ysq::correlation(xs, xs), 1.0);
    EXPECT_APPROX(ysq::correlation(xs, doubled), 1.0);
    EXPECT_APPROX(ysq::correlation(xs, negated), -1.0);
    EXPECT_APPROX(ysq::covariance(xs, doubled), 2.0 * ysq::variance(xs));

    // Symmetric.
    EXPECT_APPROX(ysq::covariance(xs, doubled), ysq::covariance(doubled, xs));

    // A constant variable has no correlation with anything, and saying so is
    // more honest than dividing by zero.
    const std::vector<double> flat{2.0, 2.0, 2.0, 2.0, 2.0};
    EXPECT_TRUE(std::isnan(ysq::correlation(xs, flat)));

    // Mismatched lengths.
    const std::vector<double> shorter{1.0, 2.0};
    EXPECT_TRUE(std::isnan(ysq::covariance(xs, shorter)));
    EXPECT_TRUE(std::isnan(ysq::correlation(xs, shorter)));
}

TEST(MathStatistics, LinearFitRecoversAnExactLine) {
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> ys;
    for (const double x : xs) {
        ys.push_back(3.5 * x - 2.25);
    }

    const auto fit = ysq::linearFit(xs, ys);
    EXPECT_APPROX(fit.slope, 3.5);
    EXPECT_APPROX(fit.intercept, -2.25);
    EXPECT_NEAR(fit.rSquared, 1.0, 1e-12);
}

TEST(MathStatistics, LinearFitOnScatteredDataSitsBetweenThePoints) {
    // A symmetric perturbation about a line: the fit has to come back to the
    // line it was perturbed from, with rSquared short of one.
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> ys{1.0, 1.0, 3.0, 3.0};  // about y = x + 1

    const auto fit = ysq::linearFit(xs, ys);
    EXPECT_APPROX(fit.slope, 0.8);
    EXPECT_APPROX(fit.intercept, 0.8);
    EXPECT_GT(fit.rSquared, 0.6);
    EXPECT_LT(fit.rSquared, 1.0);

    // Vertical data has no unique fit.
    const std::vector<double> constantX{2.0, 2.0, 2.0};
    EXPECT_TRUE(std::isnan(ysq::linearFit(constantX, xs).slope));
}

TEST(MathStatistics, ANotANumberInTheInputComesOutAsANotANumber) {
    // NaN has no ordering, so it is not something the order-based functions
    // can work around. Sorting past one is undefined behaviour rather than
    // merely inaccurate, and skipping it would have minimum() report a
    // smallest value that is not the smallest anything while mean() reported
    // NaN. Every one of these agrees instead.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> spoiled{3.0, 1.0, nan, 2.0, 5.0};

    EXPECT_TRUE(std::isnan(ysq::mean(spoiled)));
    EXPECT_TRUE(std::isnan(ysq::variance(spoiled)));
    EXPECT_TRUE(std::isnan(ysq::minimum(spoiled)));
    EXPECT_TRUE(std::isnan(ysq::maximum(spoiled)));
    EXPECT_TRUE(std::isnan(ysq::range(spoiled)));
    EXPECT_TRUE(std::isnan(ysq::median(spoiled)));
    EXPECT_TRUE(std::isnan(ysq::quantile(spoiled, 0.25)));

    // The online accumulator agrees with the batch functions, including on the
    // extremes, which std::min would otherwise have quietly kept clean.
    ysq::RunningStatistics<double> running;
    for (const double value : spoiled) {
        running.add(value);
    }
    EXPECT_TRUE(std::isnan(running.mean()));
    EXPECT_TRUE(std::isnan(running.minimum()));
    EXPECT_TRUE(std::isnan(running.maximum()));
    EXPECT_EQ(running.count(), spoiled.size()) << "the samples were still seen";

    // A histogram counts observations, and a count cannot be NaN, so the only
    // honest thing is to drop it rather than bin it somewhere.
    const auto bins = ysq::histogram(spoiled, 3, 0.0, 6.0);
    std::size_t total = 0;
    for (const std::size_t count : bins) {
        total += count;
    }
    EXPECT_EQ(total, 4u) << "four real observations, not five";

    // Infinities are values, not failures, and are handled by the ordinary
    // range test rather than being special-cased.
    const double inf = std::numeric_limits<double>::infinity();
    const auto withInfinities =
        ysq::histogram(std::vector<double>{1.0, inf, -inf}, 2, 0.0, 2.0);
    EXPECT_EQ(withInfinities[0] + withInfinities[1], 1u)
        << "both infinities fall outside the range and are dropped";
    EXPECT_APPROX(ysq::maximum(std::vector<double>{1.0, inf}), inf);
    EXPECT_APPROX(ysq::median(std::vector<double>{1.0, inf, 3.0}), 3.0);
}

// --- Histogram --------------------------------------------------------------

TEST(MathStatistics, HistogramCountsIntoEqualWidthBins) {
    const std::vector<double> values{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    const auto bins = ysq::histogram(values, 3, 0.0, 3.0);

    ASSERT_EQ(bins.size(), 3u);
    EXPECT_EQ(bins[0], 2u);  // 0.0, 0.5
    EXPECT_EQ(bins[1], 2u);  // 1.0, 1.5
    EXPECT_EQ(bins[2], 3u);  // 2.0, 2.5 and 3.0, the closed upper edge

    // Everything inside the range is counted exactly once.
    std::size_t total = 0;
    for (const std::size_t count : bins) {
        total += count;
    }
    EXPECT_EQ(total, values.size());

    // Outside the range is dropped, not clamped into an end bin.
    const auto narrow = ysq::histogram(values, 2, 1.0, 2.0);
    ASSERT_EQ(narrow.size(), 2u);
    EXPECT_EQ(narrow[0] + narrow[1], 3u);  // 1.0, 1.5, 2.0

    EXPECT_TRUE(ysq::histogram(values, 0, 0.0, 1.0).empty());
}

// --- Online accumulation ----------------------------------------------------

TEST(MathStatistics, RunningStatisticsAgreeWithTheTwoPassAnswer) {
    ysq::RunningStatistics<double> running;
    for (const double value : kSimple) {
        running.add(value);
    }

    EXPECT_EQ(running.count(), kSimple.size());
    EXPECT_NEAR(running.mean(), ysq::mean(kSimple), 1e-13);
    EXPECT_NEAR(running.variance(), ysq::variance(kSimple), 1e-13);
    EXPECT_NEAR(running.sampleVariance(), ysq::sampleVariance(kSimple), 1e-13);
    EXPECT_NEAR(running.standardDeviation(), ysq::standardDeviation(kSimple),
                1e-13);
    EXPECT_APPROX(running.minimum(), 2.0);
    EXPECT_APPROX(running.maximum(), 9.0);
    EXPECT_APPROX(running.range(), 7.0);
}

TEST(MathStatistics, RunningStatisticsStaysStableWithALargeOffset) {
    // Welford's update subtracts the running mean before squaring, so the same
    // offset that destroys the sum-of-squares shortcut costs it nothing.
    ysq::RunningStatistics<double> running;
    for (const double value : {1.0, 2.0, 3.0, 4.0, 5.0}) {
        running.add(1e9 + value);
    }
    EXPECT_NEAR(running.variance(), 2.0, 1e-6);
    EXPECT_NEAR(running.mean(), 1e9 + 3.0, 1e-6);
}

TEST(MathStatistics, RunningStatisticsMergeMatchesASinglePass) {
    const std::vector<double> first{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> second{10.0, 20.0, 30.0};

    ysq::RunningStatistics<double> a;
    ysq::RunningStatistics<double> b;
    ysq::RunningStatistics<double> combined;

    for (const double value : first) {
        a.add(value);
        combined.add(value);
    }
    for (const double value : second) {
        b.add(value);
        combined.add(value);
    }
    a.merge(b);

    EXPECT_EQ(a.count(), combined.count());
    EXPECT_NEAR(a.mean(), combined.mean(), 1e-13);
    EXPECT_NEAR(a.variance(), combined.variance(), 1e-12);
    EXPECT_APPROX(a.minimum(), combined.minimum());
    EXPECT_APPROX(a.maximum(), combined.maximum());

    // Merging an empty accumulator changes nothing, in either direction.
    const ysq::RunningStatistics<double> empty;
    ysq::RunningStatistics<double> unchanged = a;
    unchanged.merge(empty);
    EXPECT_EQ(unchanged.count(), a.count());
    EXPECT_APPROX(unchanged.mean(), a.mean());

    ysq::RunningStatistics<double> fromEmpty;
    fromEmpty.merge(a);
    EXPECT_EQ(fromEmpty.count(), a.count());
    EXPECT_APPROX(fromEmpty.mean(), a.mean());
}

TEST(MathStatistics, RunningStatisticsIsEmptyUntilFedAndAfterReset) {
    ysq::RunningStatistics<double> running;
    EXPECT_EQ(running.count(), 0u);
    EXPECT_TRUE(std::isnan(running.mean()));
    EXPECT_TRUE(std::isnan(running.variance()));

    running.add(1.0);
    EXPECT_TRUE(std::isnan(running.sampleVariance())) << "one sample is not two";

    running.reset();
    EXPECT_EQ(running.count(), 0u);
    EXPECT_TRUE(std::isnan(running.mean()));
}

TEST(MathStatistics, RunningRangeTellsBoundedDriftFromSecularDrift) {
    // The exact question a conservation test asks. Both series have a similar
    // standard deviation; only the range and the endpoints say which one is
    // going somewhere.
    ysq::RunningStatistics<double> oscillating;
    ysq::RunningStatistics<double> drifting;

    for (std::size_t i = 0; i < 1000; ++i) {
        const double t = static_cast<double>(i);
        oscillating.add(std::sin(t * 0.1));
        drifting.add(t * 0.002);
    }

    EXPECT_LT(oscillating.range(), 2.1);
    EXPECT_NEAR(oscillating.mean(), 0.0, 0.05);
    EXPECT_GT(drifting.range(), 1.9);
    EXPECT_NEAR(drifting.mean(), 1.0, 0.01)
        << "a drifting series has a mean far from its starting value";
}

// --- Ranges and precision ---------------------------------------------------

TEST(MathStatistics, AcceptsAnyContiguousRange) {
    // The reason the signatures take a range rather than a span: a span
    // parameter cannot deduce its element type, so every call would need the
    // conversion spelled out.
    const std::vector<double> asVector{1.0, 2.0, 3.0};
    const std::array<double, 3> asArray{1.0, 2.0, 3.0};
    const std::span<const double> asSpan{asArray};

    EXPECT_APPROX(ysq::mean(asVector), 2.0);
    EXPECT_APPROX(ysq::mean(asArray), 2.0);
    EXPECT_APPROX(ysq::mean(asSpan), 2.0);
}

TEST(MathStatistics, WorksAtSinglePrecision) {
    const std::vector<float> values{2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f};
    EXPECT_NEAR(ysq::mean(values), 5.0f, 1e-5f);
    EXPECT_NEAR(ysq::variance(values), 4.0f, 1e-5f);

    ysq::RunningStatistics<float> running;
    for (const float value : values) {
        running.add(value);
    }
    EXPECT_NEAR(running.mean(), 5.0f, 1e-5f);
    EXPECT_NEAR(running.variance(), 4.0f, 1e-5f);
}

}  // namespace
