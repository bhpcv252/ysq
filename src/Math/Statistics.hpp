#pragma once

#include <Math/Scalar.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <vector>

namespace ysq {

/// Summary statistics over a run.
///
/// The two things here that are not textbook are the ones that matter for this
/// project. Compensated summation is what lets an energy total stay meaningful
/// after a million steps, and the online accumulator is what lets a
/// conservation test watch a quantity without keeping every sample.
///
/// Insufficient data yields NaN rather than an exception or a zero: a variance
/// of "no samples" is not zero, and a NaN propagates into whatever it feeds
/// instead of quietly reading as a good result. The functions that can fail
/// for a structural reason rather than a statistical one, such as a spline
/// through unsorted data, return std::optional instead.
///
/// **A NaN in the input comes out as a NaN**, uniformly. That is not free:
/// NaN has no ordering, so `median` cannot sort past one without undefined
/// behaviour and `minimum` would otherwise skip it and report a smallest value
/// that is not the smallest anything. Both check. `histogram` drops NaN rather
/// than counting it, since a count is not a NaN and silently binning one would
/// make the totals lie.

namespace detail {

template <std::floating_point T>
inline constexpr T notANumber = std::numeric_limits<T>::quiet_NaN();

/// Any contiguous range of a floating-point type: vector, array, span, or a
/// subrange of one.
///
/// Taking a range rather than a std::span directly is purely about the call
/// site. A span parameter cannot deduce its element type from a vector, so
/// every caller would have to write the conversion out; the functions below
/// make a span internally and work on that.
template <class R>
concept FloatRange = std::ranges::contiguous_range<R> && std::ranges::sized_range<R> &&
                     std::floating_point<std::ranges::range_value_t<R>>;

template <class R>
using RangeValue = std::ranges::range_value_t<R>;

template <FloatRange R>
[[nodiscard]] constexpr std::span<const RangeValue<R>> viewOf(const R& range) {
    return {std::data(range), std::size(range)};
}

/// NaN compares false against everything, including itself, so it is not a
/// value any of the order-based functions here can work with. std::sort on a
/// range containing one is undefined, not merely wrong, and converting one to
/// an array index is undefined too. Every function that would hit either
/// checks first.
template <std::floating_point T>
[[nodiscard]] bool anyNotANumber(std::span<const T> values) {
    for (const T value : values) {
        if (std::isnan(value)) {
            return true;
        }
    }
    return false;
}

}  // namespace detail

/// Neumaier compensated summation.
///
/// Naive summation loses the low bits of every addend once the running total
/// grows large, and the error accumulates with the number of terms. This
/// carries the lost part forward in a second register, which costs about three
/// extra flops per element and bounds the error at roughly two epsilons
/// regardless of length.
///
/// Neumaier rather than plain Kahan because Kahan silently loses the
/// correction when an addend is larger than the running total, which is
/// exactly what happens when a total passes through zero.
template <detail::FloatRange R>
[[nodiscard]] auto sum(const R& range) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    T total{};
    T compensation{};

    for (const T value : values) {
        const T next = total + value;
        if (std::abs(total) >= std::abs(value)) {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    return total + compensation;
}

/// Plain left-to-right summation. Here so a test can show what the compensated
/// version is buying; production code should not need it.
template <detail::FloatRange R>
[[nodiscard]] auto naiveSum(const R& range) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    T total{};
    for (const T value : values) {
        total += value;
    }
    return total;
}

template <detail::FloatRange R>
[[nodiscard]] auto mean(const R& range) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    if (values.empty()) {
        return detail::notANumber<T>;
    }
    return sum(values) / static_cast<T>(values.size());
}

/// Divides by n. This is the variance of the data as a whole population.
template <detail::FloatRange R>
[[nodiscard]] auto variance(const R& range) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    if (values.empty()) {
        return detail::notANumber<T>;
    }
    const T centre = mean(values);
    T total{};
    for (const T value : values) {
        const T deviation = value - centre;
        total += deviation * deviation;
    }
    return total / static_cast<T>(values.size());
}

/// Divides by n - 1, Bessel's correction: the unbiased estimate of the
/// variance of a larger population the data was drawn from. Needs two samples.
template <detail::FloatRange R>
[[nodiscard]] auto sampleVariance(const R& range) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    if (values.size() < 2) {
        return detail::notANumber<T>;
    }
    const T centre = mean(values);
    T total{};
    for (const T value : values) {
        const T deviation = value - centre;
        total += deviation * deviation;
    }
    return total / static_cast<T>(values.size() - 1);
}

template <detail::FloatRange R>
[[nodiscard]] auto standardDeviation(const R& range) -> detail::RangeValue<R> {
    return std::sqrt(variance(range));
}

template <detail::FloatRange R>
[[nodiscard]] auto sampleStandardDeviation(const R& range) -> detail::RangeValue<R> {
    return std::sqrt(sampleVariance(range));
}

template <detail::FloatRange R>
[[nodiscard]] auto minimum(const R& range) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    if (values.empty() || detail::anyNotANumber(values)) {
        return detail::notANumber<T>;
    }
    return *std::min_element(values.begin(), values.end());
}

template <detail::FloatRange R>
[[nodiscard]] auto maximum(const R& range) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    if (values.empty() || detail::anyNotANumber(values)) {
        return detail::notANumber<T>;
    }
    return *std::max_element(values.begin(), values.end());
}

template <detail::FloatRange R>
[[nodiscard]] auto range(const R& values) -> detail::RangeValue<R> {
    return maximum(values) - minimum(values);
}

/// The quantile at p in [0, 1], interpolating linearly between the two order
/// statistics that bracket it. This is the definition R calls type 7 and the
/// one numpy uses by default, so a result here matches what an analysis script
/// would report.
///
/// Copies and sorts, so it is O(n log n) and allocates.
template <detail::FloatRange R>
[[nodiscard]] auto quantile(const R& range, detail::RangeValue<R> p)
    -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    if (values.empty() || !(T{0} <= p && p <= T{1}) || detail::anyNotANumber(values)) {
        return detail::notANumber<T>;
    }
    if (values.size() == 1) {
        return values.front();
    }

    std::vector<T> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());

    const T position = p * static_cast<T>(sorted.size() - 1);
    const T floorPosition = std::floor(position);
    const auto lower = static_cast<std::size_t>(floorPosition);
    if (lower + 1 == sorted.size()) {
        return sorted.back();
    }
    const T fraction = position - floorPosition;

    // Landing exactly on an order statistic is returned directly rather than
    // interpolated. The interpolation would be a * 1 + b * 0, and if b is an
    // infinity that last term is zero times infinity, which is NaN: the median
    // of {1, inf, 3} would come back undefined when it is plainly 3.
    if (fraction == T{0}) {
        return sorted[lower];
    }
    return sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower]);
}

template <detail::FloatRange R>
[[nodiscard]] auto median(const R& values) -> detail::RangeValue<R> {
    return quantile(values, detail::RangeValue<R>{0.5});
}

/// Divides by n, matching variance(). NaN unless both spans are the same
/// non-empty length.
template <detail::FloatRange R>
[[nodiscard]] auto covariance(const R& xRange, const R& yRange) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const auto xs = detail::viewOf(xRange);
    const auto ys = detail::viewOf(yRange);
    if (xs.empty() || xs.size() != ys.size()) {
        return detail::notANumber<T>;
    }
    const T meanX = mean(xs);
    const T meanY = mean(ys);
    T total{};
    for (std::size_t i = 0; i < xs.size(); ++i) {
        total += (xs[i] - meanX) * (ys[i] - meanY);
    }
    return total / static_cast<T>(xs.size());
}

/// Pearson's correlation coefficient, in [-1, 1]. NaN when either variable is
/// constant, since the correlation genuinely is not defined there.
template <detail::FloatRange R>
[[nodiscard]] auto correlation(const R& xs, const R& ys) -> detail::RangeValue<R> {
    using T = detail::RangeValue<R>;
    const T scale = standardDeviation(xs) * standardDeviation(ys);
    if (!(T{0} < scale)) {
        return detail::notANumber<T>;
    }
    return clamp(covariance(xs, ys) / scale, T{-1}, T{1});
}

template <std::floating_point T>
struct LinearFit {
    T slope{};
    T intercept{};
    /// The fraction of the variance in y the fit accounts for.
    T rSquared{};
};

/// Ordinary least squares of y on x. NaN throughout when x is constant, which
/// is the case with no unique answer.
template <detail::FloatRange R>
[[nodiscard]] LinearFit<detail::RangeValue<R>> linearFit(const R& xRange,
                                                         const R& yRange) {
    using T = detail::RangeValue<R>;
    const auto xs = detail::viewOf(xRange);
    const auto ys = detail::viewOf(yRange);
    const T varianceX = variance(xs);
    if (xs.size() < 2 || xs.size() != ys.size() || !(T{0} < varianceX)) {
        return {detail::notANumber<T>, detail::notANumber<T>, detail::notANumber<T>};
    }

    const T slope = covariance(xs, ys) / varianceX;
    const T intercept = mean(ys) - slope * mean(xs);
    const T r = correlation(xs, ys);
    return {slope, intercept, r * r};
}

/// Counts into `binCount` equal-width bins spanning [low, high]. Values
/// outside the range are dropped; `high` itself falls in the last bin rather
/// than off the end.
template <detail::FloatRange R>
[[nodiscard]] std::vector<std::size_t> histogram(const R& range, std::size_t binCount,
                                                 detail::RangeValue<R> low,
                                                 detail::RangeValue<R> high) {
    using T = detail::RangeValue<R>;
    const auto values = detail::viewOf(range);
    std::vector<std::size_t> bins(binCount, 0);
    if (binCount == 0 || !(low < high)) {
        return bins;
    }

    const T width = (high - low) / static_cast<T>(binCount);
    for (const T value : values) {
        // Written as the positive test rather than as its negation, so a NaN
        // fails it and is dropped. The negated form let one through, and
        // converting a NaN to an array index is undefined.
        if (!(low <= value && value <= high)) {
            continue;
        }
        auto index = static_cast<std::size_t>((value - low) / width);
        if (index >= binCount) {
            index = binCount - 1;  // the closed upper edge
        }
        ++bins[index];
    }
    return bins;
}

/// Welford's online algorithm: mean and variance in one pass, without keeping
/// the samples and without the catastrophic cancellation of the
/// sum-of-squares-minus-square-of-sum shortcut.
///
/// This is what watches a conserved quantity over a long integration. A run of
/// ten million steps cannot keep its energy history in memory, and the naive
/// formula would lose the drift signal in the noise long before then.
template <std::floating_point T>
class RunningStatistics {
public:
    void add(T value) {
        ++m_count;
        const T deviation = value - m_mean;
        m_mean += deviation / static_cast<T>(m_count);
        // The second deviation is taken against the updated mean. That is what
        // makes the update numerically stable rather than merely correct.
        m_sumSquaredDeviations += deviation * (value - m_mean);

        // std::min would keep the running extreme and drop the NaN, which
        // would leave minimum() reporting a smallest value while mean()
        // reported NaN. Seeding both with it instead makes them agree, and
        // once either is NaN std::min keeps it that way.
        if (std::isnan(value)) {
            m_minimum = value;
            m_maximum = value;
        } else {
            m_minimum = std::min(m_minimum, value);
            m_maximum = std::max(m_maximum, value);
        }
    }

    /// Chan's parallel combination, for statistics gathered separately.
    void merge(const RunningStatistics& other) {
        if (other.m_count == 0) {
            return;
        }
        if (m_count == 0) {
            *this = other;
            return;
        }

        const auto total = static_cast<T>(m_count + other.m_count);
        const T deviation = other.m_mean - m_mean;
        const T weighted = deviation * static_cast<T>(other.m_count) / total;

        m_sumSquaredDeviations +=
            other.m_sumSquaredDeviations + deviation * weighted * static_cast<T>(m_count);
        m_mean += weighted;
        m_count += other.m_count;
        m_minimum = std::min(m_minimum, other.m_minimum);
        m_maximum = std::max(m_maximum, other.m_maximum);
    }

    void reset() { *this = RunningStatistics{}; }

    [[nodiscard]] std::size_t count() const noexcept { return m_count; }

    [[nodiscard]] T mean() const noexcept {
        return (m_count == 0) ? detail::notANumber<T> : m_mean;
    }

    [[nodiscard]] T variance() const noexcept {
        return (m_count == 0) ? detail::notANumber<T>
                              : m_sumSquaredDeviations / static_cast<T>(m_count);
    }

    [[nodiscard]] T sampleVariance() const noexcept {
        return (m_count < 2) ? detail::notANumber<T>
                             : m_sumSquaredDeviations / static_cast<T>(m_count - 1);
    }

    [[nodiscard]] T standardDeviation() const { return std::sqrt(variance()); }

    [[nodiscard]] T sampleStandardDeviation() const {
        return std::sqrt(sampleVariance());
    }

    [[nodiscard]] T minimum() const noexcept {
        return (m_count == 0) ? detail::notANumber<T> : m_minimum;
    }

    [[nodiscard]] T maximum() const noexcept {
        return (m_count == 0) ? detail::notANumber<T> : m_maximum;
    }

    /// Peak-to-peak spread. For a conserved quantity this is the number that
    /// says whether the error is bounded or growing.
    [[nodiscard]] T range() const noexcept { return maximum() - minimum(); }

private:
    std::size_t m_count = 0;
    T m_mean{};
    T m_sumSquaredDeviations{};
    T m_minimum = std::numeric_limits<T>::infinity();
    T m_maximum = -std::numeric_limits<T>::infinity();
};

}  // namespace ysq
