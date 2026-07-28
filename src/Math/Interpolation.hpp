#pragma once

#include <Math/Scalar.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace ysq {

/// Interpolation, from a straight line up to a natural cubic spline.
///
/// The curve functions are generic in the value type and take their parameter
/// separately, so they work on scalars and on vectors alike: catmullRom over
/// Vector3 traces a path through control points, and over double it smooths a
/// scalar table. Only the easing functions are scalar-only, since they are
/// shaping a parameter rather than a quantity.

/// Exact at both endpoints, which the cheaper `a + (b - a) * t` is not.
/// Extrapolates outside [0, 1]. Vectors have their own overload in the vector
/// headers; this one is for scalars.
template <Numeric T>
[[nodiscard]] constexpr T lerp(T a, T b, T t) noexcept {
    return a * (T{1} - t) + b * t;
}

/// Where `value` sits between a and b, as the t that lerp would need.
/// Undefined for a == b.
template <Numeric T>
[[nodiscard]] constexpr T inverseLerp(T a, T b, T value) noexcept {
    return (value - a) / (b - a);
}

template <Numeric T>
[[nodiscard]] constexpr T remap(T value, T fromLow, T fromHigh, T toLow,
                                T toHigh) noexcept {
    return lerp(toLow, toHigh, inverseLerp(fromLow, fromHigh, value));
}

/// The cubic 3t^2 - 2t^3 on a clamped, normalised parameter. Its first
/// derivative vanishes at both ends, so a value eased through it starts and
/// stops smoothly.
template <std::floating_point T>
[[nodiscard]] constexpr T smoothstep(T edgeLow, T edgeHigh, T at) noexcept {
    const T t = clamp(inverseLerp(edgeLow, edgeHigh, at), T{0}, T{1});
    return t * t * (T{3} - T{2} * t);
}

/// The quintic whose first and second derivatives both vanish at the ends.
/// Worth the extra terms wherever the acceleration is visible, such as a
/// camera move.
template <std::floating_point T>
[[nodiscard]] constexpr T smootherstep(T edgeLow, T edgeHigh, T at) noexcept {
    const T t = clamp(inverseLerp(edgeLow, edgeHigh, at), T{0}, T{1});
    return t * t * t * (t * (T{6} * t - T{15}) + T{10});
}

/// Values are named for the corner they sit on: v10 is high in x, low in y.
template <class V, Numeric T>
[[nodiscard]] constexpr V bilinear(const V& v00, const V& v10, const V& v01, const V& v11,
                                   T tx, T ty) noexcept {
    const V low = v00 * (T{1} - tx) + v10 * tx;
    const V high = v01 * (T{1} - tx) + v11 * tx;
    return low * (T{1} - ty) + high * ty;
}

template <class V, Numeric T>
[[nodiscard]] constexpr V
trilinear(const V& v000, const V& v100, const V& v010, const V& v110, const V& v001,
          const V& v101, const V& v011, const V& v111, T tx, T ty, T tz) noexcept {
    const V low = bilinear(v000, v100, v010, v110, tx, ty);
    const V high = bilinear(v001, v101, v011, v111, tx, ty);
    return low * (T{1} - tz) + high * tz;
}

/// The cubic through p0 at t = 0 and p1 at t = 1, leaving with tangent m0 and
/// arriving with tangent m1.
template <class V, Numeric T>
[[nodiscard]] constexpr V cubicHermite(const V& p0, const V& m0, const V& p1, const V& m1,
                                       T t) noexcept {
    const T t2 = t * t;
    const T t3 = t2 * t;
    return p0 * (T{2} * t3 - T{3} * t2 + T{1}) + m0 * (t3 - T{2} * t2 + t) +
           p1 * (T{-2} * t3 + T{3} * t2) + m1 * (t3 - t2);
}

/// Interpolates between p1 and p2, using p0 and p3 only to pick the tangents.
/// It passes exactly through every control point, which is what makes it the
/// usual choice for a path through measured positions.
template <class V, Numeric T>
[[nodiscard]] constexpr V catmullRom(const V& p0, const V& p1, const V& p2, const V& p3,
                                     T t) noexcept {
    return cubicHermite(p1, (p2 - p0) / T{2}, p2, (p3 - p1) / T{2}, t);
}

/// The Bezier cubic. Unlike Catmull-Rom it passes through only its first and
/// last control points; the middle two pull at it.
template <class V, Numeric T>
[[nodiscard]] constexpr V cubicBezier(const V& p0, const V& p1, const V& p2, const V& p3,
                                      T t) noexcept {
    const T u = T{1} - t;
    return p0 * (u * u * u) + p1 * (T{3} * u * u * t) + p2 * (T{3} * u * t * t) +
           p3 * (t * t * t);
}

namespace detail {

/// Index of the interval containing `at`, for a strictly increasing table.
/// Clamped to a valid interval at both ends.
template <std::floating_point T>
[[nodiscard]] std::size_t intervalFor(std::span<const T> knots, T at) {
    const auto upper = std::upper_bound(knots.begin(), knots.end(), at);
    if (upper == knots.begin()) {
        return 0;
    }
    const auto index = static_cast<std::size_t>(upper - knots.begin()) - 1;
    return std::min(index, knots.size() - 2);
}

}  // namespace detail

/// Piecewise linear lookup. `xs` must be strictly increasing, which is a
/// precondition rather than a check: the whole point of the binary search is
/// not to touch every element.
///
/// Outside the table the endpoint value is held rather than extrapolated,
/// since a table usually encodes a measured range and a linear continuation
/// past it would be an invention. nullopt if the two ranges do not describe a
/// table of at least two points.
///
/// Takes contiguous ranges rather than spans so a vector or array can be
/// passed directly; a span parameter cannot deduce its element type. Spans
/// still work, since a span is itself such a range. CubicSpline::natural has
/// no such trouble, because its element type comes from the class rather than
/// from deduction.
template <class R>
    requires std::ranges::contiguous_range<R> && std::ranges::sized_range<R> &&
             std::floating_point<std::ranges::range_value_t<R>>
[[nodiscard]] auto interpolateTable(const R& xRange, const R& yRange,
                                    std::ranges::range_value_t<R> at)
    -> std::optional<std::ranges::range_value_t<R>> {
    using T = std::ranges::range_value_t<R>;
    const std::span<const T> xs{std::data(xRange), std::size(xRange)};
    const std::span<const T> ys{std::data(yRange), std::size(yRange)};

    if (xs.size() < 2 || xs.size() != ys.size()) {
        return std::nullopt;
    }
    if (at <= xs.front()) {
        return ys.front();
    }
    if (xs.back() <= at) {
        return ys.back();
    }

    const std::size_t i = detail::intervalFor(xs, at);
    const T t = (at - xs[i]) / (xs[i + 1] - xs[i]);
    return lerp(ys[i], ys[i + 1], t);
}

/// A natural cubic spline through tabulated points.
///
/// C2 continuous, with the second derivative pinned to zero at both ends,
/// which is what "natural" names. That end condition is why the spline
/// reproduces a cubic exactly only in the interior: a cubic generally has a
/// non-zero second derivative at the boundary and the spline is not allowed
/// one.
///
/// Built once and evaluated many times. Construction solves a tridiagonal
/// system in O(n); each evaluation is a binary search and a few multiplies.
template <std::floating_point T>
class CubicSpline {
public:
    /// nullopt unless the two spans are the same length, hold at least three
    /// points, and xs is strictly increasing. Three because two points have no
    /// interior knot and the spline degenerates to the straight line that
    /// interpolateTable already gives.
    [[nodiscard]] static std::optional<CubicSpline> natural(std::span<const T> xs,
                                                            std::span<const T> ys) {
        if (xs.size() < 3 || xs.size() != ys.size()) {
            return std::nullopt;
        }
        for (std::size_t i = 1; i < xs.size(); ++i) {
            if (!(xs[i - 1] < xs[i])) {
                return std::nullopt;
            }
        }

        CubicSpline spline;
        spline.m_x.assign(xs.begin(), xs.end());
        spline.m_y.assign(ys.begin(), ys.end());
        spline.solve();
        return spline;
    }

    /// Held flat outside the table, matching interpolateTable. A cubic
    /// continued past its last knot diverges fast, and silently.
    [[nodiscard]] T operator()(T at) const {
        if (at <= m_x.front()) {
            return m_y.front();
        }
        if (m_x.back() <= at) {
            return m_y.back();
        }

        const std::size_t i = detail::intervalFor(std::span<const T>(m_x), at);
        const T width = m_x[i + 1] - m_x[i];
        const T left = (m_x[i + 1] - at) / width;
        const T right = (at - m_x[i]) / width;

        return left * m_y[i] + right * m_y[i + 1] +
               ((left * left * left - left) * m_secondDerivative[i] +
                (right * right * right - right) * m_secondDerivative[i + 1]) *
                   (width * width) / T{6};
    }

    /// The slope of the spline. Zero outside the table, where the value is
    /// held flat.
    [[nodiscard]] T derivative(T at) const {
        if (at <= m_x.front() || m_x.back() <= at) {
            return T{0};
        }

        const std::size_t i = detail::intervalFor(std::span<const T>(m_x), at);
        const T width = m_x[i + 1] - m_x[i];
        const T left = (m_x[i + 1] - at) / width;
        const T right = (at - m_x[i]) / width;

        return (m_y[i + 1] - m_y[i]) / width +
               ((T{1} - T{3} * left * left) * m_secondDerivative[i] +
                (T{3} * right * right - T{1}) * m_secondDerivative[i + 1]) *
                   width / T{6};
    }

    [[nodiscard]] std::size_t size() const noexcept { return m_x.size(); }
    [[nodiscard]] T lowerBound() const noexcept { return m_x.front(); }
    [[nodiscard]] T upperBound() const noexcept { return m_x.back(); }

private:
    /// Thomas algorithm on the tridiagonal system for the second derivatives,
    /// with both ends pinned to zero.
    void solve() {
        const std::size_t n = m_x.size();
        m_secondDerivative.assign(n, T{0});
        std::vector<T> scratch(n, T{0});

        for (std::size_t i = 1; i + 1 < n; ++i) {
            const T ratio = (m_x[i] - m_x[i - 1]) / (m_x[i + 1] - m_x[i - 1]);
            const T pivot = ratio * m_secondDerivative[i - 1] + T{2};
            m_secondDerivative[i] = (ratio - T{1}) / pivot;

            const T slopeRight = (m_y[i + 1] - m_y[i]) / (m_x[i + 1] - m_x[i]);
            const T slopeLeft = (m_y[i] - m_y[i - 1]) / (m_x[i] - m_x[i - 1]);
            scratch[i] = (T{6} * (slopeRight - slopeLeft) / (m_x[i + 1] - m_x[i - 1]) -
                          ratio * scratch[i - 1]) /
                         pivot;
        }

        for (std::size_t k = n - 1; k-- > 0;) {
            m_secondDerivative[k] =
                m_secondDerivative[k] * m_secondDerivative[k + 1] + scratch[k];
        }
        m_secondDerivative.front() = T{0};
        m_secondDerivative.back() = T{0};
    }

    std::vector<T> m_x;
    std::vector<T> m_y;
    std::vector<T> m_secondDerivative;
};

}  // namespace ysq
