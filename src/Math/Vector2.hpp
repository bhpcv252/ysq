#pragma once

#include <Math/Scalar.hpp>

#include <cassert>
#include <cstddef>
#include <optional>

namespace ysq {

/// Two-component vector.
///
/// An aggregate with public members, so `Vector2<double>{1.0, 2.0}` works and
/// the layout is exactly two Ts with nothing added. Default member initializers
/// zero it, which costs nothing measurable and removes a whole class of
/// uninitialised-math bug; it stays standard layout and trivially copyable, so
/// an array of these uploads to a GPU buffer as-is.
template <Numeric T>
struct Vector2 {
    using value_type = T;

    T x{};
    T y{};

    [[nodiscard]] static constexpr std::size_t size() noexcept { return 2; }

    /// Index must be less than size(). Out of range yields the last component
    /// rather than reading past the object.
    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? x : y;
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? x : y;
    }

    [[nodiscard]] static constexpr Vector2 zero() noexcept { return {}; }
    [[nodiscard]] static constexpr Vector2 splat(T value) noexcept {
        return {value, value};
    }
    [[nodiscard]] static constexpr Vector2 unitX() noexcept { return {T{1}, T{0}}; }
    [[nodiscard]] static constexpr Vector2 unitY() noexcept { return {T{0}, T{1}}; }

    constexpr Vector2& operator+=(const Vector2& other) noexcept {
        x += other.x;
        y += other.y;
        return *this;
    }

    constexpr Vector2& operator-=(const Vector2& other) noexcept {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    constexpr Vector2& operator*=(T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    constexpr Vector2& operator/=(T scalar) noexcept {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    // Hidden friends: found only by ADL, so they never widen overload
    // resolution for unrelated types and the error messages stay short.

    [[nodiscard]] friend constexpr Vector2 operator+(const Vector2& v) noexcept {
        return v;
    }

    [[nodiscard]] friend constexpr Vector2 operator-(const Vector2& v) noexcept {
        return {-v.x, -v.y};
    }

    [[nodiscard]] friend constexpr Vector2 operator+(const Vector2& a,
                                                     const Vector2& b) noexcept {
        return {a.x + b.x, a.y + b.y};
    }

    [[nodiscard]] friend constexpr Vector2 operator-(const Vector2& a,
                                                     const Vector2& b) noexcept {
        return {a.x - b.x, a.y - b.y};
    }

    [[nodiscard]] friend constexpr Vector2 operator*(const Vector2& v,
                                                     T scalar) noexcept {
        return {v.x * scalar, v.y * scalar};
    }

    [[nodiscard]] friend constexpr Vector2 operator*(T scalar,
                                                     const Vector2& v) noexcept {
        return {scalar * v.x, scalar * v.y};
    }

    /// Divides rather than multiplying by a reciprocal. One rounding step per
    /// component instead of two matters more here than the saved division.
    [[nodiscard]] friend constexpr Vector2 operator/(const Vector2& v,
                                                     T scalar) noexcept {
        return {v.x / scalar, v.y / scalar};
    }

    /// Exact, component by component. Use approxEqual for anything that has
    /// been through arithmetic.
    [[nodiscard]] friend constexpr bool operator==(const Vector2&,
                                                   const Vector2&) = default;
};

template <Numeric T>
[[nodiscard]] constexpr T dot(const Vector2<T>& a, const Vector2<T>& b) noexcept {
    return a.x * b.x + a.y * b.y;
}

/// The 2D cross product: the z component of the 3D cross of the two vectors
/// lifted into the plane. Also the signed area of the parallelogram they span,
/// and the perp-dot product of a with b.
template <Numeric T>
[[nodiscard]] constexpr T cross(const Vector2<T>& a, const Vector2<T>& b) noexcept {
    return a.x * b.y - a.y * b.x;
}

/// Rotated a quarter turn counter-clockwise.
template <Numeric T>
[[nodiscard]] constexpr Vector2<T> perpendicular(const Vector2<T>& v) noexcept {
    return {-v.y, v.x};
}

template <Numeric T>
[[nodiscard]] constexpr T lengthSquared(const Vector2<T>& v) noexcept {
    return dot(v, v);
}

template <Numeric T>
[[nodiscard]] auto length(const Vector2<T>& v) {
    return detail::sqrtOf(lengthSquared(v));
}

/// Undefined direction at zero length: every component comes back NaN, which
/// propagates rather than quietly returning a wrong unit vector. Use
/// tryNormalized where the input can legitimately be zero.
template <Numeric T>
[[nodiscard]] Vector2<T> normalized(const Vector2<T>& v) {
    return v / length(v);
}

/// nullopt for a zero or non-finite vector, and also for one whose squared
/// length overflows: at double precision that is a component beyond about
/// 1.3e154, or an underflow below about 1.5e-162. The vector is normalisable
/// in principle there, but not without rescaling first, and reporting that is
/// preferable to returning a zero vector as if it were a valid result.
///
/// Complex::abs has no such limit because it goes through hypot. Doing the
/// same here would cost a hypot on every normalisation on the integration
/// inner path, to widen a range that no physical quantity approaches.
template <Numeric T>
[[nodiscard]] std::optional<Vector2<T>> tryNormalized(const Vector2<T>& v) {
    const T len = length(v);
    // Negated so a NaN fails too, and checked for finiteness so an overflow
    // does not divide through to zero and report success.
    if (!(T{0} < len) || !detail::isFiniteValue(len)) {
        return std::nullopt;
    }
    return v / len;
}

template <Numeric T>
[[nodiscard]] constexpr T distanceSquared(const Vector2<T>& a,
                                          const Vector2<T>& b) noexcept {
    return lengthSquared(a - b);
}

template <Numeric T>
[[nodiscard]] auto distance(const Vector2<T>& a, const Vector2<T>& b) {
    return length(a - b);
}

/// Exact at both endpoints, which the cheaper `a + (b - a) * t` is not.
/// Extrapolates outside [0, 1].
template <Numeric T>
[[nodiscard]] constexpr Vector2<T> lerp(const Vector2<T>& a, const Vector2<T>& b,
                                        T t) noexcept {
    return a * (T{1} - t) + b * t;
}

/// Component of a along onto. Undefined for a zero `onto`.
template <Numeric T>
[[nodiscard]] constexpr Vector2<T> project(const Vector2<T>& a,
                                           const Vector2<T>& onto) noexcept {
    return onto * (dot(a, onto) / lengthSquared(onto));
}

/// The part of a orthogonal to `from`. project + reject reconstructs a.
template <Numeric T>
[[nodiscard]] constexpr Vector2<T> reject(const Vector2<T>& a,
                                          const Vector2<T>& from) noexcept {
    return a - project(a, from);
}

/// Mirror of v in the plane whose normal is n. n must be a unit vector.
template <Numeric T>
[[nodiscard]] constexpr Vector2<T> reflect(const Vector2<T>& v,
                                           const Vector2<T>& n) noexcept {
    return v - n * (T{2} * dot(v, n));
}

template <Numeric T>
[[nodiscard]] constexpr Vector2<T> hadamard(const Vector2<T>& a,
                                            const Vector2<T>& b) noexcept {
    return {a.x * b.x, a.y * b.y};
}

template <Numeric T>
[[nodiscard]] constexpr Vector2<T> min(const Vector2<T>& a,
                                       const Vector2<T>& b) noexcept {
    return {(a.x < b.x) ? a.x : b.x, (a.y < b.y) ? a.y : b.y};
}

template <Numeric T>
[[nodiscard]] constexpr Vector2<T> max(const Vector2<T>& a,
                                       const Vector2<T>& b) noexcept {
    return {(a.x < b.x) ? b.x : a.x, (a.y < b.y) ? b.y : a.y};
}

template <Numeric T>
[[nodiscard]] Vector2<T> abs(const Vector2<T>& v) {
    return {detail::absOf(v.x), detail::absOf(v.y)};
}

/// Signed angle from a to b, in (-pi, pi], counter-clockwise positive.
///
/// atan2 of the cross over the dot, not acos of the normalised dot. acos loses
/// most of its significant digits for nearly parallel or nearly antiparallel
/// inputs, where its argument sits on a flat part of the curve; the atan2 form
/// is accurate across the whole range.
template <Numeric T>
[[nodiscard]] auto angleBetween(const Vector2<T>& a, const Vector2<T>& b) {
    return detail::atan2Of(cross(a, b), dot(a, b));
}

using Vec2 = Vector2<double>;
using Vec2f = Vector2<float>;

}  // namespace ysq
