#pragma once

#include <Math/Scalar.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>

#include <cassert>
#include <cstddef>
#include <optional>

namespace ysq {

/// Four-component vector: homogeneous coordinates for the renderer, and
/// four-vectors for the spacetime code, where the components are (t, x, y, z)
/// and the metric rather than this type decides what the inner product means.
///
/// There is no cross product. The 3D cross is specific to three dimensions;
/// the four-dimensional analogue is a wedge product and belongs in Tensor.
template <Numeric T>
struct Vector4 {
    using value_type = T;

    T x{};
    T y{};
    T z{};
    T w{};

    [[nodiscard]] static constexpr std::size_t size() noexcept { return 4; }

    /// Index must be less than size(). Out of range yields the last component
    /// rather than reading past the object.
    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? x : ((index == 1) ? y : ((index == 2) ? z : w));
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? x : ((index == 1) ? y : ((index == 2) ? z : w));
    }

    [[nodiscard]] constexpr Vector2<T> xy() const noexcept { return {x, y}; }
    [[nodiscard]] constexpr Vector3<T> xyz() const noexcept { return {x, y, z}; }

    [[nodiscard]] static constexpr Vector4 zero() noexcept { return {}; }
    [[nodiscard]] static constexpr Vector4 splat(T value) noexcept {
        return {value, value, value, value};
    }
    [[nodiscard]] static constexpr Vector4 unitX() noexcept {
        return {T{1}, T{0}, T{0}, T{0}};
    }
    [[nodiscard]] static constexpr Vector4 unitY() noexcept {
        return {T{0}, T{1}, T{0}, T{0}};
    }
    [[nodiscard]] static constexpr Vector4 unitZ() noexcept {
        return {T{0}, T{0}, T{1}, T{0}};
    }
    [[nodiscard]] static constexpr Vector4 unitW() noexcept {
        return {T{0}, T{0}, T{0}, T{1}};
    }

    /// Homogeneous position: w = 1, so a Matrix4 translation applies to it.
    [[nodiscard]] static constexpr Vector4 point(const Vector3<T>& v) noexcept {
        return {v.x, v.y, v.z, T{1}};
    }

    /// Homogeneous direction: w = 0, so a Matrix4 translation does not.
    [[nodiscard]] static constexpr Vector4 direction(const Vector3<T>& v) noexcept {
        return {v.x, v.y, v.z, T{0}};
    }

    constexpr Vector4& operator+=(const Vector4& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }

    constexpr Vector4& operator-=(const Vector4& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }

    constexpr Vector4& operator*=(T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }

    constexpr Vector4& operator/=(T scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        return *this;
    }

    [[nodiscard]] friend constexpr Vector4 operator+(const Vector4& v) noexcept {
        return v;
    }

    [[nodiscard]] friend constexpr Vector4 operator-(const Vector4& v) noexcept {
        return {-v.x, -v.y, -v.z, -v.w};
    }

    [[nodiscard]] friend constexpr Vector4 operator+(const Vector4& a,
                                                     const Vector4& b) noexcept {
        return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
    }

    [[nodiscard]] friend constexpr Vector4 operator-(const Vector4& a,
                                                     const Vector4& b) noexcept {
        return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
    }

    [[nodiscard]] friend constexpr Vector4 operator*(const Vector4& v,
                                                     T scalar) noexcept {
        return {v.x * scalar, v.y * scalar, v.z * scalar, v.w * scalar};
    }

    [[nodiscard]] friend constexpr Vector4 operator*(T scalar,
                                                     const Vector4& v) noexcept {
        return {scalar * v.x, scalar * v.y, scalar * v.z, scalar * v.w};
    }

    /// Divides rather than multiplying by a reciprocal, for the same reason as
    /// Vector2: one rounding step per component instead of two.
    [[nodiscard]] friend constexpr Vector4 operator/(const Vector4& v,
                                                     T scalar) noexcept {
        return {v.x / scalar, v.y / scalar, v.z / scalar, v.w / scalar};
    }

    [[nodiscard]] friend constexpr bool operator==(const Vector4&,
                                                   const Vector4&) = default;
};

/// The Euclidean inner product. The spacetime inner product is not this; it
/// contracts with a metric and lives in Physics/Spacetime.
template <Numeric T>
[[nodiscard]] constexpr T dot(const Vector4<T>& a, const Vector4<T>& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <Numeric T>
[[nodiscard]] constexpr T lengthSquared(const Vector4<T>& v) noexcept {
    return dot(v, v);
}

template <Numeric T>
[[nodiscard]] auto length(const Vector4<T>& v) {
    return detail::sqrtOf(lengthSquared(v));
}

/// Undefined direction at zero length: every component comes back NaN, which
/// propagates rather than quietly returning a wrong unit vector. Use
/// tryNormalized where the input can legitimately be zero.
template <Numeric T>
[[nodiscard]] Vector4<T> normalized(const Vector4<T>& v) {
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
[[nodiscard]] std::optional<Vector4<T>> tryNormalized(const Vector4<T>& v) {
    const T len = length(v);
    // Negated so a NaN fails too, and checked for finiteness so an overflow
    // does not divide through to zero and report success.
    if (!(T{0} < len) || !detail::isFiniteValue(len)) {
        return std::nullopt;
    }
    return v / len;
}

template <Numeric T>
[[nodiscard]] constexpr T distanceSquared(const Vector4<T>& a,
                                          const Vector4<T>& b) noexcept {
    return lengthSquared(a - b);
}

template <Numeric T>
[[nodiscard]] auto distance(const Vector4<T>& a, const Vector4<T>& b) {
    return length(a - b);
}

/// Exact at both endpoints, which the cheaper `a + (b - a) * t` is not.
/// Extrapolates outside [0, 1].
template <Numeric T>
[[nodiscard]] constexpr Vector4<T> lerp(const Vector4<T>& a, const Vector4<T>& b,
                                        T t) noexcept {
    return a * (T{1} - t) + b * t;
}

/// Component of a along onto. Undefined for a zero `onto`.
template <Numeric T>
[[nodiscard]] constexpr Vector4<T> project(const Vector4<T>& a,
                                           const Vector4<T>& onto) noexcept {
    return onto * (dot(a, onto) / lengthSquared(onto));
}

/// The part of a orthogonal to `from`. project + reject reconstructs a.
template <Numeric T>
[[nodiscard]] constexpr Vector4<T> reject(const Vector4<T>& a,
                                          const Vector4<T>& from) noexcept {
    return a - project(a, from);
}

template <Numeric T>
[[nodiscard]] constexpr Vector4<T> hadamard(const Vector4<T>& a,
                                            const Vector4<T>& b) noexcept {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <Numeric T>
[[nodiscard]] constexpr Vector4<T> min(const Vector4<T>& a,
                                       const Vector4<T>& b) noexcept {
    return {(a.x < b.x) ? a.x : b.x, (a.y < b.y) ? a.y : b.y, (a.z < b.z) ? a.z : b.z,
            (a.w < b.w) ? a.w : b.w};
}

template <Numeric T>
[[nodiscard]] constexpr Vector4<T> max(const Vector4<T>& a,
                                       const Vector4<T>& b) noexcept {
    return {(a.x < b.x) ? b.x : a.x, (a.y < b.y) ? b.y : a.y, (a.z < b.z) ? b.z : a.z,
            (a.w < b.w) ? b.w : a.w};
}

template <Numeric T>
[[nodiscard]] Vector4<T> abs(const Vector4<T>& v) {
    return {detail::absOf(v.x), detail::absOf(v.y), detail::absOf(v.z),
            detail::absOf(v.w)};
}

/// Perspective divide: back to a 3D point from homogeneous coordinates.
/// Undefined for w = 0, which is a direction rather than a point.
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> perspectiveDivide(const Vector4<T>& v) noexcept {
    return {v.x / v.w, v.y / v.w, v.z / v.w};
}

using Vec4 = Vector4<double>;
using Vec4f = Vector4<float>;

}  // namespace ysq
