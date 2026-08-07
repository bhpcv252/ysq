#pragma once

#include <Math/Scalar.hpp>
#include <Math/Vector2.hpp>

#include <cassert>
#include <cstddef>
#include <optional>

namespace ysq {

/// Three-component vector. The workhorse: positions, velocities, forces, and
/// the spatial part of a four-vector.
///
/// An aggregate with public members, standard layout and trivially copyable,
/// so an array of these uploads to a GPU buffer as-is. See Vector2 for the
/// reasoning on default member initializers.
template <Numeric T>
struct Vector3 {
    using value_type = T;

    T x{};
    T y{};
    T z{};

    [[nodiscard]] static constexpr std::size_t size() noexcept { return 3; }

    /// Index must be less than size(). Out of range yields the last component
    /// rather than reading past the object.
    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? x : ((index == 1) ? y : z);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? x : ((index == 1) ? y : z);
    }

    [[nodiscard]] constexpr Vector2<T> xy() const noexcept { return {x, y}; }

    [[nodiscard]] static constexpr Vector3 zero() noexcept { return {}; }
    [[nodiscard]] static constexpr Vector3 splat(T value) noexcept {
        return {value, value, value};
    }
    [[nodiscard]] static constexpr Vector3 unitX() noexcept { return {T{1}, T{0}, T{0}}; }
    [[nodiscard]] static constexpr Vector3 unitY() noexcept { return {T{0}, T{1}, T{0}}; }
    [[nodiscard]] static constexpr Vector3 unitZ() noexcept { return {T{0}, T{0}, T{1}}; }

    constexpr Vector3& operator+=(const Vector3& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Vector3& operator-=(const Vector3& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    constexpr Vector3& operator*=(T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    constexpr Vector3& operator/=(T scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    [[nodiscard]] friend constexpr Vector3 operator+(const Vector3& v) noexcept {
        return v;
    }

    [[nodiscard]] friend constexpr Vector3 operator-(const Vector3& v) noexcept {
        return {-v.x, -v.y, -v.z};
    }

    [[nodiscard]] friend constexpr Vector3 operator+(const Vector3& a,
                                                     const Vector3& b) noexcept {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    [[nodiscard]] friend constexpr Vector3 operator-(const Vector3& a,
                                                     const Vector3& b) noexcept {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    [[nodiscard]] friend constexpr Vector3 operator*(const Vector3& v,
                                                     T scalar) noexcept {
        return {v.x * scalar, v.y * scalar, v.z * scalar};
    }

    [[nodiscard]] friend constexpr Vector3 operator*(T scalar,
                                                     const Vector3& v) noexcept {
        return {scalar * v.x, scalar * v.y, scalar * v.z};
    }

    /// Divides rather than multiplying by a reciprocal, for the same reason as
    /// Vector2: one rounding step per component instead of two.
    [[nodiscard]] friend constexpr Vector3 operator/(const Vector3& v,
                                                     T scalar) noexcept {
        return {v.x / scalar, v.y / scalar, v.z / scalar};
    }

    [[nodiscard]] friend constexpr bool operator==(const Vector3&,
                                                   const Vector3&) = default;
};

template <Numeric T>
[[nodiscard]] constexpr T dot(const Vector3<T>& a, const Vector3<T>& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <Numeric T>
[[nodiscard]] constexpr Vector3<T> cross(const Vector3<T>& a,
                                         const Vector3<T>& b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// a . (b x c). The signed volume of the parallelepiped, and the determinant of
/// the matrix with the three as its columns.
template <Numeric T>
[[nodiscard]] constexpr T scalarTriple(const Vector3<T>& a, const Vector3<T>& b,
                                       const Vector3<T>& c) noexcept {
    return dot(a, cross(b, c));
}

template <Numeric T>
[[nodiscard]] constexpr T lengthSquared(const Vector3<T>& v) noexcept {
    return dot(v, v);
}

template <Numeric T>
[[nodiscard]] auto length(const Vector3<T>& v) {
    return detail::sqrtOf(lengthSquared(v));
}

/// Undefined direction at zero length: every component comes back NaN, which
/// propagates rather than quietly returning a wrong unit vector. Use
/// tryNormalized where the input can legitimately be zero.
template <Numeric T>
[[nodiscard]] Vector3<T> normalized(const Vector3<T>& v) {
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
[[nodiscard]] std::optional<Vector3<T>> tryNormalized(const Vector3<T>& v) {
    const T len = length(v);
    // Negated so a NaN fails too, and checked for finiteness so an overflow
    // does not divide through to zero and report success.
    if (!(T{0} < len) || !detail::isFiniteValue(len)) {
        return std::nullopt;
    }
    return v / len;
}

template <Numeric T>
[[nodiscard]] constexpr T distanceSquared(const Vector3<T>& a,
                                          const Vector3<T>& b) noexcept {
    return lengthSquared(a - b);
}

template <Numeric T>
[[nodiscard]] auto distance(const Vector3<T>& a, const Vector3<T>& b) {
    return length(a - b);
}

/// Exact at both endpoints, which the cheaper `a + (b - a) * t` is not.
/// Extrapolates outside [0, 1].
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> lerp(const Vector3<T>& a, const Vector3<T>& b,
                                        T t) noexcept {
    return a * (T{1} - t) + b * t;
}

/// Component of a along onto. Undefined for a zero `onto`.
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> project(const Vector3<T>& a,
                                           const Vector3<T>& onto) noexcept {
    return onto * (dot(a, onto) / lengthSquared(onto));
}

/// The part of a orthogonal to `from`. project + reject reconstructs a.
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> reject(const Vector3<T>& a,
                                          const Vector3<T>& from) noexcept {
    return a - project(a, from);
}

/// Mirror of v in the plane whose normal is n. n must be a unit vector.
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> reflect(const Vector3<T>& v,
                                           const Vector3<T>& n) noexcept {
    return v - n * (T{2} * dot(v, n));
}

template <Numeric T>
[[nodiscard]] constexpr Vector3<T> hadamard(const Vector3<T>& a,
                                            const Vector3<T>& b) noexcept {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

template <Numeric T>
[[nodiscard]] constexpr Vector3<T> min(const Vector3<T>& a,
                                       const Vector3<T>& b) noexcept {
    return {(a.x < b.x) ? a.x : b.x, (a.y < b.y) ? a.y : b.y, (a.z < b.z) ? a.z : b.z};
}

template <Numeric T>
[[nodiscard]] constexpr Vector3<T> max(const Vector3<T>& a,
                                       const Vector3<T>& b) noexcept {
    return {(a.x < b.x) ? b.x : a.x, (a.y < b.y) ? b.y : a.y, (a.z < b.z) ? b.z : a.z};
}

template <Numeric T>
[[nodiscard]] Vector3<T> abs(const Vector3<T>& v) {
    return {detail::absOf(v.x), detail::absOf(v.y), detail::absOf(v.z)};
}

/// Unsigned angle between a and b, in [0, pi].
///
/// atan2 of the cross magnitude over the dot, not acos of the normalised dot.
/// acos loses most of its significant digits for nearly parallel or nearly
/// antiparallel inputs, where its argument sits on a flat part of the curve;
/// the atan2 form is accurate across the whole range.
template <Numeric T>
[[nodiscard]] auto angleBetween(const Vector3<T>& a, const Vector3<T>& b) {
    return detail::atan2Of(length(cross(a, b)), dot(a, b));
}

/// Rotate v about a unit axis by `angle` radians, right-handed. Rodrigues'
/// formula. This is here rather than waiting for Quaternion because rotation
/// invariance is the cheapest way to test the vector algebra itself.
template <Numeric T>
[[nodiscard]] Vector3<T> rotateAbout(const Vector3<T>& v, const Vector3<T>& axis,
                                     T angle) {
    using std::cos;
    using std::sin;
    const T c = cos(angle);
    const T s = sin(angle);
    return v * c + cross(axis, v) * s + axis * (dot(axis, v) * (T{1} - c));
}

using Vec3 = Vector3<double>;
using Vec3f = Vector3<float>;

}  // namespace ysq
