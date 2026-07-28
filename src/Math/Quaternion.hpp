#pragma once

#include <Math/Matrix3.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <cassert>
#include <cstddef>
#include <optional>

namespace ysq {

/// A rotation about a unit axis, and its axis-angle in a form that composes
/// and interpolates without gimbal lock.
///
/// Stored as (w, x, y, z), scalar part first, and indexed in that order. The
/// other common layout puts w last; the two are indistinguishable at the type
/// level and mixing them silently produces a wrong rotation, so this is stated
/// rather than implied.
///
/// Default-constructs to the identity rotation, not to zero. A zero quaternion
/// is not a rotation at all, so zeroing here would be the less safe default.
/// Quaternion::zero() is available where the additive identity is what is
/// wanted, such as accumulating a derivative.
///
/// Both conversions with Matrix3 live in this header. Matrix3 knows nothing
/// about quaternions, which is what keeps the two headers acyclic.
template <Numeric T>
struct Quaternion {
    using value_type = T;

    T w{1};
    T x{};
    T y{};
    T z{};

    [[nodiscard]] static constexpr std::size_t size() noexcept { return 4; }

    /// 0 is w, then x, y, z. Index must be less than size().
    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? w : ((index == 1) ? x : ((index == 2) ? y : z));
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? w : ((index == 1) ? x : ((index == 2) ? y : z));
    }

    /// The vector part.
    [[nodiscard]] constexpr Vector3<T> xyz() const noexcept { return {x, y, z}; }

    [[nodiscard]] static constexpr Quaternion identity() noexcept { return {}; }

    [[nodiscard]] static constexpr Quaternion zero() noexcept {
        return {T{0}, T{0}, T{0}, T{0}};
    }

    [[nodiscard]] static constexpr Quaternion
    fromScalarVector(T scalar, const Vector3<T>& vector) noexcept {
        return {scalar, vector.x, vector.y, vector.z};
    }

    /// Right-handed rotation about a unit axis. A non-unit axis produces a
    /// non-unit quaternion, which is a scaling as well as a rotation.
    [[nodiscard]] static Quaternion fromAxisAngle(const Vector3<T>& axis, T angle) {
        using std::cos;
        using std::sin;
        const T half = angle / T{2};
        return fromScalarVector(cos(half), axis * sin(half));
    }

    /// Intrinsic Z-Y-X: yaw about Z, then pitch about the new Y, then roll
    /// about the new X. Equivalent to the product Rz(yaw) Ry(pitch) Rx(roll).
    ///
    /// The order is in the name because there are twelve conventions and
    /// picking the wrong one produces a plausible-looking rotation that is
    /// simply not the one asked for.
    [[nodiscard]] static Quaternion fromEulerZYX(T yaw, T pitch, T roll) {
        using std::cos;
        using std::sin;
        const T cy = cos(yaw / T{2});
        const T sy = sin(yaw / T{2});
        const T cp = cos(pitch / T{2});
        const T sp = sin(pitch / T{2});
        const T cr = cos(roll / T{2});
        const T sr = sin(roll / T{2});
        return {cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr};
    }

    /// Shepperd's method: pick whichever of the four components is largest and
    /// derive the rest from it.
    ///
    /// The textbook derivation from the trace alone divides by something that
    /// approaches zero as the rotation approaches a half turn, so it loses all
    /// precision exactly where a naive test with small rotations would never
    /// notice. Branching on the largest component keeps the divisor bounded
    /// away from zero for every input.
    [[nodiscard]] static Quaternion fromRotationMatrix(const Matrix3<T>& m) {
        using std::sqrt;
        const T tr = trace(m);

        if (T{0} < tr) {
            const T s = sqrt(tr + T{1}) * T{2};
            return {s / T{4}, (m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s,
                    (m(1, 0) - m(0, 1)) / s};
        }
        if (m(1, 1) < m(0, 0) && m(2, 2) < m(0, 0)) {
            const T s = sqrt(T{1} + m(0, 0) - m(1, 1) - m(2, 2)) * T{2};
            return {(m(2, 1) - m(1, 2)) / s, s / T{4}, (m(0, 1) + m(1, 0)) / s,
                    (m(0, 2) + m(2, 0)) / s};
        }
        if (m(2, 2) < m(1, 1)) {
            const T s = sqrt(T{1} + m(1, 1) - m(0, 0) - m(2, 2)) * T{2};
            return {(m(0, 2) - m(2, 0)) / s, (m(0, 1) + m(1, 0)) / s, s / T{4},
                    (m(1, 2) + m(2, 1)) / s};
        }
        const T s = sqrt(T{1} + m(2, 2) - m(0, 0) - m(1, 1)) * T{2};
        return {(m(1, 0) - m(0, 1)) / s, (m(0, 2) + m(2, 0)) / s, (m(1, 2) + m(2, 1)) / s,
                s / T{4}};
    }

    constexpr Quaternion& operator+=(const Quaternion& other) noexcept {
        w += other.w;
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Quaternion& operator-=(const Quaternion& other) noexcept {
        w -= other.w;
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    constexpr Quaternion& operator*=(T scalar) noexcept {
        w *= scalar;
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    constexpr Quaternion& operator/=(T scalar) noexcept {
        w /= scalar;
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    /// Right-multiplies: `a *= b` is `a = a * b`, matching how rotations
    /// compose.
    constexpr Quaternion& operator*=(const Quaternion& other) noexcept {
        *this = *this * other;
        return *this;
    }

    [[nodiscard]] friend constexpr Quaternion operator+(const Quaternion& q) noexcept {
        return q;
    }

    [[nodiscard]] friend constexpr Quaternion operator-(const Quaternion& q) noexcept {
        return {-q.w, -q.x, -q.y, -q.z};
    }

    [[nodiscard]] friend constexpr Quaternion operator+(const Quaternion& a,
                                                        const Quaternion& b) noexcept {
        return {a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
    }

    [[nodiscard]] friend constexpr Quaternion operator-(const Quaternion& a,
                                                        const Quaternion& b) noexcept {
        return {a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z};
    }

    [[nodiscard]] friend constexpr Quaternion operator*(const Quaternion& q,
                                                        T scalar) noexcept {
        return {q.w * scalar, q.x * scalar, q.y * scalar, q.z * scalar};
    }

    [[nodiscard]] friend constexpr Quaternion operator*(T scalar,
                                                        const Quaternion& q) noexcept {
        return q * scalar;
    }

    [[nodiscard]] friend constexpr Quaternion operator/(const Quaternion& q,
                                                        T scalar) noexcept {
        return {q.w / scalar, q.x / scalar, q.y / scalar, q.z / scalar};
    }

    /// The Hamilton product. Not commutative: `a * b` applies b first, then a,
    /// the same way matrix composition reads.
    [[nodiscard]] friend constexpr Quaternion operator*(const Quaternion& a,
                                                        const Quaternion& b) noexcept {
        return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
    }

    /// Exact, component by component. Note that q and -q are the same rotation
    /// but are not equal under this; use angleBetween to compare rotations.
    [[nodiscard]] friend constexpr bool operator==(const Quaternion&,
                                                   const Quaternion&) = default;
};

/// The axis and angle a rotation was, or could have been, built from.
template <Numeric T>
struct AxisAngle {
    Vector3<T> axis{T{1}, T{0}, T{0}};
    T angle{};
};

/// Yaw about Z, pitch about Y, roll about X, in radians.
template <Numeric T>
struct EulerZYX {
    T yaw{};
    T pitch{};
    T roll{};
};

template <Numeric T>
[[nodiscard]] constexpr T dot(const Quaternion<T>& a, const Quaternion<T>& b) noexcept {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

template <Numeric T>
[[nodiscard]] constexpr T lengthSquared(const Quaternion<T>& q) noexcept {
    return dot(q, q);
}

template <Numeric T>
[[nodiscard]] auto length(const Quaternion<T>& q) {
    return detail::sqrtOf(lengthSquared(q));
}

template <Numeric T>
[[nodiscard]] constexpr Quaternion<T> conjugate(const Quaternion<T>& q) noexcept {
    return {q.w, -q.x, -q.y, -q.z};
}

template <Numeric T>
[[nodiscard]] Quaternion<T> normalized(const Quaternion<T>& q) {
    return q / length(q);
}

template <Numeric T>
[[nodiscard]] std::optional<Quaternion<T>> tryNormalized(const Quaternion<T>& q) {
    const T len = length(q);
    if (!(T{0} < len) || !detail::isFiniteValue(len)) {
        return std::nullopt;
    }
    return q / len;
}

/// The general inverse, valid for any non-zero quaternion.
template <Numeric T>
[[nodiscard]] constexpr Quaternion<T> inverse(const Quaternion<T>& q) noexcept {
    return conjugate(q) / lengthSquared(q);
}

/// The inverse of a unit quaternion is just its conjugate. Correct only for a
/// unit quaternion; wrong, silently, for anything else.
template <Numeric T>
[[nodiscard]] constexpr Quaternion<T> inverseUnit(const Quaternion<T>& q) noexcept {
    return conjugate(q);
}

/// Rotates v by a unit quaternion.
///
/// The short form of q (0, v) q*, which costs two cross products instead of
/// two full quaternion products. Requires q to be unit; scale it yourself if
/// you meant to scale.
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> rotate(const Quaternion<T>& q,
                                          const Vector3<T>& v) noexcept {
    const Vector3<T> u = q.xyz();
    const Vector3<T> t = cross(u, v) * T{2};
    return v + t * q.w + cross(u, t);
}

template <Numeric T>
[[nodiscard]] Matrix3<T> toMatrix3(const Quaternion<T>& q) {
    const T xx = q.x * q.x;
    const T yy = q.y * q.y;
    const T zz = q.z * q.z;
    const T xy = q.x * q.y;
    const T xz = q.x * q.z;
    const T yz = q.y * q.z;
    const T wx = q.w * q.x;
    const T wy = q.w * q.y;
    const T wz = q.w * q.z;

    return Matrix3<T>::fromRows(
        {T{1} - T{2} * (yy + zz), T{2} * (xy - wz), T{2} * (xz + wy)},
        {T{2} * (xy + wz), T{1} - T{2} * (xx + zz), T{2} * (yz - wx)},
        {T{2} * (xz - wy), T{2} * (yz + wx), T{1} - T{2} * (xx + yy)});
}

/// Axis and angle of a unit quaternion, with the angle in [0, pi].
///
/// atan2 of the vector part's length against w, not acos of w, for the same
/// reason Vector3's angleBetween uses atan2: near the identity, acos of a
/// value that has rounded to 1 returns exactly zero. At the identity itself
/// the axis is arbitrary and comes back as +X.
template <Numeric T>
[[nodiscard]] AxisAngle<T> toAxisAngle(const Quaternion<T>& q) {
    const Vector3<T> vector = q.xyz();
    const T vectorLength = length(vector);
    const T angle = T{2} * detail::atan2Of(vectorLength, q.w);
    if (!(T{0} < vectorLength)) {
        return {Vector3<T>::unitX(), angle};
    }
    return {vector / vectorLength, angle};
}

/// Pitch is clamped into asin's domain, so a slightly non-unit quaternion
/// yields the nearest valid angle rather than NaN.
///
/// Near pitch = +/- pi/2 the decomposition is degenerate (gimbal lock): yaw and
/// roll stop being separately determined and only their difference, or their
/// sum at the other pole, survives. The general expressions below degrade to
/// atan2(0, 0) there and would hand back two arbitrary angles that do not even
/// reconstruct the rotation.
///
/// So inside about a thousandth of a radian of either pole this pins roll to
/// zero and puts the whole determined quantity into yaw. The angles are then
/// not the ones that were originally passed in, because they cannot be, but
/// they do describe the same rotation. That is the property worth keeping.
template <Numeric T>
[[nodiscard]] EulerZYX<T> toEulerZYX(const Quaternion<T>& q) {
    using std::asin;

    // These two are cos(roll) cos(pitch) and sin(roll) cos(pitch), so they
    // vanish together exactly as pitch approaches a pole and never otherwise.
    const T rollX = T{1} - T{2} * (q.x * q.x + q.y * q.y);
    const T rollY = T{2} * (q.w * q.x + q.y * q.z);
    const T sinPitch = T{2} * (q.w * q.y - q.z * q.x);

    // 2^-40, exactly representable at float and double alike. The threshold is
    // deliberately far below the point where the general branch starts losing
    // accuracy: that branch degrades gracefully, roughly as epsilon over the
    // distance from the pole, while this one is only exactly right at the pole
    // itself. Switching early would be the worse error of the two.
    const T degenerate = T{1} / T{1024} / T{1024} / T{1024} / T{1024};

    if (detail::absOf(rollX) < degenerate && detail::absOf(rollY) < degenerate) {
        // asin(1) is pi/2 correctly rounded, which avoids needing kPi<T> and
        // keeps this usable for any Numeric. Taking pitch as exactly a right
        // angle rather than through asin(sinPitch) matters: asin loses half
        // its significant digits at its endpoints, about 1.5e-8 radians, which
        // is large enough to show up in the reconstructed rotation.
        const T rightAngle = asin(T{1});
        if (T{0} < sinPitch) {
            return {T{2} * detail::atan2Of(-q.x, q.w), rightAngle, T{0}};
        }
        return {T{2} * detail::atan2Of(q.x, q.w), -rightAngle, T{0}};
    }

    return {detail::atan2Of(T{2} * (q.w * q.z + q.x * q.y),
                            T{1} - T{2} * (q.y * q.y + q.z * q.z)),
            asin(clamp(sinPitch, T{-1}, T{1})), detail::atan2Of(rollY, rollX)};
}

/// The angle of the rotation that takes a to b, in [0, pi].
///
/// Forms the relative rotation and reads its angle off with atan2, rather than
/// recovering the half-angle sine from the dot product. Going through
/// sqrt(1 - cos^2) would lose half the significant digits for a small angle,
/// exactly the way acos does: the answer would bottom out around 3e-8 and
/// never resolve anything finer, which is the failure mode Vector3's
/// angleBetween avoids for the same reason.
///
/// The absolute value of the scalar part is what makes q and -q the same
/// rotation, which they are, and what keeps the result in [0, pi].
template <Numeric T>
[[nodiscard]] auto angleBetween(const Quaternion<T>& a, const Quaternion<T>& b) {
    const Quaternion<T> relative = conjugate(normalized(a)) * normalized(b);
    return T{2} * detail::atan2Of(length(relative.xyz()), detail::absOf(relative.w));
}

/// Normalised linear interpolation. Cheap, and follows the shortest arc, but
/// its angular rate is not constant: it lingers at the ends and hurries
/// through the middle.
template <Numeric T>
[[nodiscard]] Quaternion<T> nlerp(const Quaternion<T>& a, const Quaternion<T>& b, T t) {
    const Quaternion<T> target = (dot(a, b) < T{0}) ? -b : b;
    return normalized(a * (T{1} - t) + target * t);
}

/// Spherical linear interpolation: constant angular rate along the shortest
/// arc.
///
/// Falls back to nlerp when the two are nearly parallel, where dividing by
/// sin(theta) would divide by nearly zero and the two agree to well within the
/// error that division would introduce anyway.
template <Numeric T>
[[nodiscard]] Quaternion<T> slerp(const Quaternion<T>& a, const Quaternion<T>& b, T t) {
    using std::acos;
    using std::sin;

    T cosTheta = dot(a, b);
    Quaternion<T> target = b;
    if (cosTheta < T{0}) {  // take the short way round
        target = -b;
        cosTheta = -cosTheta;
    }

    constexpr T parallelEnough = static_cast<T>(0.9995);
    if (parallelEnough < cosTheta) {
        return normalized(a * (T{1} - t) + target * t);
    }

    const T theta = acos(clamp(cosTheta, T{-1}, T{1}));
    const T sinTheta = sin(theta);
    return (a * sin((T{1} - t) * theta) + target * sin(t * theta)) / sinTheta;
}

using Quat = Quaternion<double>;
using Quatf = Quaternion<float>;

}  // namespace ysq
