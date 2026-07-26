#pragma once

#include <Math/Matrix2.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

namespace ysq {

/// Spherical, cylindrical and polar coordinates, and the bases that go with
/// them.
///
/// **The convention is the physics one**: `polar` is the angle down from +z,
/// in [0, pi], and `azimuth` is the angle round from +x in the xy plane, in
/// (-pi, pi]. Mathematics texts routinely swap the names of the two, and a
/// swapped pair produces a plausible-looking result that is simply the wrong
/// point. It is stated here, in the member names, and in the tests, rather
/// than left to be inferred.
///
///     x = r sin(polar) cos(azimuth)
///     y = r sin(polar) sin(azimuth)
///     z = r cos(polar)
///
/// Converting a *point* between systems is only half of what is needed. A
/// vector quantity, a velocity or a field, has components against a local
/// basis that changes from point to point, so it needs the basis as well. That
/// is what the basis and Jacobian functions are for.
template <Numeric T>
struct Spherical {
    using value_type = T;

    T radius{};
    /// Down from +z, in [0, pi].
    T polar{};
    /// Round from +x in the xy plane, in (-pi, pi].
    T azimuth{};

    [[nodiscard]] static constexpr std::size_t size() noexcept { return 3; }

    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? radius : ((index == 1) ? polar : azimuth);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? radius : ((index == 1) ? polar : azimuth);
    }

    [[nodiscard]] friend constexpr bool operator==(const Spherical&,
                                                   const Spherical&) = default;
};

/// Distance from the z axis, angle round it, and height along it.
template <Numeric T>
struct Cylindrical {
    using value_type = T;

    T radius{};
    /// Round from +x in the xy plane, in (-pi, pi].
    T azimuth{};
    T height{};

    [[nodiscard]] static constexpr std::size_t size() noexcept { return 3; }

    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? radius : ((index == 1) ? azimuth : height);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? radius : ((index == 1) ? azimuth : height);
    }

    [[nodiscard]] friend constexpr bool operator==(const Cylindrical&,
                                                   const Cylindrical&) = default;
};

/// Two-dimensional polar coordinates.
template <Numeric T>
struct Polar {
    using value_type = T;

    T radius{};
    /// Counter-clockwise from +x, in (-pi, pi].
    T angle{};

    [[nodiscard]] static constexpr std::size_t size() noexcept { return 2; }

    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? radius : angle;
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? radius : angle;
    }

    [[nodiscard]] friend constexpr bool operator==(const Polar&,
                                                   const Polar&) = default;
};

// --- Points -----------------------------------------------------------------

template <Numeric T>
[[nodiscard]] Vector3<T> toCartesian(const Spherical<T>& at) {
    using std::cos;
    using std::sin;
    const T sinPolar = sin(at.polar);
    return {at.radius * sinPolar * cos(at.azimuth),
            at.radius * sinPolar * sin(at.azimuth), at.radius * cos(at.polar)};
}

/// The polar angle comes from atan2 of the distance from the axis against z,
/// not from acos(z / r).
///
/// Both are correct on paper. acos loses half its digits near the poles, where
/// its argument has rounded to +/-1 and the answer it wants is a small angle,
/// which is exactly where a polar-orbit calculation spends its time. Dividing
/// by r would also have to special-case the origin.
template <Numeric T>
[[nodiscard]] Spherical<T> toSpherical(const Vector3<T>& at) {
    const T fromAxis = detail::hypotOf(at.x, at.y);
    return {length(at), detail::atan2Of(fromAxis, at.z),
            detail::atan2Of(at.y, at.x)};
}

template <Numeric T>
[[nodiscard]] Vector3<T> toCartesian(const Cylindrical<T>& at) {
    using std::cos;
    using std::sin;
    return {at.radius * cos(at.azimuth), at.radius * sin(at.azimuth), at.height};
}

template <Numeric T>
[[nodiscard]] Cylindrical<T> toCylindrical(const Vector3<T>& at) {
    return {detail::hypotOf(at.x, at.y), detail::atan2Of(at.y, at.x), at.z};
}

template <Numeric T>
[[nodiscard]] Vector2<T> toCartesian(const Polar<T>& at) {
    using std::cos;
    using std::sin;
    return {at.radius * cos(at.angle), at.radius * sin(at.angle)};
}

template <Numeric T>
[[nodiscard]] Polar<T> toPolar(const Vector2<T>& at) {
    return {length(at), detail::atan2Of(at.y, at.x)};
}

// --- Local bases ------------------------------------------------------------

/// The orthonormal spherical basis at a point, as the columns of a matrix:
/// column 0 is e_r, column 1 is e_polar, column 2 is e_azimuth.
///
/// As columns rather than as three returned vectors because that makes the
/// matrix itself the change of basis. Multiplying by it takes components in
/// the local basis to Cartesian ones, and since it is orthonormal its
/// transpose takes them back.
///
/// Degenerate on the z axis, where the azimuthal direction is not defined.
/// There the third column comes out as +y, which is a choice rather than an
/// answer.
template <Numeric T>
[[nodiscard]] Matrix3<T> sphericalBasis(const Spherical<T>& at) {
    using std::cos;
    using std::sin;
    const T sinPolar = sin(at.polar);
    const T cosPolar = cos(at.polar);
    const T sinAzimuth = sin(at.azimuth);
    const T cosAzimuth = cos(at.azimuth);

    return Matrix3<T>::fromColumns(
        {sinPolar * cosAzimuth, sinPolar * sinAzimuth, cosPolar},
        {cosPolar * cosAzimuth, cosPolar * sinAzimuth, -sinPolar},
        {-sinAzimuth, cosAzimuth, T{0}});
}

/// Columns e_radius, e_azimuth, e_height. Orthonormal, and degenerate only on
/// the axis itself.
template <Numeric T>
[[nodiscard]] Matrix3<T> cylindricalBasis(const Cylindrical<T>& at) {
    using std::cos;
    using std::sin;
    const T sinAzimuth = sin(at.azimuth);
    const T cosAzimuth = cos(at.azimuth);

    return Matrix3<T>::fromColumns({cosAzimuth, sinAzimuth, T{0}},
                                   {-sinAzimuth, cosAzimuth, T{0}},
                                   {T{0}, T{0}, T{1}});
}

/// Columns e_radius, e_angle.
template <Numeric T>
[[nodiscard]] Matrix2<T> polarBasis(const Polar<T>& at) {
    using std::cos;
    using std::sin;
    return Matrix2<T>::fromColumns({cos(at.angle), sin(at.angle)},
                                   {-sin(at.angle), cos(at.angle)});
}

// --- Jacobians --------------------------------------------------------------

/// d(x, y, z) / d(radius, polar, azimuth): the coordinate basis, unnormalised.
///
/// The same directions as sphericalBasis, scaled by how far the point actually
/// moves per unit of each coordinate: 1, r, and r sin(polar). Its determinant
/// is r^2 sin(polar), the volume element every spherical integral carries.
template <Numeric T>
[[nodiscard]] Matrix3<T> sphericalJacobian(const Spherical<T>& at) {
    using std::sin;
    const Matrix3<T> basis = sphericalBasis(at);
    return Matrix3<T>::fromColumns(basis[0], basis[1] * at.radius,
                                   basis[2] * (at.radius * sin(at.polar)));
}

/// d(x, y, z) / d(radius, azimuth, height). Its determinant is the radius.
template <Numeric T>
[[nodiscard]] Matrix3<T> cylindricalJacobian(const Cylindrical<T>& at) {
    const Matrix3<T> basis = cylindricalBasis(at);
    return Matrix3<T>::fromColumns(basis[0], basis[1] * at.radius, basis[2]);
}

// --- Vector components ------------------------------------------------------

/// Cartesian components of a vector whose components are given against the
/// local spherical basis at `at`.
template <Numeric T>
[[nodiscard]] Vector3<T> sphericalComponentsToCartesian(
    const Spherical<T>& at, const Vector3<T>& components) {
    return sphericalBasis(at) * components;
}

/// The inverse. The basis is orthonormal, so this is a transpose rather than
/// an inversion.
template <Numeric T>
[[nodiscard]] Vector3<T> cartesianComponentsToSpherical(const Spherical<T>& at,
                                                        const Vector3<T>& vector) {
    return transpose(sphericalBasis(at)) * vector;
}

template <Numeric T>
[[nodiscard]] Vector3<T> cylindricalComponentsToCartesian(
    const Cylindrical<T>& at, const Vector3<T>& components) {
    return cylindricalBasis(at) * components;
}

template <Numeric T>
[[nodiscard]] Vector3<T> cartesianComponentsToCylindrical(
    const Cylindrical<T>& at, const Vector3<T>& vector) {
    return transpose(cylindricalBasis(at)) * vector;
}

}  // namespace ysq
