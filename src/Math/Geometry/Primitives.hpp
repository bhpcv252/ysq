#pragma once

#include <Math/Vector3.hpp>

#include <array>

namespace ysq {

/// Euclidean geometry: shapes with no physical meaning (nothing here has a
/// mass, a material, or an owner), the layer `Math/Geometry/Intersection.hpp`
/// and `Math/Geometry/Queries.hpp` build their operations on top of.

/// A ray: `origin` plus `direction`, not required to be unit length. Where a
/// hit distance is returned, it is in units of `direction`'s own length, the
/// same convention a caller already fixes by choosing whether `direction`
/// is normalized.
template <Numeric T>
struct Ray3 {
    Vector3<T> origin{};
    Vector3<T> direction{T{1}, T{0}, T{0}};
};

template <Numeric T>
struct Sphere3 {
    Vector3<T> center{};
    T radius{};
};

/// A plane in Hessian normal form: a point `p` lies on the plane exactly
/// when `dot(normal, p) == distance`. `normal` is the caller's
/// responsibility to keep unit length — nothing here renormalizes it, the
/// same convention `Quaternion` leaves to its own caller.
template <Numeric T>
struct Plane3 {
    Vector3<T> normal{T{0}, T{1}, T{0}};
    T distance{};
};

template <Numeric T>
struct Segment3 {
    Vector3<T> start{};
    Vector3<T> end{};
};

/// Axis-aligned bounding box: every point `p` with `min <= p <= max`,
/// componentwise. An empty box (no point satisfies that) is not
/// representable and not checked for — a caller builds one from real
/// geometry, which always has a well-defined extent.
template <Numeric T>
struct AABB3 {
    Vector3<T> min{};
    Vector3<T> max{};
};

template <Numeric T>
struct Triangle3 {
    Vector3<T> a{};
    Vector3<T> b{};
    Vector3<T> c{};
};

/// Oriented bounding box: `axes` is an orthonormal frame (the caller's
/// responsibility to keep orthonormal, same convention as `Plane3::normal`),
/// and every point within `halfExtents` of `center` along each axis is
/// inside the box.
template <Numeric T>
struct OBB3 {
    Vector3<T> center{};
    std::array<Vector3<T>, 3> axes{Vector3<T>::unitX(), Vector3<T>::unitY(),
                                   Vector3<T>::unitZ()};
    Vector3<T> halfExtents{};
};

}  // namespace ysq
