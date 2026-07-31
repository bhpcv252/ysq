#pragma once

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <optional>

namespace ysq {

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

/// The nearest intersection of `ray` with `sphere`, as a distance along
/// `ray.direction` from `ray.origin`, or nullopt if the ray misses the
/// sphere or the sphere is entirely behind the origin.
///
/// Pure geometry: nothing here is a physical law, only the quadratic that
/// finds where a line meets a sphere, so it carries no opinion about what a
/// hit means to whoever asked (a render, an occlusion test, anything else).
template <Numeric T>
[[nodiscard]] std::optional<T> intersect(const Ray3<T>& ray, const Sphere3<T>& sphere) {
    using std::sqrt;

    const Vector3<T> toOrigin = ray.origin - sphere.center;
    const T a = lengthSquared(ray.direction);
    const T b = T{2} * dot(ray.direction, toOrigin);
    const T c = lengthSquared(toOrigin) - sphere.radius * sphere.radius;

    const T discriminant = b * b - T{4} * a * c;
    if (discriminant < T{0}) {
        return std::nullopt;
    }

    const T root = sqrt(discriminant);
    const T nearer = (-b - root) / (T{2} * a);
    const T farther = (-b + root) / (T{2} * a);

    if (T{0} <= nearer) {
        return nearer;
    }
    if (T{0} <= farther) {
        return farther;
    }
    return std::nullopt;
}

/// Whether `sphere` blocks the line of sight from `from` to `to`: an
/// intersection strictly between the two, not merely somewhere along the
/// infinite ray. `from` and `to` sitting exactly on the sphere's surface
/// (t = 0 or t = 1) does not count as blocking, so a ray leaving or arriving
/// tangent to a sphere is not occluded by that same sphere.
template <Numeric T>
[[nodiscard]] bool segmentIntersectsSphere(const Vector3<T>& from, const Vector3<T>& to,
                                           const Sphere3<T>& sphere) {
    const Ray3<T> ray{from, to - from};
    const std::optional<T> hit = intersect(ray, sphere);
    return hit.has_value() && T{0} < *hit && *hit < T{1};
}

}  // namespace ysq
