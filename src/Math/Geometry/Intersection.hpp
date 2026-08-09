#pragma once

#include <Math/Geometry/Primitives.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>

namespace ysq {

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

/// Where `ray` crosses `plane`, or nullopt if it is parallel to it (no
/// crossing, or the ray lies in the plane, which has no single answer) or
/// the crossing is behind `ray.origin`.
template <Numeric T>
[[nodiscard]] std::optional<T> intersect(const Ray3<T>& ray, const Plane3<T>& plane) {
    const T denominator = dot(plane.normal, ray.direction);
    if (approxEqual(denominator, T{0})) {
        return std::nullopt;
    }
    const T t = (plane.distance - dot(plane.normal, ray.origin)) / denominator;
    if (t < T{0}) {
        return std::nullopt;
    }
    return t;
}

/// The Möller-Trumbore algorithm: where `ray` crosses `triangle`, in
/// barycentric coordinates internally, without ever computing the
/// triangle's plane explicitly. `nullopt` for a ray parallel to (or missing)
/// the triangle, or a crossing behind `ray.origin`.
template <Numeric T>
[[nodiscard]] std::optional<T> intersect(const Ray3<T>& ray,
                                         const Triangle3<T>& triangle) {
    const Vector3<T> edge1 = triangle.b - triangle.a;
    const Vector3<T> edge2 = triangle.c - triangle.a;
    const Vector3<T> pVec = cross(ray.direction, edge2);
    const T determinant = dot(edge1, pVec);

    if (detail::absOf(determinant) < std::numeric_limits<T>::epsilon() * T{100}) {
        return std::nullopt;
    }

    const T invDeterminant = T{1} / determinant;
    const Vector3<T> tVec = ray.origin - triangle.a;
    const T u = dot(tVec, pVec) * invDeterminant;
    if (u < T{0} || u > T{1}) {
        return std::nullopt;
    }

    const Vector3<T> qVec = cross(tVec, edge1);
    const T v = dot(ray.direction, qVec) * invDeterminant;
    if (v < T{0} || u + v > T{1}) {
        return std::nullopt;
    }

    const T t = dot(edge2, qVec) * invDeterminant;
    if (t < T{0}) {
        return std::nullopt;
    }
    return t;
}

/// The slab method: intersects `ray` against each pair of axis-aligned
/// planes bounding `box` in turn, narrowing `[tMin, tMax]` each time.
/// `nullopt` if the narrowed interval ever becomes empty (the ray misses the
/// box on at least one axis) or lies entirely behind the origin. When the
/// origin starts inside the box, `tMin` is negative (the box's near face is
/// behind the origin) and this returns `tMax`, the exit point, matching
/// `intersect(Ray3, Sphere3)`'s convention of returning the nearer
/// nonnegative crossing.
template <Numeric T>
[[nodiscard]] std::optional<T> intersect(const Ray3<T>& ray, const AABB3<T>& box) {
    T tMin = std::numeric_limits<T>::lowest();
    T tMax = std::numeric_limits<T>::max();

    for (std::size_t axis = 0; axis < 3; ++axis) {
        const T origin = ray.origin[axis];
        const T direction = ray.direction[axis];
        const T boxMin = box.min[axis];
        const T boxMax = box.max[axis];

        if (direction == T{0}) {
            if (origin < boxMin || origin > boxMax) {
                return std::nullopt;
            }
            continue;
        }

        T t1 = (boxMin - origin) / direction;
        T t2 = (boxMax - origin) / direction;
        if (t1 > t2) {
            std::swap(t1, t2);
        }
        tMin = std::max(tMin, t1);
        tMax = std::min(tMax, t2);
        if (tMin > tMax) {
            return std::nullopt;
        }
    }

    if (tMax < T{0}) {
        return std::nullopt;
    }
    return (tMin >= T{0}) ? tMin : tMax;
}

/// Whether two spheres overlap (including touching exactly): the distance
/// between centers compared against the sum of radii, both squared so
/// neither needs a square root.
template <Numeric T>
[[nodiscard]] bool intersects(const Sphere3<T>& a, const Sphere3<T>& b) {
    const T radiusSum = a.radius + b.radius;
    return distanceSquared(a.center, b.center) <= radiusSum * radiusSum;
}

/// Whether two axis-aligned boxes overlap (including touching exactly):
/// they fail to overlap only if separated along at least one axis, so this
/// checks all three axes and requires none of them to separate the boxes.
template <Numeric T>
[[nodiscard]] bool intersects(const AABB3<T>& a, const AABB3<T>& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y &&
           a.max.y >= b.min.y && a.min.z <= b.max.z && a.max.z >= b.min.z;
}

/// Whether `plane` passes through `sphere`: the sphere's center is within
/// one radius of the plane, measured by the signed distance
/// `dot(normal, center) - distance`.
template <Numeric T>
[[nodiscard]] bool intersects(const Plane3<T>& plane, const Sphere3<T>& sphere) {
    const T signedDistance = dot(plane.normal, sphere.center) - plane.distance;
    return detail::absOf(signedDistance) <= sphere.radius;
}

/// Whether two oriented boxes overlap: the Separating Axis Theorem (SAT)
/// applied to the fifteen candidate axes that can possibly separate two
/// convex polyhedra with these face normals — each box's own three face
/// normals, plus the nine pairwise cross products of one box's axes with
/// the other's (Gottschalk, Lin & Manocha 1996). The cross-product axes
/// are the ones a naive six-axis test misses: two boxes can overlap on
/// every face normal and still be separated edge-to-edge. `epsilon`
/// guards the cross-product terms against the near-parallel case, where
/// two axes from different boxes coincide and the true separating axis
/// has near-zero length.
template <Numeric T>
[[nodiscard]] bool intersects(const OBB3<T>& a, const OBB3<T>& b) {
    const T epsilon = std::numeric_limits<T>::epsilon() * T{100};

    T r[3][3];
    T absR[3][3];
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            r[i][j] = dot(a.axes[i], b.axes[j]);
            absR[i][j] = detail::absOf(r[i][j]) + epsilon;
        }
    }

    const Vector3<T> centerOffset = b.center - a.center;
    const Vector3<T> t{dot(centerOffset, a.axes[0]), dot(centerOffset, a.axes[1]),
                       dot(centerOffset, a.axes[2])};

    // a's own three face-normal axes.
    for (std::size_t i = 0; i < 3; ++i) {
        const T extentB = b.halfExtents[0] * absR[i][0] + b.halfExtents[1] * absR[i][1] +
                          b.halfExtents[2] * absR[i][2];
        if (detail::absOf(t[i]) > a.halfExtents[i] + extentB) {
            return false;
        }
    }

    // b's own three face-normal axes.
    for (std::size_t j = 0; j < 3; ++j) {
        const T extentA = a.halfExtents[0] * absR[0][j] + a.halfExtents[1] * absR[1][j] +
                          a.halfExtents[2] * absR[2][j];
        const T projection = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
        if (detail::absOf(projection) > extentA + b.halfExtents[j]) {
            return false;
        }
    }

    // The nine cross-product axes a.axes[i] x b.axes[j]: the only axes that
    // can separate two boxes edge-to-edge rather than face-to-face.
    for (std::size_t i = 0; i < 3; ++i) {
        const std::size_t i1 = (i + 1) % 3;
        const std::size_t i2 = (i + 2) % 3;
        for (std::size_t j = 0; j < 3; ++j) {
            const std::size_t j1 = (j + 1) % 3;
            const std::size_t j2 = (j + 2) % 3;
            const T extentA =
                a.halfExtents[i1] * absR[i2][j] + a.halfExtents[i2] * absR[i1][j];
            const T extentB =
                b.halfExtents[j1] * absR[i][j2] + b.halfExtents[j2] * absR[i][j1];
            const T projection = t[i2] * r[i1][j] - t[i1] * r[i2][j];
            if (detail::absOf(projection) > extentA + extentB) {
                return false;
            }
        }
    }

    return true;
}

/// Whether an oriented box and an axis-aligned box overlap: `box` is
/// exactly the degenerate case of an oriented box whose axes are the
/// standard basis, so this builds that `OBB3` and reuses
/// `intersects(OBB3, OBB3)` rather than re-deriving SAT for a case it
/// already covers.
template <Numeric T>
[[nodiscard]] bool intersects(const OBB3<T>& obb, const AABB3<T>& box) {
    const OBB3<T> boxAsObb{
        (box.min + box.max) * T{0.5},
        {Vector3<T>::unitX(), Vector3<T>::unitY(), Vector3<T>::unitZ()},
        (box.max - box.min) * T{0.5}};
    return intersects(obb, boxAsObb);
}

/// Where `ray` crosses `box`: rotates the ray into the box's own local
/// frame (a pure change of basis, since `axes` is orthonormal) and reuses
/// the slab method already proven for axis-aligned boxes above, which is
/// exactly what an oriented box's own frame reduces this problem to.
template <Numeric T>
[[nodiscard]] std::optional<T> intersect(const Ray3<T>& ray, const OBB3<T>& box) {
    const Vector3<T> originOffset = ray.origin - box.center;
    const Ray3<T> localRay{
        Vector3<T>{dot(originOffset, box.axes[0]), dot(originOffset, box.axes[1]),
                   dot(originOffset, box.axes[2])},
        Vector3<T>{dot(ray.direction, box.axes[0]), dot(ray.direction, box.axes[1]),
                   dot(ray.direction, box.axes[2])}};
    const AABB3<T> localBox{-box.halfExtents, box.halfExtents};
    return intersect(localRay, localBox);
}

}  // namespace ysq
