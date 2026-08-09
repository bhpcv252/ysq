#pragma once

#include <Math/Geometry/Primitives.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>

#include <cstddef>
#include <span>

namespace ysq {

/// The point on `segment` nearest to `point`: projects `point` onto the
/// infinite line through `segment`, then clamps the resulting parameter to
/// `[0, 1]` so the result never leaves the segment itself. A zero-length
/// segment (both endpoints equal) returns that shared point, sidestepping
/// the otherwise-undefined division by the segment's own squared length.
template <Numeric T>
[[nodiscard]] Vector3<T> closestPoint(const Segment3<T>& segment,
                                      const Vector3<T>& point) {
    const Vector3<T> ab = segment.end - segment.start;
    const T lengthSq = lengthSquared(ab);
    if (lengthSq == T{0}) {
        return segment.start;
    }
    const T t = clamp(dot(point - segment.start, ab) / lengthSq, T{0}, T{1});
    return segment.start + ab * t;
}

/// The point on `plane` nearest to `point`: `point` moved along the plane's
/// own normal by exactly its signed distance from the plane.
template <Numeric T>
[[nodiscard]] Vector3<T> closestPoint(const Plane3<T>& plane, const Vector3<T>& point) {
    const T signedDistance = dot(plane.normal, point) - plane.distance;
    return point - plane.normal * signedDistance;
}

/// The point on (or inside) `box` nearest to `point`: each coordinate of
/// `point` clamped independently to the box's extent on that axis, which is
/// what an axis-aligned box's separability across axes reduces the general
/// closest-point problem to.
template <Numeric T>
[[nodiscard]] Vector3<T> closestPoint(const AABB3<T>& box, const Vector3<T>& point) {
    return Vector3<T>{clamp(point.x, box.min.x, box.max.x),
                      clamp(point.y, box.min.y, box.max.y),
                      clamp(point.z, box.min.z, box.max.z)};
}

/// The point on `triangle` nearest to `point`, by the region-based method
/// (Ericson, "Real-Time Collision Detection"): classifies `point` against
/// each of the triangle's two vertex/edge Voronoi regions per side, in turn,
/// falling through to the face's own interior (a barycentric-coordinate
/// combination of all three vertices) only once every vertex and edge
/// region has been ruled out. Six early-exit checks rather than one general
/// formula, but each is a handful of dot products against quantities the
/// next one also needs, not six independent computations.
template <Numeric T>
[[nodiscard]] Vector3<T> closestPoint(const Triangle3<T>& triangle,
                                      const Vector3<T>& point) {
    const Vector3<T>& a = triangle.a;
    const Vector3<T>& b = triangle.b;
    const Vector3<T>& c = triangle.c;

    const Vector3<T> ab = b - a;
    const Vector3<T> ac = c - a;
    const Vector3<T> ap = point - a;
    const T d1 = dot(ab, ap);
    const T d2 = dot(ac, ap);
    if (d1 <= T{0} && d2 <= T{0}) {
        return a;
    }

    const Vector3<T> bp = point - b;
    const T d3 = dot(ab, bp);
    const T d4 = dot(ac, bp);
    if (d3 >= T{0} && d4 <= d3) {
        return b;
    }

    const T vc = d1 * d4 - d3 * d2;
    if (vc <= T{0} && d1 >= T{0} && d3 <= T{0}) {
        const T v = d1 / (d1 - d3);
        return a + ab * v;
    }

    const Vector3<T> cp = point - c;
    const T d5 = dot(ab, cp);
    const T d6 = dot(ac, cp);
    if (d6 >= T{0} && d5 <= d6) {
        return c;
    }

    const T vb = d5 * d2 - d1 * d6;
    if (vb <= T{0} && d2 >= T{0} && d6 <= T{0}) {
        const T w = d2 / (d2 - d6);
        return a + ac * w;
    }

    const T va = d3 * d6 - d5 * d4;
    if (va <= T{0} && (d4 - d3) >= T{0} && (d5 - d6) >= T{0}) {
        const T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    const T denominator = T{1} / (va + vb + vc);
    const T v = vb * denominator;
    const T w = vc * denominator;
    return a + ab * v + ac * w;
}

/// The point on (or inside) `box` nearest to `point`: `point` projected
/// onto each of the box's own axes, clamped to that axis's half-extent,
/// then reconstructed in world space — the oriented-box counterpart of
/// `closestPoint(AABB3, Vector3)`'s per-axis clamp, done in the box's own
/// rotated frame instead of world axes.
template <Numeric T>
[[nodiscard]] Vector3<T> closestPoint(const OBB3<T>& box, const Vector3<T>& point) {
    const Vector3<T> offset = point - box.center;
    Vector3<T> result = box.center;
    for (std::size_t i = 0; i < 3; ++i) {
        const T distance =
            clamp(dot(offset, box.axes[i]), -box.halfExtents[i], box.halfExtents[i]);
        result += box.axes[i] * distance;
    }
    return result;
}

/// The even-odd (PNPOLY) rule: casts a ray from `point` along +x and counts
/// how many of `polygon`'s edges it crosses, odd meaning inside. `polygon`
/// is a simple (non-self-intersecting) polygon, open (no repeated first/last
/// point); vertex order (clockwise or counterclockwise) does not matter to
/// this test.
template <Numeric T>
[[nodiscard]] bool pointInPolygon(std::span<const Vector2<T>> polygon,
                                  const Vector2<T>& point) {
    bool inside = false;
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const Vector2<T>& pi = polygon[i];
        const Vector2<T>& pj = polygon[j];
        const bool straddles = (pi.y > point.y) != (pj.y > point.y);
        if (straddles &&
            point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x) {
            inside = !inside;
        }
    }
    return inside;
}

}  // namespace ysq
