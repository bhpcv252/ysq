#pragma once

#include <Math/Geometry/Primitives.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace ysq {

/// The 2D convex hull of `points`, by Andrew's monotone chain: sort by `x`
/// (then `y`), then build the lower and upper chains of the hull in one pass
/// each, popping the last point off whichever chain whenever it would make a
/// clockwise (non-left) turn. Every popped point is provably inside the
/// hull of the points around it, so nothing this discards needs revisiting.
///
/// Returned counterclockwise, without a repeated closing point. Fewer than
/// three distinct input points has no well-defined hull interior; this
/// returns the (deduplicated) points themselves in that case, same as a
/// hull of them would degenerate to.
template <std::floating_point T>
[[nodiscard]] std::vector<Vector2<T>> convexHull2D(std::vector<Vector2<T>> points) {
    std::sort(points.begin(), points.end(), [](const Vector2<T>& a, const Vector2<T>& b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    });
    points.erase(std::unique(points.begin(), points.end()), points.end());

    const std::size_t n = points.size();
    if (n < 3) {
        return points;
    }

    const auto cross2 = [](const Vector2<T>& o, const Vector2<T>& a,
                           const Vector2<T>& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };

    std::vector<Vector2<T>> hull(2 * n);
    std::size_t k = 0;

    for (std::size_t i = 0; i < n; ++i) {
        while (k >= 2 && cross2(hull[k - 2], hull[k - 1], points[i]) <= T{0}) {
            --k;
        }
        hull[k++] = points[i];
    }

    const std::size_t lowerHullEnd = k + 1;
    for (std::size_t i = n - 1; i-- > 0;) {
        while (k >= lowerHullEnd && cross2(hull[k - 2], hull[k - 1], points[i]) <= T{0}) {
            --k;
        }
        hull[k++] = points[i];
    }

    hull.resize(k - 1);
    return hull;
}

namespace detail {

template <std::floating_point T>
struct ConvexHullFace3D {
    std::size_t a;
    std::size_t b;
    std::size_t c;
    Vector3<T> normal;
};

/// Builds a face with `normal` guaranteed to point away from `opposite`
/// (checked directly against that point, not assumed): the initial
/// tetrahedron's four faces each have exactly one of the tetrahedron's own
/// other vertices to check against, which is enough to fix every face's
/// orientation without knowing the hull's centroid.
template <std::floating_point T>
[[nodiscard]] ConvexHullFace3D<T> makeOutwardFace(std::span<const Vector3<T>> points,
                                                  std::size_t a, std::size_t b,
                                                  std::size_t c, std::size_t opposite) {
    Vector3<T> normal = cross(points[b] - points[a], points[c] - points[a]);
    if (dot(normal, points[opposite] - points[a]) > T{0}) {
        std::swap(b, c);
        normal = -normal;
    }
    return ConvexHullFace3D<T>{a, b, c, normalized(normal)};
}

}  // namespace detail

/// The 3D convex hull of `points`, by the incremental algorithm: start from
/// a tetrahedron built from four of the points, then add every remaining
/// point in turn, removing every face it sees past (its "visible" faces,
/// the ones whose outward side it is on) and patching the hole with new
/// faces from the exposed boundary (the "horizon") to the new point.
///
/// The horizon is found without maintaining a face-adjacency graph: every
/// visible face contributes its three directed edges, and a directed edge
/// survives as part of the horizon exactly when its reverse does not also
/// appear among them (an edge shared between two visible faces cancels;
/// one shared between a visible and a surviving face does not). Reusing the
/// horizon edge's own direction for the new face keeps every face's
/// outward orientation consistent without a separate check, the same
/// property `makeOutwardFace` establishes for the initial tetrahedron by
/// checking directly instead.
///
/// A point already inside the current hull (visible to no face) is simply
/// skipped, matching what the true hull excludes it for either way. Fewer
/// than four points, or four points with zero enclosed volume (coplanar),
/// has no well-defined polyhedron; this returns an empty hull in either
/// case rather than guessing at a degenerate one.
template <std::floating_point T>
[[nodiscard]] std::vector<Triangle3<T>> convexHull3D(std::span<const Vector3<T>> points) {
    using Face = detail::ConvexHullFace3D<T>;
    const T epsilon = std::numeric_limits<T>::epsilon() * T{1000};

    if (points.size() < 4) {
        return {};
    }

    // p0, p1 the two points farthest apart give a stable starting edge; p2 the
    // point farthest from the line through them; p3 the point farthest from
    // their plane, by absolute distance (the hull needs it whichever side it
    // is on). Zero volume there means every point is coplanar.
    std::size_t p0 = 0;
    std::size_t p1 = 1;
    T bestDistance = distanceSquared(points[0], points[1]);
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1; j < points.size(); ++j) {
            const T d = distanceSquared(points[i], points[j]);
            if (d > bestDistance) {
                bestDistance = d;
                p0 = i;
                p1 = j;
            }
        }
    }

    std::size_t p2 = 0;
    T bestLineDistance{-1};
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i == p0 || i == p1) {
            continue;
        }
        const T d = lengthSquared(cross(points[p1] - points[p0], points[i] - points[p0]));
        if (d > bestLineDistance) {
            bestLineDistance = d;
            p2 = i;
        }
    }

    const Vector3<T> planeNormal =
        cross(points[p1] - points[p0], points[p2] - points[p0]);
    std::size_t p3 = 0;
    T bestPlaneDistance{-1};
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i == p0 || i == p1 || i == p2) {
            continue;
        }
        const T d = detail::absOf(dot(planeNormal, points[i] - points[p0]));
        if (d > bestPlaneDistance) {
            bestPlaneDistance = d;
            p3 = i;
        }
    }
    if (bestPlaneDistance < epsilon) {
        return {};
    }

    std::vector<Face> faces;
    faces.push_back(detail::makeOutwardFace<T>(points, p0, p1, p2, p3));
    faces.push_back(detail::makeOutwardFace<T>(points, p0, p3, p1, p2));
    faces.push_back(detail::makeOutwardFace<T>(points, p0, p2, p3, p1));
    faces.push_back(detail::makeOutwardFace<T>(points, p1, p3, p2, p0));

    for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
        if (pointIndex == p0 || pointIndex == p1 || pointIndex == p2 ||
            pointIndex == p3) {
            continue;
        }
        const Vector3<T>& p = points[pointIndex];

        std::vector<std::size_t> visible;
        for (std::size_t f = 0; f < faces.size(); ++f) {
            if (dot(faces[f].normal, p - points[faces[f].a]) > epsilon) {
                visible.push_back(f);
            }
        }
        if (visible.empty()) {
            continue;
        }

        std::vector<std::pair<std::size_t, std::size_t>> edges;
        edges.reserve(visible.size() * 3);
        for (std::size_t f : visible) {
            const Face& face = faces[f];
            edges.emplace_back(face.a, face.b);
            edges.emplace_back(face.b, face.c);
            edges.emplace_back(face.c, face.a);
        }

        std::vector<std::pair<std::size_t, std::size_t>> horizon;
        for (const auto& edge : edges) {
            const std::pair<std::size_t, std::size_t> reverseEdge{edge.second,
                                                                  edge.first};
            if (std::find(edges.begin(), edges.end(), reverseEdge) == edges.end()) {
                horizon.push_back(edge);
            }
        }

        std::vector<Face> keptFaces;
        keptFaces.reserve(faces.size());
        for (std::size_t f = 0; f < faces.size(); ++f) {
            if (std::find(visible.begin(), visible.end(), f) == visible.end()) {
                keptFaces.push_back(faces[f]);
            }
        }
        faces = std::move(keptFaces);

        for (const auto& [u, v] : horizon) {
            const Vector3<T> normal = cross(points[v] - points[u], p - points[u]);
            faces.push_back(Face{u, v, pointIndex, normalized(normal)});
        }
    }

    std::vector<Triangle3<T>> result;
    result.reserve(faces.size());
    for (const Face& face : faces) {
        result.push_back(Triangle3<T>{points[face.a], points[face.b], points[face.c]});
    }
    return result;
}

}  // namespace ysq
