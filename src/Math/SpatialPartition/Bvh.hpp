#pragma once

#include <Math/Geometry/Intersection.hpp>
#include <Math/Geometry/Primitives.hpp>
#include <Math/Vector3.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace ysq {

/// A bounding volume hierarchy over a fixed set of axis-aligned boxes, one
/// per object: the bounds-based counterpart to `KdTree3`'s point-based
/// index, for exactly the case a k-d tree does not cover — objects with
/// extent (a body's collision shape, a triangle, anything with its own
/// `AABB3`) rather than dimensionless points. Built once from every
/// object's box, then queried many times, which is the shape a broad-phase
/// collision pass or a scene raycast actually has: many queries against one
/// static (for the query's duration) arrangement of objects.
template <std::floating_point T>
class Bvh3 {
public:
    explicit Bvh3(std::vector<AABB3<T>> bounds) : m_bounds(std::move(bounds)) {
        std::vector<std::size_t> indices(m_bounds.size());
        std::iota(indices.begin(), indices.end(), std::size_t{0});
        m_nodes.reserve(2 * m_bounds.size());
        m_root = build(indices, 0, indices.size());
    }

    /// Every object index whose box overlaps `region`.
    [[nodiscard]] std::vector<std::size_t> overlapQuery(const AABB3<T>& region) const {
        std::vector<std::size_t> result;
        overlapRecursive(m_root, region, result);
        return result;
    }

    /// Every object index whose box `ray` crosses: a broad-phase result, the
    /// candidates a caller then tests against each object's own real shape,
    /// not necessarily objects the ray actually hits.
    [[nodiscard]] std::vector<std::size_t> rayQuery(const Ray3<T>& ray) const {
        std::vector<std::size_t> result;
        rayRecursive(m_root, ray, result);
        return result;
    }

private:
    struct Node {
        AABB3<T> bounds;
        int left = -1;
        int right = -1;
        std::size_t objectIndex = 0;  // valid only when left < 0 && right < 0 (a leaf)
    };

    std::vector<AABB3<T>> m_bounds;
    std::vector<Node> m_nodes;
    int m_root = -1;

    [[nodiscard]] AABB3<T> boundsOf(const std::vector<std::size_t>& indices,
                                    std::size_t begin, std::size_t end) const {
        AABB3<T> result = m_bounds[indices[begin]];
        for (std::size_t i = begin + 1; i < end; ++i) {
            const AABB3<T>& box = m_bounds[indices[i]];
            result.min = min(result.min, box.min);
            result.max = max(result.max, box.max);
        }
        return result;
    }

    /// Splits along whichever axis the current range's combined bounds are
    /// longest on, at the median object center along that axis: a simple
    /// top-down split, not the surface-area heuristic a production renderer
    /// would use to minimize expected query cost, but one that already
    /// gives every query the O(log n) depth a balanced binary tree provides,
    /// which is the property broad-phase queries actually need over the
    /// O(n) alternative of testing every object.
    int build(std::vector<std::size_t>& indices, std::size_t begin, std::size_t end) {
        const std::size_t count = end - begin;
        const AABB3<T> bounds = boundsOf(indices, begin, end);

        if (count == 1) {
            const int nodeIndex = static_cast<int>(m_nodes.size());
            m_nodes.push_back(Node{bounds, -1, -1, indices[begin]});
            return nodeIndex;
        }

        const Vector3<T> extent = bounds.max - bounds.min;
        std::size_t axis = 0;
        if (extent.y > extent[axis]) {
            axis = 1;
        }
        if (extent.z > extent[axis]) {
            axis = 2;
        }

        const std::size_t mid = begin + count / 2;
        std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                         indices.begin() + static_cast<std::ptrdiff_t>(mid),
                         indices.begin() + static_cast<std::ptrdiff_t>(end),
                         [this, axis](std::size_t a, std::size_t b) {
                             const T centerA =
                                 m_bounds[a].min[axis] + m_bounds[a].max[axis];
                             const T centerB =
                                 m_bounds[b].min[axis] + m_bounds[b].max[axis];
                             return centerA < centerB;
                         });

        const int nodeIndex = static_cast<int>(m_nodes.size());
        m_nodes.push_back(Node{bounds, -1, -1, 0});
        const int left = build(indices, begin, mid);
        const int right = build(indices, mid, end);
        m_nodes[static_cast<std::size_t>(nodeIndex)].left = left;
        m_nodes[static_cast<std::size_t>(nodeIndex)].right = right;
        return nodeIndex;
    }

    void overlapRecursive(int nodeIndex, const AABB3<T>& region,
                          std::vector<std::size_t>& result) const {
        if (nodeIndex < 0) {
            return;
        }
        const Node& node = m_nodes[static_cast<std::size_t>(nodeIndex)];
        if (!intersects(node.bounds, region)) {
            return;
        }
        if (node.left < 0 && node.right < 0) {
            result.push_back(node.objectIndex);
            return;
        }
        overlapRecursive(node.left, region, result);
        overlapRecursive(node.right, region, result);
    }

    void rayRecursive(int nodeIndex, const Ray3<T>& ray,
                      std::vector<std::size_t>& result) const {
        if (nodeIndex < 0) {
            return;
        }
        const Node& node = m_nodes[static_cast<std::size_t>(nodeIndex)];
        if (!intersect(ray, node.bounds).has_value()) {
            return;
        }
        if (node.left < 0 && node.right < 0) {
            result.push_back(node.objectIndex);
            return;
        }
        rayRecursive(node.left, ray, result);
        rayRecursive(node.right, ray, result);
    }
};

}  // namespace ysq
