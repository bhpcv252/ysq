#pragma once

#include <Math/Vector3.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace ysq {

/// A k-d tree over a fixed point cloud: splits the points into two halves
/// by their median along an axis that cycles `x, y, z` with depth, so every
/// level halves the search space along whichever axis still has room to
/// discriminate. Built once, queried many times — this is the general
/// spatial index `Physics/Fluids/SPH.cpp`'s neighbor search reaches for
/// instead of the brute-force all-pairs loop a smoothing-length radius
/// query would otherwise need.
template <std::floating_point T>
class KdTree3 {
public:
    explicit KdTree3(std::vector<Vector3<T>> points) : m_points(std::move(points)) {
        std::vector<std::size_t> indices(m_points.size());
        std::iota(indices.begin(), indices.end(), std::size_t{0});
        m_nodes.reserve(m_points.size());
        m_root = build(indices, 0, indices.size(), 0);
    }

    /// Every point's index within `radius` of `queryPoint`, in no
    /// particular order.
    [[nodiscard]] std::vector<std::size_t> radiusQuery(const Vector3<T>& queryPoint,
                                                       T radius) const {
        std::vector<std::size_t> result;
        radiusQueryRecursive(m_root, queryPoint, radius * radius, result);
        return result;
    }

    /// The `k` nearest points' indices, ascending by distance to
    /// `queryPoint`. Fewer than `k` if the tree itself holds fewer points.
    [[nodiscard]] std::vector<std::size_t> nearestNeighbors(const Vector3<T>& queryPoint,
                                                            std::size_t k) const {
        std::priority_queue<std::pair<T, std::size_t>> heap;
        nearestRecursive(m_root, queryPoint, k, heap);

        std::vector<std::pair<T, std::size_t>> found;
        found.reserve(heap.size());
        while (!heap.empty()) {
            found.push_back(heap.top());
            heap.pop();
        }
        std::sort(found.begin(), found.end());

        std::vector<std::size_t> result;
        result.reserve(found.size());
        for (const auto& entry : found) {
            result.push_back(entry.second);
        }
        return result;
    }

private:
    struct Node {
        std::size_t pointIndex;
        int axis;
        int left = -1;
        int right = -1;
    };

    std::vector<Vector3<T>> m_points;
    std::vector<Node> m_nodes;
    int m_root = -1;

    /// Partitions `indices[begin, end)` around the median along `depth`'s
    /// axis with `std::nth_element` (average O(n) per level, so O(n log n)
    /// overall across every level), recursing on each half. A node's slot in
    /// `m_nodes` is reserved before its children are built and filled in
    /// after, so a reallocation triggered by a child's own `push_back`
    /// cannot invalidate a reference this function is still holding — there
    /// is none; the index is looked up fresh instead.
    int build(std::vector<std::size_t>& indices, std::size_t begin, std::size_t end,
              int depth) {
        if (begin >= end) {
            return -1;
        }

        const int axis = depth % 3;
        const std::size_t mid = begin + (end - begin) / 2;
        std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                         indices.begin() + static_cast<std::ptrdiff_t>(mid),
                         indices.begin() + static_cast<std::ptrdiff_t>(end),
                         [this, axis](std::size_t a, std::size_t b) {
                             return m_points[a][static_cast<std::size_t>(axis)] <
                                    m_points[b][static_cast<std::size_t>(axis)];
                         });

        const int nodeIndex = static_cast<int>(m_nodes.size());
        m_nodes.push_back(Node{indices[mid], axis, -1, -1});
        const int left = build(indices, begin, mid, depth + 1);
        const int right = build(indices, mid + 1, end, depth + 1);
        m_nodes[static_cast<std::size_t>(nodeIndex)].left = left;
        m_nodes[static_cast<std::size_t>(nodeIndex)].right = right;
        return nodeIndex;
    }

    /// Descends the near side of the splitting plane unconditionally, and
    /// the far side only if the plane itself is within `radius` — the
    /// distance a point on the far side would need to travel just to cross
    /// back over the plane, which bounds how close any far-side point could
    /// possibly be regardless of where exactly it sits.
    void radiusQueryRecursive(int nodeIndex, const Vector3<T>& queryPoint,
                              T radiusSquared, std::vector<std::size_t>& result) const {
        if (nodeIndex < 0) {
            return;
        }
        const Node& node = m_nodes[static_cast<std::size_t>(nodeIndex)];
        const Vector3<T>& point = m_points[node.pointIndex];

        if (distanceSquared(point, queryPoint) <= radiusSquared) {
            result.push_back(node.pointIndex);
        }

        const T diff = queryPoint[static_cast<std::size_t>(node.axis)] -
                       point[static_cast<std::size_t>(node.axis)];
        const int nearSide = (diff < T{0}) ? node.left : node.right;
        const int farSide = (diff < T{0}) ? node.right : node.left;

        radiusQueryRecursive(nearSide, queryPoint, radiusSquared, result);
        if (diff * diff <= radiusSquared) {
            radiusQueryRecursive(farSide, queryPoint, radiusSquared, result);
        }
    }

    /// The same near/far pruning as `radiusQueryRecursive`, against a
    /// shrinking radius instead of a fixed one: `heap`'s own worst
    /// (largest) distance once it holds `k` candidates, so the far side is
    /// skipped entirely once nothing there could possibly beat what is
    /// already found.
    void nearestRecursive(int nodeIndex, const Vector3<T>& queryPoint, std::size_t k,
                          std::priority_queue<std::pair<T, std::size_t>>& heap) const {
        if (nodeIndex < 0) {
            return;
        }
        const Node& node = m_nodes[static_cast<std::size_t>(nodeIndex)];
        const Vector3<T>& point = m_points[node.pointIndex];
        const T d = distanceSquared(point, queryPoint);

        if (heap.size() < k) {
            heap.emplace(d, node.pointIndex);
        } else if (d < heap.top().first) {
            heap.pop();
            heap.emplace(d, node.pointIndex);
        }

        const T diff = queryPoint[static_cast<std::size_t>(node.axis)] -
                       point[static_cast<std::size_t>(node.axis)];
        const int nearSide = (diff < T{0}) ? node.left : node.right;
        const int farSide = (diff < T{0}) ? node.right : node.left;

        nearestRecursive(nearSide, queryPoint, k, heap);
        if (heap.size() < k || diff * diff < heap.top().first) {
            nearestRecursive(farSide, queryPoint, k, heap);
        }
    }
};

}  // namespace ysq
