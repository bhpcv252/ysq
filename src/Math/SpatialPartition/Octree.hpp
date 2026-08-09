#pragma once

#include <Math/Geometry/Intersection.hpp>
#include <Math/Geometry/Primitives.hpp>
#include <Math/Vector3.hpp>

#include <array>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace ysq {

/// A region octree over a fixed set of axis-aligned boxes, one per object:
/// the space-partitioning counterpart to `Bvh3`. `Bvh3` splits the objects
/// themselves at their median (object partitioning); an octree instead
/// splits a fixed geometric region into eight equal octants (space
/// partitioning), so its cell boundaries never move once `worldBounds` is
/// chosen, which is what a caller wants when cells need to stay meaningful
/// across incremental updates or spatial hashing, rather than only for a
/// single build-once static query pass. Same object-with-`AABB3` model and
/// query interface as `Bvh3`, so either is a drop-in choice.
template <std::floating_point T>
class Octree3 {
public:
    Octree3(std::vector<AABB3<T>> bounds, AABB3<T> worldBounds,
            std::size_t maxObjectsPerNode = 8, int maxDepth = 8)
        : m_bounds(std::move(bounds)),
          m_maxObjectsPerNode(maxObjectsPerNode),
          m_maxDepth(maxDepth) {
        std::vector<std::size_t> indices(m_bounds.size());
        std::iota(indices.begin(), indices.end(), std::size_t{0});
        m_root = build(worldBounds, std::move(indices), 0);
    }

    /// Every object index whose box overlaps `region`.
    [[nodiscard]] std::vector<std::size_t> overlapQuery(const AABB3<T>& region) const {
        std::vector<std::size_t> result;
        overlapRecursive(m_root, region, result);
        return result;
    }

    /// Every object index whose box `ray` crosses: a broad-phase result,
    /// same convention as `Bvh3::rayQuery`.
    [[nodiscard]] std::vector<std::size_t> rayQuery(const Ray3<T>& ray) const {
        std::vector<std::size_t> result;
        rayRecursive(m_root, ray, result);
        return result;
    }

private:
    /// `objects` holds two different things depending on whether this node
    /// has children: at a leaf (no children built), every object that fell
    /// in this cell; at an internal node, only the objects whose box
    /// straddles the split and so could not be handed down to a single
    /// child unambiguously. Either way, `bounds` is the cell's own fixed
    /// geometric extent, not a tight fit around its objects the way
    /// `Bvh3::Node::bounds` is -- a query must still test each object's own
    /// box, not just trust that overlapping the cell means overlapping the
    /// object.
    struct Node {
        AABB3<T> bounds;
        std::vector<std::size_t> objects;
        std::array<int, 8> children{-1, -1, -1, -1, -1, -1, -1, -1};
    };

    std::vector<AABB3<T>> m_bounds;
    std::vector<Node> m_nodes;
    int m_root = -1;
    std::size_t m_maxObjectsPerNode;
    int m_maxDepth;

    /// Which side of `centerCoord` the interval `[boxMin, boxMax]` falls
    /// entirely on: 0 (low), 1 (high), or -1 if it straddles.
    [[nodiscard]] static int axisSide(T boxMin, T boxMax, T centerCoord) {
        if (boxMax <= centerCoord) {
            return 0;
        }
        if (boxMin >= centerCoord) {
            return 1;
        }
        return -1;
    }

    /// The single octant (0-7, one bit per axis) that fully contains `box`
    /// given `region`'s center, or -1 if `box` straddles the split on any
    /// axis and so cannot be handed to one child alone.
    [[nodiscard]] static int octantContaining(const AABB3<T>& box,
                                              const Vector3<T>& center) {
        const int sideX = axisSide(box.min.x, box.max.x, center.x);
        if (sideX < 0) {
            return -1;
        }
        const int sideY = axisSide(box.min.y, box.max.y, center.y);
        if (sideY < 0) {
            return -1;
        }
        const int sideZ = axisSide(box.min.z, box.max.z, center.z);
        if (sideZ < 0) {
            return -1;
        }
        return sideX | (sideY << 1) | (sideZ << 2);
    }

    [[nodiscard]] static AABB3<T> octantBounds(const AABB3<T>& region,
                                               const Vector3<T>& center, int octant) {
        AABB3<T> result;
        result.min.x = (octant & 1) ? center.x : region.min.x;
        result.max.x = (octant & 1) ? region.max.x : center.x;
        result.min.y = (octant & 2) ? center.y : region.min.y;
        result.max.y = (octant & 2) ? region.max.y : center.y;
        result.min.z = (octant & 4) ? center.z : region.min.z;
        result.max.z = (octant & 4) ? region.max.z : center.z;
        return result;
    }

    /// Splits `region` into its eight octants only if `indices` still has
    /// more than `m_maxObjectsPerNode` entries and `depth` allows it,
    /// bucketing each object into the one octant that fully contains it, or
    /// leaving it at this node when it straddles every candidate split.
    /// Reads `m_nodes[nodeIndex]` fresh after each recursive call rather
    /// than holding a reference across it, the same reallocation-safety
    /// convention `KdTree3::build` and `Bvh3::build` already use.
    int build(const AABB3<T>& region, std::vector<std::size_t> indices, int depth) {
        const int nodeIndex = static_cast<int>(m_nodes.size());
        // Two initializers, not three: `children` is left to its own
        // default member initializer (every entry -1) rather than being
        // overridden by an explicit `{}`, which would zero-initialize it
        // instead -- indistinguishable from a real node index 0 and fatal,
        // since node 0 is always the root.
        m_nodes.push_back(Node{region, {}});

        if (indices.size() <= m_maxObjectsPerNode || depth >= m_maxDepth) {
            m_nodes[static_cast<std::size_t>(nodeIndex)].objects = std::move(indices);
            return nodeIndex;
        }

        const Vector3<T> center = (region.min + region.max) * T{0.5};
        std::array<std::vector<std::size_t>, 8> buckets;
        std::vector<std::size_t> straddling;

        for (std::size_t index : indices) {
            const int octant = octantContaining(m_bounds[index], center);
            if (octant < 0) {
                straddling.push_back(index);
            } else {
                buckets[static_cast<std::size_t>(octant)].push_back(index);
            }
        }

        m_nodes[static_cast<std::size_t>(nodeIndex)].objects = std::move(straddling);

        for (int octant = 0; octant < 8; ++octant) {
            std::vector<std::size_t>& bucket = buckets[static_cast<std::size_t>(octant)];
            if (bucket.empty()) {
                continue;
            }
            const AABB3<T> childRegion = octantBounds(region, center, octant);
            const int child = build(childRegion, std::move(bucket), depth + 1);
            m_nodes[static_cast<std::size_t>(nodeIndex)]
                .children[static_cast<std::size_t>(octant)] = child;
        }

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
        for (std::size_t index : node.objects) {
            if (intersects(m_bounds[index], region)) {
                result.push_back(index);
            }
        }
        for (int child : node.children) {
            overlapRecursive(child, region, result);
        }
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
        for (std::size_t index : node.objects) {
            if (intersect(ray, m_bounds[index]).has_value()) {
                result.push_back(index);
            }
        }
        for (int child : node.children) {
            rayRecursive(child, ray, result);
        }
    }
};

}  // namespace ysq
