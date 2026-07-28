#include <Physics/Gravity/BarnesHut.hpp>

#include <Physics/Gravity/Newtonian.hpp>

#include <Math/Vector3.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ysq {

namespace {

[[nodiscard]] std::size_t octantOf(const Vec3& point, const Vec3& center) {
    std::size_t index = 0;
    if (point.x >= center.x) {
        index |= 1u;
    }
    if (point.y >= center.y) {
        index |= 2u;
    }
    if (point.z >= center.z) {
        index |= 4u;
    }
    return index;
}

[[nodiscard]] Vec3 childCenter(const Vec3& parentCenter, double parentHalfWidth,
                               std::size_t octant) {
    const double offset = parentHalfWidth / 2.0;
    return Vec3{parentCenter.x + ((octant & 1u) ? offset : -offset),
                parentCenter.y + ((octant & 2u) ? offset : -offset),
                parentCenter.z + ((octant & 4u) ? offset : -offset)};
}

/// The tree itself: built fresh from one snapshot of positions, queried, and
/// discarded. Node indices are `int` with -1 as the "absent" sentinel, since
/// that reads far better than a magic std::size_t value; at() is the one
/// place that bridges to the vector's actual size_t indexing.
class Octree {
public:
    Octree(const NBodyState& positions, std::span<const double> gravitationalParameters);

    [[nodiscard]] Vec3 accelerationAt(std::size_t queryIndex, const NBodyState& positions,
                                      double openingAngle, double softeningSquared) const;

private:
    // A depth this deep only happens when many bodies sit at (near-)identical
    // positions, since every other case separates into its own octant well
    // before here. Past it, insert() stops subdividing and merges instead,
    // which trades an unresolvable, physically meaningless case for a
    // guarantee that insertion terminates.
    static constexpr int kMaxDepth = 40;

    struct Node {
        Vec3 center{};
        double halfWidth = 0.0;
        Vec3 centerOfMass{};
        double totalGm = 0.0;
        std::array<int, 8> children{-1, -1, -1, -1, -1, -1, -1, -1};
        /// >= 0 only while this leaf holds exactly one, still-identifiable
        /// body. Cleared to -1 on subdivision and on the depth-guard merge.
        int bodyIndex = -1;
        bool isLeaf = true;
        bool occupied = false;
        /// Populated only by the depth-guard merge below, with every body
        /// folded into this otherwise-unresolvable leaf: bodyIndex alone
        /// cannot identify "self" once more than one body shares it.
        std::vector<int> mergedBodies;
    };

    std::vector<Node> m_nodes;
    std::span<const double> m_gm;
    int m_root = -1;

    [[nodiscard]] Node& at(int index) { return m_nodes[static_cast<std::size_t>(index)]; }
    [[nodiscard]] const Node& at(int index) const {
        return m_nodes[static_cast<std::size_t>(index)];
    }

    [[nodiscard]] int newNode(const Vec3& center, double halfWidth);
    void insert(int nodeIndex, std::size_t bodyIndex, const NBodyState& positions,
                int depth);
    void insertIntoChild(int nodeIndex, std::size_t bodyIndex, const Vec3& center,
                         double halfWidth, const NBodyState& positions, int depth);
    [[nodiscard]] Vec3 traverse(const Vec3& queryPosition, std::size_t queryIndex,
                                int nodeIndex, double openingAngle,
                                double softeningSquared) const;
};

Octree::Octree(const NBodyState& positions,
               std::span<const double> gravitationalParameters)
    : m_gm(gravitationalParameters) {
    if (positions.size() == 0) {
        return;
    }

    Vec3 minCorner = positions[0];
    Vec3 maxCorner = positions[0];
    for (std::size_t i = 1; i < positions.size(); ++i) {
        minCorner = min(minCorner, positions[i]);
        maxCorner = max(maxCorner, positions[i]);
    }

    const Vec3 center = (minCorner + maxCorner) / 2.0;
    const Vec3 extent = maxCorner - minCorner;
    const double maxExtent = std::max({extent.x, extent.y, extent.z});
    // A small multiplicative pad keeps a body exactly on the bounding box's
    // surface strictly inside it, and a floor of 1 metre keeps a cube of
    // positive size even when every body coincides.
    const double halfWidth = std::max(maxExtent, 1.0) * 0.5 * 1.001;

    m_root = newNode(center, halfWidth);
    for (std::size_t i = 0; i < positions.size(); ++i) {
        insert(m_root, i, positions, 0);
    }
}

int Octree::newNode(const Vec3& center, double halfWidth) {
    m_nodes.push_back(Node{});
    Node& node = m_nodes.back();
    node.center = center;
    node.halfWidth = halfWidth;
    return static_cast<int>(m_nodes.size() - 1);
}

void Octree::insert(int nodeIndex, std::size_t bodyIndex, const NBodyState& positions,
                    int depth) {
    const double gm = m_gm[bodyIndex];
    const Vec3 pos = positions[bodyIndex];

    // Read what insertion needs before any call that might grow m_nodes and
    // invalidate a reference into it (newNode does, via push_back).
    const Vec3 center = at(nodeIndex).center;
    const double halfWidth = at(nodeIndex).halfWidth;
    const bool isLeaf = at(nodeIndex).isLeaf;
    const bool wasOccupied = at(nodeIndex).occupied;
    const int existingBody = at(nodeIndex).bodyIndex;

    if (!wasOccupied) {
        at(nodeIndex).bodyIndex = static_cast<int>(bodyIndex);
        at(nodeIndex).totalGm = gm;
        at(nodeIndex).centerOfMass = pos;
        at(nodeIndex).occupied = true;
        return;
    }

    // The node gains another body either way from here: fold it into the
    // aggregate first, no push_back involved yet so the reference is safe.
    {
        Node& node = at(nodeIndex);
        const double newTotal = node.totalGm + gm;
        node.centerOfMass = (node.centerOfMass * node.totalGm + pos * gm) / newTotal;
        node.totalGm = newTotal;
    }

    if (isLeaf && depth >= kMaxDepth) {
        Node& node = at(nodeIndex);
        if (node.mergedBodies.empty()) {
            node.mergedBodies.push_back(existingBody);
        }
        node.mergedBodies.push_back(static_cast<int>(bodyIndex));
        node.bodyIndex = -1;
        return;
    }

    if (isLeaf) {
        // Held exactly one body until now: turn it into an internal node and
        // push both bodies down into children.
        at(nodeIndex).isLeaf = false;
        at(nodeIndex).bodyIndex = -1;
        insertIntoChild(nodeIndex, static_cast<std::size_t>(existingBody), center,
                        halfWidth, positions, depth);
        insertIntoChild(nodeIndex, bodyIndex, center, halfWidth, positions, depth);
        return;
    }

    insertIntoChild(nodeIndex, bodyIndex, center, halfWidth, positions, depth);
}

void Octree::insertIntoChild(int nodeIndex, std::size_t bodyIndex, const Vec3& center,
                             double halfWidth, const NBodyState& positions, int depth) {
    const std::size_t octant = octantOf(positions[bodyIndex], center);
    int childIndex = at(nodeIndex).children[octant];
    if (childIndex < 0) {
        childIndex = newNode(childCenter(center, halfWidth, octant), halfWidth / 2.0);
        at(nodeIndex).children[octant] = childIndex;
    }
    insert(childIndex, bodyIndex, positions, depth + 1);
}

Vec3 Octree::accelerationAt(std::size_t queryIndex, const NBodyState& positions,
                            double openingAngle, double softeningSquared) const {
    if (m_root < 0) {
        return Vec3{};
    }
    return traverse(positions[queryIndex], queryIndex, m_root, openingAngle,
                    softeningSquared);
}

Vec3 Octree::traverse(const Vec3& queryPosition, std::size_t queryIndex, int nodeIndex,
                      double openingAngle, double softeningSquared) const {
    const Node& node = at(nodeIndex);
    if (!node.occupied) {
        return Vec3{};
    }
    // A leaf holding exactly the body being computed for contributes nothing
    // to its own acceleration; the same holds for a depth-guard-merged leaf
    // that happens to hold it among several bodies it can no longer tell
    // apart.
    if (node.isLeaf) {
        if (node.bodyIndex == static_cast<int>(queryIndex)) {
            return Vec3{};
        }
        for (const int merged : node.mergedBodies) {
            if (merged == static_cast<int>(queryIndex)) {
                return Vec3{};
            }
        }
    }

    const Vec3 delta = node.centerOfMass - queryPosition;
    const double distanceSquared = lengthSquared(delta);

    // A leaf is always evaluated exactly, whatever the opening angle: there
    // is nothing left to approximate once it is one body, or an unresolved
    // merged aggregate from the depth guard in insert().
    const double width = 2.0 * node.halfWidth;
    const bool accept =
        node.isLeaf || (width * width < openingAngle * openingAngle * distanceSquared);

    if (accept) {
        if (distanceSquared <= 0.0 && softeningSquared <= 0.0) {
            // Coincident with an unsoftened point mass: no direction is
            // defined. Contributes nothing rather than a NaN.
            return Vec3{};
        }
        const double r2 = distanceSquared + softeningSquared;
        const double r = std::sqrt(r2);
        return delta * (node.totalGm / (r2 * r));
    }

    Vec3 total{};
    for (int child : node.children) {
        if (child >= 0) {
            total += traverse(queryPosition, queryIndex, child, openingAngle,
                              softeningSquared);
        }
    }
    return total;
}

}  // namespace

BarnesHutTree::BarnesHutTree(std::span<const Body> bodies, double openingAngle,
                             Length softening)
    : m_openingAngle(openingAngle),
      m_softeningSquared(softening.value() * softening.value()) {
    m_gravitationalParameters.reserve(bodies.size());
    for (const Body& body : bodies) {
        m_gravitationalParameters.push_back(constants::G.value() * body.mass.value());
    }
}

NBodyState BarnesHutTree::operator()(double, const NBodyState& positions) const {
    assert(positions.size() == m_gravitationalParameters.size());
    const Octree tree(positions, m_gravitationalParameters);
    NBodyState result(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        result[i] = tree.accelerationAt(i, positions, m_openingAngle, m_softeningSquared);
    }
    return result;
}

}  // namespace ysq
