#include <Math/SpatialPartition/Octree.hpp>

#include <Math/Geometry/Intersection.hpp>
#include <Math/Geometry/Primitives.hpp>
#include <Math/Random.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace {

using ysq::AABB3;
using ysq::Ray3;
using ysq::Vec3;

}  // namespace

TEST(MathOctree, OverlapQueryFindsAllAndOnlyOverlappingBoxes) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}},
        AABB3<double>{Vec3{5.0, 5.0, 5.0}, Vec3{6.0, 6.0, 6.0}},
        AABB3<double>{Vec3{0.5, 0.5, 0.5}, Vec3{1.5, 1.5, 1.5}}};
    const AABB3<double> worldBounds{Vec3{-10.0, -10.0, -10.0}, Vec3{10.0, 10.0, 10.0}};
    const ysq::Octree3<double> octree(boxes, worldBounds);

    std::vector<std::size_t> found =
        octree.overlapQuery(AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{2.0, 2.0, 2.0}});
    std::sort(found.begin(), found.end());

    EXPECT_EQ(found, (std::vector<std::size_t>{0, 2}));
}

TEST(MathOctree, OverlapQueryFindsNothingForARegionThatMissesEveryBox) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}}};
    const AABB3<double> worldBounds{Vec3{-200.0, -200.0, -200.0},
                                    Vec3{200.0, 200.0, 200.0}};
    const ysq::Octree3<double> octree(boxes, worldBounds);

    EXPECT_TRUE(octree
                    .overlapQuery(AABB3<double>{Vec3{100.0, 100.0, 100.0},
                                                Vec3{101.0, 101.0, 101.0}})
                    .empty());
}

TEST(MathOctree, OverlapQueryHandlesASingleBoxTree) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}}};
    const AABB3<double> worldBounds{Vec3{-10.0, -10.0, -10.0}, Vec3{10.0, 10.0, 10.0}};
    const ysq::Octree3<double> octree(boxes, worldBounds);

    const std::vector<std::size_t> found =
        octree.overlapQuery(AABB3<double>{Vec3{0.5, 0.5, 0.5}, Vec3{2.0, 2.0, 2.0}});
    EXPECT_EQ(found, (std::vector<std::size_t>{0}));
}

TEST(MathOctree, OverlapQueryAgreesWithBruteForceOnManyRandomBoxes) {
    std::vector<AABB3<double>> boxes;
    ysq::RandomEngine engine = ysq::makeRandomEngine(7);
    for (int i = 0; i < 100; ++i) {
        const Vec3 center{ysq::uniformReal(engine, -20.0, 20.0),
                          ysq::uniformReal(engine, -20.0, 20.0),
                          ysq::uniformReal(engine, -20.0, 20.0)};
        const Vec3 halfExtent{ysq::uniformReal(engine, 0.1, 2.0),
                              ysq::uniformReal(engine, 0.1, 2.0),
                              ysq::uniformReal(engine, 0.1, 2.0)};
        boxes.push_back(AABB3<double>{center - halfExtent, center + halfExtent});
    }
    const AABB3<double> worldBounds{Vec3{-25.0, -25.0, -25.0}, Vec3{25.0, 25.0, 25.0}};
    const ysq::Octree3<double> octree(boxes, worldBounds);

    const AABB3<double> region{Vec3{-5.0, -5.0, -5.0}, Vec3{5.0, 5.0, 5.0}};
    std::vector<std::size_t> octreeResult = octree.overlapQuery(region);
    std::sort(octreeResult.begin(), octreeResult.end());

    std::vector<std::size_t> bruteForce;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        if (ysq::intersects(boxes[i], region)) {
            bruteForce.push_back(i);
        }
    }

    EXPECT_EQ(octreeResult, bruteForce);
}

TEST(MathOctree, RayQueryFindsTheBoxItCrosses) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}},
        AABB3<double>{Vec3{5.0, 5.0, 5.0}, Vec3{6.0, 6.0, 6.0}}};
    const AABB3<double> worldBounds{Vec3{-10.0, -10.0, -10.0}, Vec3{10.0, 10.0, 10.0}};
    const ysq::Octree3<double> octree(boxes, worldBounds);

    const Ray3<double> ray{Vec3{-5.0, 0.5, 0.5}, Vec3{1.0, 0.0, 0.0}};
    const std::vector<std::size_t> found = octree.rayQuery(ray);
    EXPECT_EQ(found, (std::vector<std::size_t>{0}));
}

TEST(MathOctree, RayQueryFindsNothingForARayThatMissesEveryBox) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}}};
    const AABB3<double> worldBounds{Vec3{-10.0, -10.0, -10.0}, Vec3{10.0, 10.0, 10.0}};
    const ysq::Octree3<double> octree(boxes, worldBounds);

    const Ray3<double> ray{Vec3{-5.0, 10.0, 10.0}, Vec3{1.0, 0.0, 0.0}};
    EXPECT_TRUE(octree.rayQuery(ray).empty());
}

TEST(MathOctree, AnObjectStraddlingTheRootSplitStaysFindableFromAnAncestorNode) {
    // A box centered on the world origin straddles every one of the first
    // split's three axes at once, so it can never be handed down to a
    // single octant -- it stays at the root, not at any leaf -- while a
    // small box safely inside one octant does descend.
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{-5.0, -5.0, -5.0}, Vec3{5.0, 5.0, 5.0}},
        AABB3<double>{Vec3{7.0, 7.0, 7.0}, Vec3{7.5, 7.5, 7.5}}};
    const AABB3<double> worldBounds{Vec3{-10.0, -10.0, -10.0}, Vec3{10.0, 10.0, 10.0}};
    const ysq::Octree3<double> octree(boxes, worldBounds, 1, 8);

    std::vector<std::size_t> found =
        octree.overlapQuery(AABB3<double>{Vec3{-0.1, -0.1, -0.1}, Vec3{0.1, 0.1, 0.1}});
    EXPECT_EQ(found, (std::vector<std::size_t>{0}));

    // Box 0 only reaches to 5 on each axis, so a query near box 1 (which
    // fully descended into its own octant) finds only box 1.
    found = octree.overlapQuery(AABB3<double>{Vec3{7.2, 7.2, 7.2}, Vec3{7.3, 7.3, 7.3}});
    EXPECT_EQ(found, (std::vector<std::size_t>{1}));
}

TEST(MathOctree, MaxDepthBoundsRecursionForManyCoincidentBoxes) {
    // Every box sits at the same location, so no split ever separates them
    // -- without maxDepth, recursion would never terminate on its own.
    std::vector<AABB3<double>> boxes;
    for (int i = 0; i < 200; ++i) {
        boxes.push_back(AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{0.01, 0.01, 0.01}});
    }
    const AABB3<double> worldBounds{Vec3{-10.0, -10.0, -10.0}, Vec3{10.0, 10.0, 10.0}};
    const ysq::Octree3<double> octree(boxes, worldBounds, 4, 6);

    const std::vector<std::size_t> found =
        octree.overlapQuery(AABB3<double>{Vec3{-1.0, -1.0, -1.0}, Vec3{1.0, 1.0, 1.0}});
    EXPECT_EQ(found.size(), boxes.size());
}
