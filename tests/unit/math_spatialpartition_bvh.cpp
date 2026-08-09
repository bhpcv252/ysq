#include <Math/SpatialPartition/Bvh.hpp>

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

TEST(MathBvh, OverlapQueryFindsAllAndOnlyOverlappingBoxes) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}},
        AABB3<double>{Vec3{5.0, 5.0, 5.0}, Vec3{6.0, 6.0, 6.0}},
        AABB3<double>{Vec3{0.5, 0.5, 0.5}, Vec3{1.5, 1.5, 1.5}}};
    const ysq::Bvh3<double> bvh(boxes);

    std::vector<std::size_t> found =
        bvh.overlapQuery(AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{2.0, 2.0, 2.0}});
    std::sort(found.begin(), found.end());

    EXPECT_EQ(found, (std::vector<std::size_t>{0, 2}));
}

TEST(MathBvh, OverlapQueryFindsNothingForARegionThatMissesEveryBox) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}}};
    const ysq::Bvh3<double> bvh(boxes);

    EXPECT_TRUE(bvh.overlapQuery(AABB3<double>{Vec3{100.0, 100.0, 100.0},
                                               Vec3{101.0, 101.0, 101.0}})
                    .empty());
}

TEST(MathBvh, OverlapQueryHandlesASingleBoxTree) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}}};
    const ysq::Bvh3<double> bvh(boxes);

    const std::vector<std::size_t> found =
        bvh.overlapQuery(AABB3<double>{Vec3{0.5, 0.5, 0.5}, Vec3{2.0, 2.0, 2.0}});
    EXPECT_EQ(found, (std::vector<std::size_t>{0}));
}

TEST(MathBvh, OverlapQueryAgreesWithBruteForceOnManyRandomBoxes) {
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
    const ysq::Bvh3<double> bvh(boxes);

    const AABB3<double> region{Vec3{-5.0, -5.0, -5.0}, Vec3{5.0, 5.0, 5.0}};
    std::vector<std::size_t> bvhResult = bvh.overlapQuery(region);
    std::sort(bvhResult.begin(), bvhResult.end());

    std::vector<std::size_t> bruteForce;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        if (ysq::intersects(boxes[i], region)) {
            bruteForce.push_back(i);
        }
    }

    EXPECT_EQ(bvhResult, bruteForce);
}

TEST(MathBvh, RayQueryFindsTheBoxItCrosses) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}},
        AABB3<double>{Vec3{5.0, 5.0, 5.0}, Vec3{6.0, 6.0, 6.0}}};
    const ysq::Bvh3<double> bvh(boxes);

    const Ray3<double> ray{Vec3{-5.0, 0.5, 0.5}, Vec3{1.0, 0.0, 0.0}};
    const std::vector<std::size_t> found = bvh.rayQuery(ray);
    EXPECT_EQ(found, (std::vector<std::size_t>{0}));
}

TEST(MathBvh, RayQueryFindsNothingForARayThatMissesEveryBox) {
    const std::vector<AABB3<double>> boxes{
        AABB3<double>{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}}};
    const ysq::Bvh3<double> bvh(boxes);

    const Ray3<double> ray{Vec3{-5.0, 10.0, 10.0}, Vec3{1.0, 0.0, 0.0}};
    EXPECT_TRUE(bvh.rayQuery(ray).empty());
}
