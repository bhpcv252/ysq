#include <Math/SpatialPartition/KdTree.hpp>

#include <Math/Random.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace {

using ysq::Vec3;

}  // namespace

TEST(MathKdTree, RadiusQueryFindsEveryPointWithinRadiusAndNoOthers) {
    const std::vector<Vec3> points{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
                                   Vec3{0.0, 1.0, 0.0}, Vec3{10.0, 10.0, 10.0},
                                   Vec3{0.5, 0.5, 0.0}};
    const ysq::KdTree3<double> tree(points);

    std::vector<std::size_t> found = tree.radiusQuery(Vec3{0.0, 0.0, 0.0}, 1.5);
    std::sort(found.begin(), found.end());

    // Points 0, 1, 2, 4 are within 1.5 of the origin; point 3 is not.
    EXPECT_EQ(found, (std::vector<std::size_t>{0, 1, 2, 4}));
}

TEST(MathKdTree, RadiusQueryWithZeroRadiusFindsOnlyExactMatches) {
    const std::vector<Vec3> points{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const ysq::KdTree3<double> tree(points);

    const std::vector<std::size_t> found = tree.radiusQuery(Vec3{0.0, 0.0, 0.0}, 0.0);
    EXPECT_EQ(found, (std::vector<std::size_t>{0}));
}

TEST(MathKdTree, RadiusQueryOnAnEmptyTreeReturnsNothing) {
    const ysq::KdTree3<double> tree(std::vector<Vec3>{});
    EXPECT_TRUE(tree.radiusQuery(Vec3{0.0, 0.0, 0.0}, 100.0).empty());
}

TEST(MathKdTree, NearestNeighborsReturnsTheClosestPointFirst) {
    const std::vector<Vec3> points{Vec3{5.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
                                   Vec3{3.0, 0.0, 0.0}};
    const ysq::KdTree3<double> tree(points);

    const std::vector<std::size_t> nearest =
        tree.nearestNeighbors(Vec3{0.0, 0.0, 0.0}, 2);
    ASSERT_EQ(nearest.size(), 2U);
    EXPECT_EQ(nearest[0], 1U);  // (1,0,0), distance 1
    EXPECT_EQ(nearest[1], 2U);  // (3,0,0), distance 3
}

TEST(MathKdTree, NearestNeighborsRequestingMoreThanAvailableReturnsWhatExists) {
    const std::vector<Vec3> points{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const ysq::KdTree3<double> tree(points);

    EXPECT_EQ(tree.nearestNeighbors(Vec3{0.0, 0.0, 0.0}, 10).size(), 2U);
}

TEST(MathKdTree, NearestNeighborsAgreesWithBruteForceOnARandomCloud) {
    std::vector<Vec3> points;
    ysq::RandomEngine engine = ysq::makeRandomEngine(99);
    for (int i = 0; i < 200; ++i) {
        points.push_back(Vec3{ysq::uniformReal(engine, -10.0, 10.0),
                              ysq::uniformReal(engine, -10.0, 10.0),
                              ysq::uniformReal(engine, -10.0, 10.0)});
    }
    const ysq::KdTree3<double> tree(points);

    const Vec3 query{0.0, 0.0, 0.0};
    const std::vector<std::size_t> treeResult = tree.nearestNeighbors(query, 5);

    std::vector<std::pair<double, std::size_t>> bruteForce;
    for (std::size_t i = 0; i < points.size(); ++i) {
        bruteForce.emplace_back(distanceSquared(points[i], query), i);
    }
    std::sort(bruteForce.begin(), bruteForce.end());

    ASSERT_EQ(treeResult.size(), 5U);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(treeResult[i], bruteForce[i].second);
    }
}
