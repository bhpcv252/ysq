#include <Math/Geometry/ConvexHull.hpp>

#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using ysq::Vec2;
using ysq::Vec3;

bool containsPointNear(const std::vector<Vec2>& hull, const Vec2& point,
                       double tolerance = 1e-9) {
    return std::any_of(hull.begin(), hull.end(), [&](const Vec2& p) {
        return std::abs(p.x - point.x) < tolerance && std::abs(p.y - point.y) < tolerance;
    });
}

double signedArea(const std::vector<Vec2>& hull) {
    double area = 0.0;
    for (std::size_t i = 0; i < hull.size(); ++i) {
        const Vec2& a = hull[i];
        const Vec2& b = hull[(i + 1) % hull.size()];
        area += a.x * b.y - b.x * a.y;
    }
    return area / 2.0;
}

}  // namespace

TEST(MathGeometryConvexHull2D, SquareCornersReturnAllFourCorners) {
    const std::vector<Vec2> points{Vec2{0.0, 0.0}, Vec2{4.0, 0.0}, Vec2{4.0, 4.0},
                                   Vec2{0.0, 4.0}};
    const std::vector<Vec2> hull = ysq::convexHull2D(points);

    ASSERT_EQ(hull.size(), 4U);
    for (const Vec2& corner : points) {
        EXPECT_TRUE(containsPointNear(hull, corner));
    }
}

TEST(MathGeometryConvexHull2D, InteriorPointsAreExcluded) {
    std::vector<Vec2> points{Vec2{0.0, 0.0}, Vec2{4.0, 0.0}, Vec2{4.0, 4.0},
                             Vec2{0.0, 4.0}};
    // A cluster of points strictly inside the square.
    points.push_back(Vec2{1.0, 1.0});
    points.push_back(Vec2{2.0, 2.0});
    points.push_back(Vec2{3.0, 1.5});

    const std::vector<Vec2> hull = ysq::convexHull2D(points);

    ASSERT_EQ(hull.size(), 4U);
    EXPECT_FALSE(containsPointNear(hull, Vec2{2.0, 2.0}));
}

TEST(MathGeometryConvexHull2D, HullIsReturnedCounterclockwise) {
    const std::vector<Vec2> points{Vec2{0.0, 0.0}, Vec2{4.0, 0.0}, Vec2{4.0, 4.0},
                                   Vec2{0.0, 4.0}};
    const std::vector<Vec2> hull = ysq::convexHull2D(points);
    // The shoelace formula's signed area is positive for a counterclockwise polygon.
    EXPECT_GT(signedArea(hull), 0.0);
}

TEST(MathGeometryConvexHull2D, FewerThanThreePointsReturnsThemUnchanged) {
    const std::vector<Vec2> points{Vec2{1.0, 1.0}, Vec2{2.0, 2.0}};
    const std::vector<Vec2> hull = ysq::convexHull2D(points);
    EXPECT_EQ(hull.size(), 2U);
}

TEST(MathGeometryConvexHull2D, DuplicatePointsAreCollapsed) {
    const std::vector<Vec2> points{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{4.0, 0.0},
                                   Vec2{4.0, 4.0}, Vec2{0.0, 4.0}};
    const std::vector<Vec2> hull = ysq::convexHull2D(points);
    EXPECT_EQ(hull.size(), 4U);
}

TEST(MathGeometryConvexHull3D, CubeCornersProduceATwelveTriangleHull) {
    const std::vector<Vec3> cube{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
                                 Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0},
                                 Vec3{1.0, 1.0, 0.0}, Vec3{1.0, 0.0, 1.0},
                                 Vec3{0.0, 1.0, 1.0}, Vec3{1.0, 1.0, 1.0}};

    const std::vector<ysq::Triangle3<double>> hull = ysq::convexHull3D<double>(cube);

    // A cube's convex hull is 6 faces, 2 triangles each.
    EXPECT_EQ(hull.size(), 12U);
}

TEST(MathGeometryConvexHull3D, InteriorPointIsExcludedFromTheHull) {
    std::vector<Vec3> points{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
                             Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0},
                             Vec3{1.0, 1.0, 0.0}, Vec3{1.0, 0.0, 1.0},
                             Vec3{0.0, 1.0, 1.0}, Vec3{1.0, 1.0, 1.0}};
    const Vec3 interiorPoint{0.5, 0.5, 0.5};
    points.push_back(interiorPoint);

    const std::vector<ysq::Triangle3<double>> hull = ysq::convexHull3D<double>(points);

    // Still exactly a cube: the interior point contributes no face vertex.
    EXPECT_EQ(hull.size(), 12U);
    for (const ysq::Triangle3<double>& triangle : hull) {
        EXPECT_FALSE(distanceSquared(triangle.a, interiorPoint) < 1e-9);
        EXPECT_FALSE(distanceSquared(triangle.b, interiorPoint) < 1e-9);
        EXPECT_FALSE(distanceSquared(triangle.c, interiorPoint) < 1e-9);
    }
}

TEST(MathGeometryConvexHull3D, EveryFaceNormalPointsAwayFromTheCentroid) {
    const std::vector<Vec3> cube{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
                                 Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0},
                                 Vec3{1.0, 1.0, 0.0}, Vec3{1.0, 0.0, 1.0},
                                 Vec3{0.0, 1.0, 1.0}, Vec3{1.0, 1.0, 1.0}};
    const Vec3 centroid{0.5, 0.5, 0.5};

    const std::vector<ysq::Triangle3<double>> hull = ysq::convexHull3D<double>(cube);
    ASSERT_FALSE(hull.empty());

    for (const ysq::Triangle3<double>& triangle : hull) {
        const Vec3 normal = cross(triangle.b - triangle.a, triangle.c - triangle.a);
        const Vec3 faceCenter = (triangle.a + triangle.b + triangle.c) / 3.0;
        EXPECT_GT(dot(normal, faceCenter - centroid), 0.0);
    }
}

TEST(MathGeometryConvexHull3D, FewerThanFourPointsReturnsAnEmptyHull) {
    const std::vector<Vec3> points{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
                                   Vec3{0.0, 1.0, 0.0}};
    EXPECT_TRUE(ysq::convexHull3D<double>(points).empty());
}

TEST(MathGeometryConvexHull3D, CoplanarPointsReturnAnEmptyHull) {
    const std::vector<Vec3> points{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
                                   Vec3{0.0, 1.0, 0.0}, Vec3{1.0, 1.0, 0.0}};
    EXPECT_TRUE(ysq::convexHull3D<double>(points).empty());
}
