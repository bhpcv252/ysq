#include <Math/Geometry/Queries.hpp>

#include <Math/Geometry/Primitives.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

using ysq::AABB3;
using ysq::OBB3;
using ysq::Plane3;
using ysq::Segment3;
using ysq::Triangle3;
using ysq::Vec2;
using ysq::Vec3;

TEST(MathGeometryQueries, ClosestPointOnSegmentClampsToTheNearerEndpoint) {
    const Segment3<double> segment{Vec3{0.0, 0.0, 0.0}, Vec3{10.0, 0.0, 0.0}};
    const Vec3 point{-5.0, 3.0, 0.0};

    const Vec3 closest = ysq::closestPoint(segment, point);
    EXPECT_NEAR(closest.x, 0.0, 1e-12);
    EXPECT_NEAR(closest.y, 0.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnSegmentProjectsAnInteriorPoint) {
    const Segment3<double> segment{Vec3{0.0, 0.0, 0.0}, Vec3{10.0, 0.0, 0.0}};
    const Vec3 point{4.0, 3.0, 0.0};

    const Vec3 closest = ysq::closestPoint(segment, point);
    EXPECT_NEAR(closest.x, 4.0, 1e-12);
    EXPECT_NEAR(closest.y, 0.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnAZeroLengthSegmentIsThatPoint) {
    const Segment3<double> segment{Vec3{2.0, 2.0, 2.0}, Vec3{2.0, 2.0, 2.0}};
    const Vec3 closest = ysq::closestPoint(segment, Vec3{0.0, 0.0, 0.0});
    EXPECT_NEAR(closest.x, 2.0, 1e-12);
    EXPECT_NEAR(closest.y, 2.0, 1e-12);
    EXPECT_NEAR(closest.z, 2.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnAPlaneIsTheProjectionAlongItsNormal) {
    const Plane3<double> plane{Vec3{0.0, 1.0, 0.0}, 0.0};  // y = 0
    const Vec3 closest = ysq::closestPoint(plane, Vec3{3.0, 5.0, -2.0});
    EXPECT_NEAR(closest.x, 3.0, 1e-12);
    EXPECT_NEAR(closest.y, 0.0, 1e-12);
    EXPECT_NEAR(closest.z, -2.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnAnAABBClampsEachAxisIndependently) {
    const AABB3<double> box{Vec3{-1.0, -1.0, -1.0}, Vec3{1.0, 1.0, 1.0}};
    const Vec3 closest = ysq::closestPoint(box, Vec3{5.0, 0.0, -5.0});
    EXPECT_NEAR(closest.x, 1.0, 1e-12);
    EXPECT_NEAR(closest.y, 0.0, 1e-12);
    EXPECT_NEAR(closest.z, -1.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnAnAABBForAnInteriorPointIsThatPoint) {
    const AABB3<double> box{Vec3{-1.0, -1.0, -1.0}, Vec3{1.0, 1.0, 1.0}};
    const Vec3 point{0.2, -0.3, 0.5};
    const Vec3 closest = ysq::closestPoint(box, point);
    EXPECT_NEAR(closest.x, point.x, 1e-12);
    EXPECT_NEAR(closest.y, point.y, 1e-12);
    EXPECT_NEAR(closest.z, point.z, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnATriangleReturnsAVertexOutsideItsCorner) {
    const Triangle3<double> triangle{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
                                     Vec3{0.0, 1.0, 0.0}};
    const Vec3 closest = ysq::closestPoint(triangle, Vec3{-1.0, -1.0, 0.0});
    EXPECT_NEAR(closest.x, 0.0, 1e-12);
    EXPECT_NEAR(closest.y, 0.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnATriangleReturnsAnEdgePoint) {
    const Triangle3<double> triangle{Vec3{0.0, 0.0, 0.0}, Vec3{4.0, 0.0, 0.0},
                                     Vec3{0.0, 4.0, 0.0}};
    // Directly below the midpoint of the hypotenuse-adjacent leg (the x axis edge).
    const Vec3 closest = ysq::closestPoint(triangle, Vec3{2.0, -3.0, 0.0});
    EXPECT_NEAR(closest.x, 2.0, 1e-12);
    EXPECT_NEAR(closest.y, 0.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnATriangleForAPointAboveTheFaceProjectsOntoIt) {
    const Triangle3<double> triangle{Vec3{0.0, 0.0, 0.0}, Vec3{4.0, 0.0, 0.0},
                                     Vec3{0.0, 4.0, 0.0}};
    const Vec3 closest = ysq::closestPoint(triangle, Vec3{1.0, 1.0, 5.0});
    EXPECT_NEAR(closest.x, 1.0, 1e-9);
    EXPECT_NEAR(closest.y, 1.0, 1e-9);
    EXPECT_NEAR(closest.z, 0.0, 1e-9);
}

TEST(MathGeometryQueries, ClosestPointOnAnObbForTheCenterIsTheCenter) {
    const double angle = ysq::kPi<double> / 4.0;
    const Vec3 bx{std::cos(angle), std::sin(angle), 0.0};
    const Vec3 by{-std::sin(angle), std::cos(angle), 0.0};
    const OBB3<double> box{
        Vec3{0.0, 0.0, 0.0}, {bx, by, Vec3::unitZ()}, Vec3{1.0, 1.0, 1.0}};

    const Vec3 closest = ysq::closestPoint(box, Vec3{0.0, 0.0, 0.0});
    EXPECT_NEAR(closest.x, 0.0, 1e-12);
    EXPECT_NEAR(closest.y, 0.0, 1e-12);
    EXPECT_NEAR(closest.z, 0.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnAnObbClampsToACornerOutsideBothAxes) {
    // The diamond footprint (a square rotated 45 degrees) has a vertex at
    // (sqrt(2), 0, 0); a point straight out along +x clamps to both local
    // axes at once and lands there.
    const double angle = ysq::kPi<double> / 4.0;
    const Vec3 bx{std::cos(angle), std::sin(angle), 0.0};
    const Vec3 by{-std::sin(angle), std::cos(angle), 0.0};
    const OBB3<double> box{
        Vec3{0.0, 0.0, 0.0}, {bx, by, Vec3::unitZ()}, Vec3{1.0, 1.0, 1.0}};

    const Vec3 closest = ysq::closestPoint(box, Vec3{5.0, 0.0, 0.0});
    EXPECT_NEAR(closest.x, std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(closest.y, 0.0, 1e-12);
    EXPECT_NEAR(closest.z, 0.0, 1e-12);
}

TEST(MathGeometryQueries, ClosestPointOnAnObbClampsToAFaceNotACorner) {
    // A point three units out along bx alone clamps on that axis only, so
    // the result is the face point at bx itself, not a corner.
    const double angle = ysq::kPi<double> / 4.0;
    const Vec3 bx{std::cos(angle), std::sin(angle), 0.0};
    const Vec3 by{-std::sin(angle), std::cos(angle), 0.0};
    const OBB3<double> box{
        Vec3{0.0, 0.0, 0.0}, {bx, by, Vec3::unitZ()}, Vec3{1.0, 1.0, 1.0}};

    const Vec3 closest = ysq::closestPoint(box, bx * 3.0);
    EXPECT_NEAR(closest.x, bx.x, 1e-12);
    EXPECT_NEAR(closest.y, bx.y, 1e-12);
    EXPECT_NEAR(closest.z, bx.z, 1e-12);
}

TEST(MathGeometryQueries, PointInPolygonIsTrueForAPointInsideASquare) {
    const std::array<Vec2, 4> square{Vec2{0.0, 0.0}, Vec2{4.0, 0.0}, Vec2{4.0, 4.0},
                                     Vec2{0.0, 4.0}};
    EXPECT_TRUE(ysq::pointInPolygon<double>(square, Vec2{2.0, 2.0}));
}

TEST(MathGeometryQueries, PointInPolygonIsFalseForAPointOutsideASquare) {
    const std::array<Vec2, 4> square{Vec2{0.0, 0.0}, Vec2{4.0, 0.0}, Vec2{4.0, 4.0},
                                     Vec2{0.0, 4.0}};
    EXPECT_FALSE(ysq::pointInPolygon<double>(square, Vec2{10.0, 2.0}));
}

TEST(MathGeometryQueries, PointInPolygonHandlesANonConvexShape) {
    // An L-shape; the notch at (3, 3) is outside the polygon even though it
    // is inside the L's overall bounding box.
    const std::array<Vec2, 6> lShape{Vec2{0.0, 0.0}, Vec2{4.0, 0.0}, Vec2{4.0, 2.0},
                                     Vec2{2.0, 2.0}, Vec2{2.0, 4.0}, Vec2{0.0, 4.0}};
    EXPECT_TRUE(ysq::pointInPolygon<double>(lShape, Vec2{1.0, 1.0}));
    EXPECT_FALSE(ysq::pointInPolygon<double>(lShape, Vec2{3.0, 3.0}));
}

}  // namespace
