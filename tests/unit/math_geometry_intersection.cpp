#include <Math/Geometry/Intersection.hpp>

#include <Math/Geometry/Primitives.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>

namespace {

using ysq::AABB3;
using ysq::OBB3;
using ysq::Plane3;
using ysq::Ray3;
using ysq::Sphere3;
using ysq::Triangle3;
using ysq::Vec3;

TEST(MathGeometryIntersection, RayThroughTheCenterHitsAtRadiusFromOrigin) {
    const Ray3<double> ray{Vec3{-5.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 2.0};

    const std::optional<double> hit = ysq::intersect(ray, sphere);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 3.0, 1e-12);
}

TEST(MathGeometryIntersection, RayEntirelyMissingTheSphereReturnsNullopt) {
    const Ray3<double> ray{Vec3{-5.0, 5.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 2.0};

    EXPECT_FALSE(ysq::intersect(ray, sphere).has_value());
}

TEST(MathGeometryIntersection, RayOriginatingInsideTheSphereHitsGoingOutward) {
    const Ray3<double> ray{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 2.0};

    const std::optional<double> hit = ysq::intersect(ray, sphere);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 2.0, 1e-12);
}

TEST(MathGeometryIntersection, SphereEntirelyBehindTheOriginReturnsNullopt) {
    const Ray3<double> ray{Vec3{5.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{-5.0, 0.0, 0.0}, 2.0};

    EXPECT_FALSE(ysq::intersect(ray, sphere).has_value());
}

TEST(MathGeometryIntersection, ATangentRayHitsAtExactlyOnePointNotTwoDistinctOnes) {
    const Ray3<double> ray{Vec3{-5.0, 2.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 2.0};

    const std::optional<double> hit = ysq::intersect(ray, sphere);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 5.0, 1e-9);
}

TEST(MathGeometryIntersection, SegmentIntersectionIsFalseWhenTheSphereIsPastTheEndpoint) {
    const Vec3 from{-1.0, 0.0, 0.0};
    const Vec3 to{-0.5, 0.0, 0.0};
    const Sphere3<double> sphere{Vec3{5.0, 0.0, 0.0}, 1.0};

    EXPECT_FALSE(ysq::segmentIntersectsSphere(from, to, sphere));
}

TEST(MathGeometryIntersection,
     SegmentIntersectionIsTrueWhenTheSphereSitsBetweenTheEndpoints) {
    const Vec3 from{-5.0, 0.0, 0.0};
    const Vec3 to{5.0, 0.0, 0.0};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 1.0};

    EXPECT_TRUE(ysq::segmentIntersectsSphere(from, to, sphere));
}

TEST(MathGeometryIntersection, SegmentIntersectionIsFalseWhenTheEndpointsAreBothInFront) {
    const Vec3 from{2.0, 0.0, 0.0};
    const Vec3 to{5.0, 0.0, 0.0};
    const Sphere3<double> sphere{Vec3{-5.0, 0.0, 0.0}, 1.0};

    EXPECT_FALSE(ysq::segmentIntersectsSphere(from, to, sphere));
}

TEST(MathGeometryIntersection, RayHitsAPlaneAtTheExpectedDistance) {
    const Ray3<double> ray{Vec3{0.0, 5.0, 0.0}, Vec3{0.0, -1.0, 0.0}};
    const Plane3<double> plane{Vec3{0.0, 1.0, 0.0}, 2.0};  // y = 2

    const std::optional<double> hit = ysq::intersect(ray, plane);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 3.0, 1e-12);
}

TEST(MathGeometryIntersection, RayParallelToAPlaneMisses) {
    const Ray3<double> ray{Vec3{0.0, 5.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Plane3<double> plane{Vec3{0.0, 1.0, 0.0}, 2.0};

    EXPECT_FALSE(ysq::intersect(ray, plane).has_value());
}

TEST(MathGeometryIntersection, RayAwayFromAPlaneMisses) {
    const Ray3<double> ray{Vec3{0.0, 5.0, 0.0}, Vec3{0.0, 1.0, 0.0}};
    const Plane3<double> plane{Vec3{0.0, 1.0, 0.0}, 2.0};

    EXPECT_FALSE(ysq::intersect(ray, plane).has_value());
}

TEST(MathGeometryIntersection, RayHitsATriangleThroughItsCentroid) {
    const Triangle3<double> triangle{Vec3{0.0, 0.0, 0.0}, Vec3{4.0, 0.0, 0.0},
                                     Vec3{0.0, 4.0, 0.0}};
    const Ray3<double> ray{Vec3{1.0, 1.0, 5.0}, Vec3{0.0, 0.0, -1.0}};

    const std::optional<double> hit = ysq::intersect(ray, triangle);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 5.0, 1e-12);
}

TEST(MathGeometryIntersection, RayMissesATriangleOutsideItsEdges) {
    const Triangle3<double> triangle{Vec3{0.0, 0.0, 0.0}, Vec3{4.0, 0.0, 0.0},
                                     Vec3{0.0, 4.0, 0.0}};
    const Ray3<double> ray{Vec3{10.0, 10.0, 5.0}, Vec3{0.0, 0.0, -1.0}};

    EXPECT_FALSE(ysq::intersect(ray, triangle).has_value());
}

TEST(MathGeometryIntersection, RayParallelToATriangleMisses) {
    const Triangle3<double> triangle{Vec3{0.0, 0.0, 0.0}, Vec3{4.0, 0.0, 0.0},
                                     Vec3{0.0, 4.0, 0.0}};
    const Ray3<double> ray{Vec3{1.0, 1.0, 5.0}, Vec3{1.0, 0.0, 0.0}};

    EXPECT_FALSE(ysq::intersect(ray, triangle).has_value());
}

TEST(MathGeometryIntersection, RayHitsAnAABBFromOutside) {
    const AABB3<double> box{Vec3{-1.0, -1.0, -1.0}, Vec3{1.0, 1.0, 1.0}};
    const Ray3<double> ray{Vec3{-5.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};

    const std::optional<double> hit = ysq::intersect(ray, box);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 4.0, 1e-12);
}

TEST(MathGeometryIntersection, RayOriginatingInsideAnAABBHitsTheExitFace) {
    const AABB3<double> box{Vec3{-1.0, -1.0, -1.0}, Vec3{1.0, 1.0, 1.0}};
    const Ray3<double> ray{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};

    const std::optional<double> hit = ysq::intersect(ray, box);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 1.0, 1e-12);
}

TEST(MathGeometryIntersection, RayMissingAnAABBReturnsNullopt) {
    const AABB3<double> box{Vec3{-1.0, -1.0, -1.0}, Vec3{1.0, 1.0, 1.0}};
    const Ray3<double> ray{Vec3{-5.0, 5.0, 0.0}, Vec3{1.0, 0.0, 0.0}};

    EXPECT_FALSE(ysq::intersect(ray, box).has_value());
}

TEST(MathGeometryIntersection, OverlappingSpheresIntersect) {
    const Sphere3<double> a{Vec3{0.0, 0.0, 0.0}, 2.0};
    const Sphere3<double> b{Vec3{3.0, 0.0, 0.0}, 2.0};
    EXPECT_TRUE(ysq::intersects(a, b));
}

TEST(MathGeometryIntersection, SeparatedSpheresDoNotIntersect) {
    const Sphere3<double> a{Vec3{0.0, 0.0, 0.0}, 1.0};
    const Sphere3<double> b{Vec3{10.0, 0.0, 0.0}, 1.0};
    EXPECT_FALSE(ysq::intersects(a, b));
}

TEST(MathGeometryIntersection, OverlappingAABBsIntersect) {
    const AABB3<double> a{Vec3{0.0, 0.0, 0.0}, Vec3{2.0, 2.0, 2.0}};
    const AABB3<double> b{Vec3{1.0, 1.0, 1.0}, Vec3{3.0, 3.0, 3.0}};
    EXPECT_TRUE(ysq::intersects(a, b));
}

TEST(MathGeometryIntersection, SeparatedAABBsDoNotIntersect) {
    const AABB3<double> a{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 1.0, 1.0}};
    const AABB3<double> b{Vec3{5.0, 5.0, 5.0}, Vec3{6.0, 6.0, 6.0}};
    EXPECT_FALSE(ysq::intersects(a, b));
}

TEST(MathGeometryIntersection, PlaneThroughASphereIntersects) {
    const Plane3<double> plane{Vec3{0.0, 1.0, 0.0}, 0.0};  // y = 0
    const Sphere3<double> sphere{Vec3{0.0, 0.5, 0.0}, 1.0};
    EXPECT_TRUE(ysq::intersects(plane, sphere));
}

TEST(MathGeometryIntersection, PlaneFarFromASphereDoesNotIntersect) {
    const Plane3<double> plane{Vec3{0.0, 1.0, 0.0}, 0.0};
    const Sphere3<double> sphere{Vec3{0.0, 10.0, 0.0}, 1.0};
    EXPECT_FALSE(ysq::intersects(plane, sphere));
}

TEST(MathGeometryIntersection, AxisAlignedObbAgreesWithAabbIntersectionTest) {
    const AABB3<double> boxA{Vec3{0.0, 0.0, 0.0}, Vec3{2.0, 2.0, 2.0}};
    const AABB3<double> boxBOverlap{Vec3{1.0, 1.0, 1.0}, Vec3{3.0, 3.0, 3.0}};
    const AABB3<double> boxBSeparate{Vec3{5.0, 5.0, 5.0}, Vec3{6.0, 6.0, 6.0}};

    const auto toObb = [](const AABB3<double>& box) {
        return OBB3<double>{(box.min + box.max) * 0.5,
                            {Vec3::unitX(), Vec3::unitY(), Vec3::unitZ()},
                            (box.max - box.min) * 0.5};
    };

    EXPECT_EQ(ysq::intersects(toObb(boxA), toObb(boxBOverlap)),
              ysq::intersects(boxA, boxBOverlap));
    EXPECT_EQ(ysq::intersects(toObb(boxA), toObb(boxBSeparate)),
              ysq::intersects(boxA, boxBSeparate));
    EXPECT_TRUE(ysq::intersects(toObb(boxA), boxBOverlap));
    EXPECT_FALSE(ysq::intersects(toObb(boxA), boxBSeparate));
}

TEST(MathGeometryIntersection, RotatedObbsOverlapWhenCentersAreClose) {
    const OBB3<double> a{Vec3{0.0, 0.0, 0.0},
                         {Vec3::unitX(), Vec3::unitY(), Vec3::unitZ()},
                         Vec3{1.0, 1.0, 1.0}};
    const double angle = ysq::kPi<double> / 4.0;
    const Vec3 bx{std::cos(angle), std::sin(angle), 0.0};
    const Vec3 by{-std::sin(angle), std::cos(angle), 0.0};
    const OBB3<double> bRotated{
        Vec3{1.0, 0.0, 0.0}, {bx, by, Vec3::unitZ()}, Vec3{1.0, 1.0, 1.0}};

    EXPECT_TRUE(ysq::intersects(a, bRotated));
}

TEST(MathGeometryIntersection, RotatedObbsSeparateWhenFarApart) {
    const OBB3<double> a{Vec3{0.0, 0.0, 0.0},
                         {Vec3::unitX(), Vec3::unitY(), Vec3::unitZ()},
                         Vec3{1.0, 1.0, 1.0}};
    const double angle = ysq::kPi<double> / 4.0;
    const Vec3 bx{std::cos(angle), std::sin(angle), 0.0};
    const Vec3 by{-std::sin(angle), std::cos(angle), 0.0};
    const OBB3<double> bRotated{
        Vec3{20.0, 0.0, 0.0}, {bx, by, Vec3::unitZ()}, Vec3{1.0, 1.0, 1.0}};

    EXPECT_FALSE(ysq::intersects(a, bRotated));
}

TEST(MathGeometryIntersection,
     ObbsSeparatedOnlyAlongAnEdgeCrossProductAxisDoNotIntersect) {
    using ysq::cross;
    using ysq::dot;
    using ysq::normalized;
    using ysq::rotateAbout;

    // Two thin rods with generically skew long axes: the classic case
    // where the true minimal separating direction between them is
    // cross(rodA's long axis, rodB's long axis), one of the nine
    // cross-product SAT axes and not a face normal of either. Because
    // that axis is by construction perpendicular to both long axes, each
    // rod's own long half-extent (3.0) contributes nothing to its
    // projection there, leaving only the two thin half-extents (0.1) --
    // tiny next to every face-axis threshold, which is what guarantees
    // this configuration fools a naive six-axis-only test without any
    // hand-tuned numeric coincidence.
    const Vec3 aLong = Vec3::unitZ();
    const std::array<Vec3, 3> aAxes{Vec3::unitX(), Vec3::unitY(), aLong};
    const Vec3 aHalfExtents{0.1, 0.1, 3.0};
    const OBB3<double> a{Vec3{0.0, 0.0, 0.0}, aAxes, aHalfExtents};

    // Composed about X then Y, not Y then Z: aLong is world Z, and
    // composing about Z last would make cross(aLong, bLong) collapse onto
    // a Z-rotation of unitY exactly (an algebraic identity of that
    // particular construction, not a numerical coincidence), which is
    // exactly one of b's own thin axes -- defeating the point.
    const double angle1 = 0.3;
    const double angle2 = 0.5;
    const Vec3 bx = rotateAbout(rotateAbout(Vec3::unitX(), Vec3::unitX(), angle1),
                                Vec3::unitY(), angle2);
    const Vec3 by = rotateAbout(rotateAbout(Vec3::unitY(), Vec3::unitX(), angle1),
                                Vec3::unitY(), angle2);
    const Vec3 bLong = rotateAbout(rotateAbout(Vec3::unitZ(), Vec3::unitX(), angle1),
                                   Vec3::unitY(), angle2);
    const std::array<Vec3, 3> bAxes{bx, by, bLong};
    const Vec3 bHalfExtents{0.1, 0.1, 3.0};

    const Vec3 testAxis = normalized(cross(aLong, bLong));

    // Each box's own half-extent projected onto testAxis, computed directly
    // (not through ysq::intersects) -- the exact quantity a separating-axis
    // test on testAxis compares against the center-to-center distance.
    double extentA = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        extentA += aHalfExtents[i] * std::abs(dot(aAxes[i], testAxis));
    }
    double extentB = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        extentB += bHalfExtents[i] * std::abs(dot(bAxes[i], testAxis));
    }

    const Vec3 centerSeparated = testAxis * (extentA + extentB + 0.1);
    const OBB3<double> bSeparated{centerSeparated, bAxes, bHalfExtents};

    // Confirm this configuration would fool a naive six-axis-only test:
    // every one of a's and b's own face-normal axes shows overlap, so the
    // cross-product axis is the only one that actually separates the boxes.
    for (std::size_t j = 0; j < 3; ++j) {
        const double centerProjection = std::abs(dot(centerSeparated, aAxes[j]));
        double reachB = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            reachB += bHalfExtents[i] * std::abs(dot(bAxes[i], aAxes[j]));
        }
        EXPECT_LE(centerProjection, aHalfExtents[j] + reachB)
            << "a's face axis wrongly appears separating";
    }
    for (std::size_t j = 0; j < 3; ++j) {
        const double centerProjection = std::abs(dot(centerSeparated, bAxes[j]));
        double reachA = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            reachA += aHalfExtents[i] * std::abs(dot(aAxes[i], bAxes[j]));
        }
        EXPECT_LE(centerProjection, reachA + bHalfExtents[j])
            << "b's face axis wrongly appears separating";
    }

    EXPECT_FALSE(ysq::intersects(a, bSeparated));

    const Vec3 centerOverlapping = testAxis * (extentA + extentB - 0.1);
    const OBB3<double> bOverlapping{centerOverlapping, bAxes, bHalfExtents};
    EXPECT_TRUE(ysq::intersects(a, bOverlapping));
}

TEST(MathGeometryIntersection, RayHitsARotatedObbThroughItsDiagonal) {
    const double angle = ysq::kPi<double> / 4.0;
    const Vec3 bx{std::cos(angle), std::sin(angle), 0.0};
    const Vec3 by{-std::sin(angle), std::cos(angle), 0.0};
    const OBB3<double> box{
        Vec3{0.0, 0.0, 0.0}, {bx, by, Vec3::unitZ()}, Vec3{1.0, 1.0, 1.0}};
    const Ray3<double> ray{Vec3{-5.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};

    // The diamond footprint (a square rotated 45 degrees) has a vertex at
    // (-sqrt(2), 0, 0), so a ray straight down the x axis enters there.
    const std::optional<double> hit = ysq::intersect(ray, box);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 5.0 - std::sqrt(2.0), 1e-9);
}

TEST(MathGeometryIntersection, RayMissingARotatedObbReturnsNullopt) {
    const double angle = ysq::kPi<double> / 4.0;
    const Vec3 bx{std::cos(angle), std::sin(angle), 0.0};
    const Vec3 by{-std::sin(angle), std::cos(angle), 0.0};
    const OBB3<double> box{
        Vec3{0.0, 0.0, 0.0}, {bx, by, Vec3::unitZ()}, Vec3{1.0, 1.0, 1.0}};
    const Ray3<double> ray{Vec3{-5.0, 5.0, 0.0}, Vec3{1.0, 0.0, 0.0}};

    EXPECT_FALSE(ysq::intersect(ray, box).has_value());
}

}  // namespace
