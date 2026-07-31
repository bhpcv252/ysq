#include <Math/Intersection.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <optional>

namespace {

using ysq::Ray3;
using ysq::Sphere3;
using ysq::Vec3;

TEST(MathIntersection, RayThroughTheCenterHitsAtRadiusFromOrigin) {
    const Ray3<double> ray{Vec3{-5.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 2.0};

    const std::optional<double> hit = ysq::intersect(ray, sphere);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 3.0, 1e-12);
}

TEST(MathIntersection, RayEntirelyMissingTheSphereReturnsNullopt) {
    const Ray3<double> ray{Vec3{-5.0, 5.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 2.0};

    EXPECT_FALSE(ysq::intersect(ray, sphere).has_value());
}

TEST(MathIntersection, RayOriginatingInsideTheSphereHitsGoingOutward) {
    const Ray3<double> ray{Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 2.0};

    const std::optional<double> hit = ysq::intersect(ray, sphere);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 2.0, 1e-12);
}

TEST(MathIntersection, SphereEntirelyBehindTheOriginReturnsNullopt) {
    const Ray3<double> ray{Vec3{5.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{-5.0, 0.0, 0.0}, 2.0};

    EXPECT_FALSE(ysq::intersect(ray, sphere).has_value());
}

TEST(MathIntersection, ATangentRayHitsAtExactlyOnePointNotTwoDistinctOnes) {
    const Ray3<double> ray{Vec3{-5.0, 2.0, 0.0}, Vec3{1.0, 0.0, 0.0}};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 2.0};

    const std::optional<double> hit = ysq::intersect(ray, sphere);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(*hit, 5.0, 1e-9);
}

TEST(MathIntersection, SegmentIntersectionIsFalseWhenTheSphereIsPastTheEndpoint) {
    // The infinite ray hits the sphere, but not within the from-to segment.
    const Vec3 from{-1.0, 0.0, 0.0};
    const Vec3 to{-0.5, 0.0, 0.0};
    const Sphere3<double> sphere{Vec3{5.0, 0.0, 0.0}, 1.0};

    EXPECT_FALSE(ysq::segmentIntersectsSphere(from, to, sphere));
}

TEST(MathIntersection, SegmentIntersectionIsTrueWhenTheSphereSitsBetweenTheEndpoints) {
    const Vec3 from{-5.0, 0.0, 0.0};
    const Vec3 to{5.0, 0.0, 0.0};
    const Sphere3<double> sphere{Vec3{0.0, 0.0, 0.0}, 1.0};

    EXPECT_TRUE(ysq::segmentIntersectsSphere(from, to, sphere));
}

TEST(MathIntersection, SegmentIntersectionIsFalseWhenTheEndpointsAreBothInFront) {
    const Vec3 from{2.0, 0.0, 0.0};
    const Vec3 to{5.0, 0.0, 0.0};
    const Sphere3<double> sphere{Vec3{-5.0, 0.0, 0.0}, 1.0};

    EXPECT_FALSE(ysq::segmentIntersectsSphere(from, to, sphere));
}

}  // namespace
