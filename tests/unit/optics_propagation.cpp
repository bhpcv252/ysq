#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Optics/Propagation.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Minkowski.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Units/Constants.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Vec3;
using ysq::Vec4;

TEST(OpticsPropagation, NullTangentIsActuallyNullInMinkowski) {
    const ysq::Minkowski flat;
    const Vec4 at{};
    const Vec4 k = ysq::nullTangent(flat, at, Vec3{3.0, -1.0, 2.0});
    EXPECT_NEAR(ysq::metricProduct(flat, at, k, k), 0.0, 1e-9);
    // Future-directed.
    EXPECT_GT(k.x, 0.0);
}

TEST(OpticsPropagation, NullTangentIsActuallyNullInSchwarzschild) {
    const ysq::Schwarzschild schwarzschild{ysq::GravitationalParameter{5.0e14}};
    const Vec4 at{0.0, 20.0 * schwarzschild.schwarzschildRadius(), 1.4, 0.2};
    const Vec4 k = ysq::nullTangent(schwarzschild, at, Vec3{-1.0, 0.05, 0.0});
    EXPECT_NEAR(ysq::metricProduct(schwarzschild, at, k, k), 0.0, 1e-6);
    EXPECT_GT(k.x, 0.0);
}

TEST(OpticsPropagation, PropagationInMinkowskiIsAStraightLine) {
    const ysq::Minkowski flat;
    const Vec4 start{1.0, 2.0, -3.0, 0.5};
    const Vec4 k = ysq::nullTangent(flat, start, Vec3{1.0, 0.0, 0.0});

    const ysq::PhaseState<Vec4> ray{start, k};
    const double affineInterval = 50.0;
    const ysq::PhaseState<Vec4> end = ysq::propagate(flat, ray, affineInterval, 25);

    const Vec4 expected = start + k * affineInterval;
    EXPECT_VEC_NEAR(end.position, expected, 1e-6);
    EXPECT_VEC_NEAR(end.velocity, k, 1e-9);
}

TEST(OpticsPropagation, ObserverSeesTheStepCountPlusOneCallbacks) {
    const ysq::Minkowski flat;
    const Vec4 start{};
    const Vec4 k = ysq::nullTangent(flat, start, Vec3{1.0, 0.0, 0.0});
    const ysq::PhaseState<Vec4> ray{start, k};

    int calls = 0;
    const ysq::PhaseState<Vec4> result = ysq::propagate(
        flat, ray, 10.0, 7, [&](double, const ysq::PhaseState<Vec4>&) { ++calls; });
    static_cast<void>(result);
    EXPECT_EQ(calls, 8);
}

}  // namespace
