#include <Math/Optimization.hpp>

#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ysq::Vec2;
using ysq::Vec3;

}  // namespace

TEST(MathOptimization, BacktrackingLineSearchSatisfiesTheArmijoCondition) {
    const auto paraboloid = [](const Vec2& v) { return dot(v, v); };
    const Vec2 x{2.0, 2.0};
    const Vec2 gradientAtX = 2.0 * x;
    const Vec2 direction = -gradientAtX;

    const double step =
        ysq::backtrackingLineSearch(paraboloid, x, direction, gradientAtX);

    const double fx = paraboloid(x);
    const double directionalDerivative = dot(gradientAtX, direction);
    EXPECT_LE(paraboloid(x + direction * step), fx + 1e-4 * step * directionalDerivative);
}

TEST(MathOptimization, GradientDescentFindsTheMinimumOfAParaboloid) {
    // f(x, y) = (x-1)^2 + (y+2)^2, minimum at (1, -2).
    const auto f = [](const Vec2& v) {
        const double dx = v.x - 1.0;
        const double dy = v.y + 2.0;
        return dx * dx + dy * dy;
    };

    const Vec2 result = ysq::gradientDescent(f, Vec2{0.0, 0.0});

    EXPECT_NEAR(result.x, 1.0, 1e-3);
    EXPECT_NEAR(result.y, -2.0, 1e-3);
}

TEST(MathOptimization, GradientDescentFindsTheMinimumOfARosenbrockLikeBowl) {
    // A convex bowl with unequal curvature per axis, sensitive to a step size
    // that isn't adapted per-direction: f(x, y) = 5x^2 + y^2, minimum at (0, 0).
    const auto f = [](const Vec2& v) { return 5.0 * v.x * v.x + v.y * v.y; };

    const Vec2 result = ysq::gradientDescent(f, Vec2{3.0, 3.0});

    EXPECT_NEAR(result.x, 0.0, 1e-3);
    EXPECT_NEAR(result.y, 0.0, 1e-3);
}

TEST(MathOptimization, NelderMeadFindsTheMinimumOfAParaboloidWithNoGradient) {
    const auto f = [](const Vec3& v) {
        const Vec3 offset = v - Vec3{1.0, 2.0, -1.0};
        return dot(offset, offset);
    };

    const Vec3 result = ysq::nelderMead(f, Vec3{0.0, 0.0, 0.0});

    EXPECT_NEAR(result.x, 1.0, 1e-2);
    EXPECT_NEAR(result.y, 2.0, 1e-2);
    EXPECT_NEAR(result.z, -1.0, 1e-2);
}

TEST(MathOptimization,
     NelderMeadHandlesADiscontinuousObjectiveGradientDescentCannotDifferentiate) {
    // Minimizing the absolute value at each coordinate: not differentiable at
    // the minimum itself, the case Nelder-Mead exists for.
    const auto f = [](const Vec2& v) {
        return std::abs(v.x - 3.0) + std::abs(v.y + 1.0);
    };

    const Vec2 result = ysq::nelderMead(f, Vec2{0.0, 0.0}, 0.5, 1e-10, 5000);

    EXPECT_NEAR(result.x, 3.0, 0.05);
    EXPECT_NEAR(result.y, -1.0, 0.05);
}
