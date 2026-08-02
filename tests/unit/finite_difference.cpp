#include <Math/FiniteDifference.hpp>
#include <Math/Grid3D.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

using ysq::Axis;

/// Fills `grid` with `f` evaluated at every stored cell's coordinate,
/// including every ghost cell the widest stencil under test needs --
/// exactly the situation a real field on a real grid is in, not a
/// convenience shortcut.
template <class F>
void fillGrid(ysq::Grid3D<double>& grid, double spacing, std::ptrdiff_t ghostCells, F&& f) {
    const auto nx = static_cast<std::ptrdiff_t>(grid.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(grid.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(grid.cellCountZ());
    for (std::ptrdiff_t i = -ghostCells; i < nx + ghostCells; ++i) {
        for (std::ptrdiff_t j = -ghostCells; j < ny + ghostCells; ++j) {
            for (std::ptrdiff_t k = -ghostCells; k < nz + ghostCells; ++k) {
                const double x = static_cast<double>(i) * spacing;
                const double y = static_cast<double>(j) * spacing;
                const double z = static_cast<double>(k) * spacing;
                grid(i, j, k) = f(x, y, z);
            }
        }
    }
}

TEST(FiniteDifference, FirstDerivativeIsExactForACubic) {
    // f = x^3: f'(x) = 3x^2 exactly, and a fourth-order stencil (exact
    // through degree-4 polynomials) should reproduce it to floating-point
    // precision, not just approximately.
    const double spacing = 0.1;
    ysq::Grid3D<double> grid(1, 1, 1, spacing, 3);
    fillGrid(grid, spacing, 3, [](double x, double, double) { return x * x * x; });

    const double derivative = ysq::firstDerivative(grid, 0, 0, 0, Axis::X, spacing);
    EXPECT_NEAR(derivative, 0.0, 1.0e-9);  // at x = 0, 3x^2 = 0

    ysq::Grid3D<double> gridShifted(1, 1, 1, spacing, 3);
    fillGrid(gridShifted, spacing, 3, [](double x, double, double) {
        const double shifted = x + 0.4;
        return shifted * shifted * shifted;
    });
    const double derivativeAtShift =
        ysq::firstDerivative(gridShifted, 0, 0, 0, Axis::X, spacing);
    EXPECT_NEAR(derivativeAtShift, 3.0 * 0.4 * 0.4, 1.0e-9);
}

TEST(FiniteDifference, SecondDerivativeIsExactForAQuartic) {
    // f = x^4: f''(x) = 12x^2 exactly.
    const double spacing = 0.1;
    ysq::Grid3D<double> grid(1, 1, 1, spacing, 3);
    fillGrid(grid, spacing, 3, [](double x, double, double) {
        const double shifted = x + 0.3;
        return shifted * shifted * shifted * shifted;
    });
    const double derivative = ysq::secondDerivative(grid, 0, 0, 0, Axis::X, spacing);
    EXPECT_NEAR(derivative, 12.0 * 0.3 * 0.3, 1.0e-8);
}

TEST(FiniteDifference, MixedSecondDerivativeMatchesAKnownPolynomial) {
    // f = x^2 y^2: d^2f/dxdy = 4xy exactly.
    const double spacing = 0.1;
    ysq::Grid3D<double> grid(1, 1, 1, spacing, 3);
    fillGrid(grid, spacing, 3, [](double x, double y, double) {
        const double sx = x + 0.2;
        const double sy = y - 0.15;
        return sx * sx * sy * sy;
    });
    const double derivative =
        ysq::mixedSecondDerivative(grid, 0, 0, 0, Axis::X, Axis::Y, spacing, spacing);
    EXPECT_NEAR(derivative, 4.0 * 0.2 * (-0.15), 1.0e-8);
}

TEST(FiniteDifference, FirstDerivativeConvergesAtFourthOrder) {
    // f = sin(x): a non-polynomial function, so the fourth-order truncation
    // error is genuinely present; halving h should shrink the error by
    // 2^4 = 16, the signature of a fourth-order method.
    const auto errorAt = [](double spacing) {
        ysq::Grid3D<double> grid(1, 1, 1, spacing, 3);
        fillGrid(grid, spacing, 3, [](double x, double, double) { return std::sin(x); });
        const double derivative = ysq::firstDerivative(grid, 0, 0, 0, Axis::X, spacing);
        return std::abs(derivative - std::cos(0.0));
    };

    const double errorCoarse = errorAt(0.1);
    const double errorFine = errorAt(0.05);
    ASSERT_GT(errorCoarse, 0.0);
    ASSERT_GT(errorFine, 0.0);
    const double ratio = errorCoarse / errorFine;
    EXPECT_NEAR(ratio, 16.0, 2.0);
}

TEST(FiniteDifference, KreissOligerVanishesForAQuinticPolynomial) {
    // Sixth-order dissipation is the exact discrete sixth derivative; any
    // polynomial of degree <= 5 has a genuinely zero sixth derivative, so
    // this must return (numerically) zero, not merely something small.
    const double spacing = 0.1;
    ysq::Grid3D<double> grid(1, 1, 1, spacing, 3);
    fillGrid(grid, spacing, 3, [](double x, double, double) {
        return x * x * x * x * x - 2.0 * x * x * x + x;
    });
    const double dissipation =
        ysq::kreissOligerDissipation(grid, 0, 0, 0, Axis::X, spacing, 0.5);
    EXPECT_NEAR(dissipation, 0.0, 1.0e-8);
}

TEST(FiniteDifference, KreissOligerDampsGridScaleNoise) {
    // A checkerboard pattern, (-1)^i, is the grid's own Nyquist mode -- the
    // exact thing this dissipation exists to damp. It must not be zero
    // here, unlike the smooth polynomial case above.
    const double spacing = 0.1;
    ysq::Grid3D<double> grid(1, 1, 1, spacing, 3);
    const auto nx = static_cast<std::ptrdiff_t>(grid.cellCountX());
    for (std::ptrdiff_t i = -3; i < nx + 3; ++i) {
        grid(i, 0, 0) = (i % 2 == 0) ? 1.0 : -1.0;
    }
    const double dissipation =
        ysq::kreissOligerDissipation(grid, 0, 0, 0, Axis::X, spacing, 0.5);
    EXPECT_GT(std::abs(dissipation), 1.0);
}

TEST(FiniteDifference, ThreeDSumIsTheSumOfEachAxis) {
    const double spacing = 0.1;
    ysq::Grid3D<double> grid(1, 1, 1, spacing, 3);
    fillGrid(grid, spacing, 3, [](double x, double y, double z) {
        return x * x * x + y * y * y - 2.0 * z * z * z;
    });
    const double total = ysq::kreissOligerDissipation3D(grid, 0, 0, 0, spacing, 0.3);
    const double sumOfAxes = ysq::kreissOligerDissipation(grid, 0, 0, 0, Axis::X, spacing, 0.3) +
                             ysq::kreissOligerDissipation(grid, 0, 0, 0, Axis::Y, spacing, 0.3) +
                             ysq::kreissOligerDissipation(grid, 0, 0, 0, Axis::Z, spacing, 0.3);
    EXPECT_DOUBLE_EQ(total, sumOfAxes);
}

}  // namespace
