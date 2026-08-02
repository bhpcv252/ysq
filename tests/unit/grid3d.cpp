#include <Math/Grid3D.hpp>

#include <gtest/gtest.h>

#include <cstddef>

namespace {

TEST(Grid3D, InteriorIndexingRoundTrips) {
    ysq::Grid3D<double> grid(3, 4, 5, 0.5, 2);
    for (std::ptrdiff_t i = 0; i < 3; ++i) {
        for (std::ptrdiff_t j = 0; j < 4; ++j) {
            for (std::ptrdiff_t k = 0; k < 5; ++k) {
                grid(i, j, k) = static_cast<double>(100 * i + 10 * j + k);
            }
        }
    }
    for (std::ptrdiff_t i = 0; i < 3; ++i) {
        for (std::ptrdiff_t j = 0; j < 4; ++j) {
            for (std::ptrdiff_t k = 0; k < 5; ++k) {
                EXPECT_DOUBLE_EQ(grid(i, j, k), static_cast<double>(100 * i + 10 * j + k));
            }
        }
    }
}

TEST(Grid3D, ReportsItsOwnShape) {
    const ysq::Grid3D<double> grid(3, 4, 5, 0.25, 2);
    EXPECT_EQ(grid.cellCountX(), 3u);
    EXPECT_EQ(grid.cellCountY(), 4u);
    EXPECT_EQ(grid.cellCountZ(), 5u);
    EXPECT_EQ(grid.ghostCells(), 2u);
    EXPECT_DOUBLE_EQ(grid.spacing(), 0.25);
}

TEST(Grid3D, GhostCellsAreWritableAndReadableWithinBounds) {
    ysq::Grid3D<double> grid(2, 2, 2, 1.0, 2);
    grid(-2, -2, -2) = 7.0;
    grid(3, 3, 3) = 9.0;  // cellCount + ghostCells - 1, the far edge of the ghost region
    EXPECT_DOUBLE_EQ(grid(-2, -2, -2), 7.0);
    EXPECT_DOUBLE_EQ(grid(3, 3, 3), 9.0);
}

TEST(Grid3D, PeriodicBoundaryWrapsEachAxisIndependently) {
    ysq::Grid3D<double> grid(3, 3, 3, 1.0, 1);
    for (std::ptrdiff_t i = 0; i < 3; ++i) {
        for (std::ptrdiff_t j = 0; j < 3; ++j) {
            for (std::ptrdiff_t k = 0; k < 3; ++k) {
                grid(i, j, k) = static_cast<double>(100 * i + 10 * j + k);
            }
        }
    }
    grid.applyPeriodicBoundary();

    // The low ghost cell along x should equal the high interior edge along
    // x, at every (j, k) including the corners, which need every axis's own
    // wrap applied.
    for (std::ptrdiff_t j = -1; j < 4; ++j) {
        for (std::ptrdiff_t k = -1; k < 4; ++k) {
            const std::ptrdiff_t wrappedJ = (j < 0) ? 2 : (j > 2 ? 0 : j);
            const std::ptrdiff_t wrappedK = (k < 0) ? 2 : (k > 2 ? 0 : k);
            EXPECT_DOUBLE_EQ(grid(-1, j, k), grid(2, wrappedJ, wrappedK));
            EXPECT_DOUBLE_EQ(grid(3, j, k), grid(0, wrappedJ, wrappedK));
        }
    }
}

TEST(Grid3D, VectorSpaceOperationsAreElementwise) {
    ysq::Grid3D<double> a(2, 2, 2, 1.0, 1);
    ysq::Grid3D<double> b(2, 2, 2, 1.0, 1);
    for (std::ptrdiff_t i = 0; i < 2; ++i) {
        for (std::ptrdiff_t j = 0; j < 2; ++j) {
            for (std::ptrdiff_t k = 0; k < 2; ++k) {
                a(i, j, k) = static_cast<double>(i + j + k);
                b(i, j, k) = static_cast<double>(2 * (i + j + k) + 1);
            }
        }
    }

    const ysq::Grid3D<double> sum = a + b;
    const ysq::Grid3D<double> difference = b - a;
    const ysq::Grid3D<double> scaled = a * 2.0;
    const ysq::Grid3D<double> scaledOther = 2.0 * a;

    for (std::ptrdiff_t i = 0; i < 2; ++i) {
        for (std::ptrdiff_t j = 0; j < 2; ++j) {
            for (std::ptrdiff_t k = 0; k < 2; ++k) {
                EXPECT_DOUBLE_EQ(sum(i, j, k), a(i, j, k) + b(i, j, k));
                EXPECT_DOUBLE_EQ(difference(i, j, k), b(i, j, k) - a(i, j, k));
                EXPECT_DOUBLE_EQ(scaled(i, j, k), a(i, j, k) * 2.0);
                EXPECT_DOUBLE_EQ(scaledOther(i, j, k), a(i, j, k) * 2.0);
            }
        }
    }

    ysq::Grid3D<double> accumulated = a;
    accumulated += b;
    accumulated *= 0.5;
    for (std::ptrdiff_t i = 0; i < 2; ++i) {
        for (std::ptrdiff_t j = 0; j < 2; ++j) {
            for (std::ptrdiff_t k = 0; k < 2; ++k) {
                EXPECT_DOUBLE_EQ(accumulated(i, j, k), 0.5 * (a(i, j, k) + b(i, j, k)));
            }
        }
    }
}

}  // namespace
