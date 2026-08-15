#include <Compute/CPU/CpuBackend.hpp>
#include <Math/Grid3D.hpp>
#include <Math/Multigrid.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

/// A harmonic function, d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2 = 0 exactly:
/// u = x^2 - y^2. A manufactured-solution test, fully decoupled from GR or
/// puncture data -- `Math/Multigrid.hpp` has to be correct as general
/// elliptic-solver infrastructure on its own, not merely "correct enough"
/// for the one nonlinear equation it happens to be used for elsewhere.
double exactSolution(double x, double y, double) {
    return x * x - y * y;
}

double cellCoordinate(std::ptrdiff_t index, double spacing, double half) {
    return (static_cast<double>(index) + 0.5) * spacing - half;
}

TEST(Multigrid, ConvergesToAKnownHarmonicFunction) {
    const std::size_t cellCount = 16;
    const double spacing = 0.125;  // half-extent = 1.0
    const std::size_t ghostCells = 1;

    const auto applyOperator = [](const ysq::Grid3D<double>& u, std::ptrdiff_t i,
                                  std::ptrdiff_t j, std::ptrdiff_t k, double h) {
        return (u(i + 1, j, k) + u(i - 1, j, k) + u(i, j + 1, k) + u(i, j - 1, k) +
                u(i, j, k + 1) + u(i, j, k - 1) - 6.0 * u(i, j, k)) /
               (h * h);
    };

    const auto relaxPoint = [](ysq::Grid3D<double>& u, std::ptrdiff_t i, std::ptrdiff_t j,
                               std::ptrdiff_t k, double h, double target) {
        const double neighborSum = u(i + 1, j, k) + u(i - 1, j, k) + u(i, j + 1, k) +
                                   u(i, j - 1, k) + u(i, j, k + 1) + u(i, j, k - 1);
        u(i, j, k) = (neighborSum - target * h * h) / 6.0;
    };

    const auto applyBoundary = [](ysq::Grid3D<double>& u) {
        const auto nx = static_cast<std::ptrdiff_t>(u.cellCountX());
        const auto ny = static_cast<std::ptrdiff_t>(u.cellCountY());
        const auto nz = static_cast<std::ptrdiff_t>(u.cellCountZ());
        const auto ghost = static_cast<std::ptrdiff_t>(u.ghostCells());
        const double h = u.spacing();
        const std::size_t halfCells = u.cellCountX() / 2;
        const double half = static_cast<double>(halfCells) * h;

        for (std::ptrdiff_t g = 1; g <= ghost; ++g) {
            for (std::ptrdiff_t j = -ghost; j < ny + ghost; ++j) {
                for (std::ptrdiff_t k = -ghost; k < nz + ghost; ++k) {
                    u(-g, j, k) = exactSolution(cellCoordinate(-g, h, half),
                                                cellCoordinate(j, h, half),
                                                cellCoordinate(k, h, half));
                    u(nx - 1 + g, j, k) = exactSolution(
                        cellCoordinate(nx - 1 + g, h, half), cellCoordinate(j, h, half),
                        cellCoordinate(k, h, half));
                }
            }
        }
        for (std::ptrdiff_t g = 1; g <= ghost; ++g) {
            for (std::ptrdiff_t i = -ghost; i < nx + ghost; ++i) {
                for (std::ptrdiff_t k = -ghost; k < nz + ghost; ++k) {
                    u(i, -g, k) = exactSolution(cellCoordinate(i, h, half),
                                                cellCoordinate(-g, h, half),
                                                cellCoordinate(k, h, half));
                    u(i, ny - 1 + g, k) = exactSolution(
                        cellCoordinate(i, h, half), cellCoordinate(ny - 1 + g, h, half),
                        cellCoordinate(k, h, half));
                }
            }
        }
        for (std::ptrdiff_t g = 1; g <= ghost; ++g) {
            for (std::ptrdiff_t i = -ghost; i < nx + ghost; ++i) {
                for (std::ptrdiff_t j = -ghost; j < ny + ghost; ++j) {
                    u(i, j, -g) = exactSolution(cellCoordinate(i, h, half),
                                                cellCoordinate(j, h, half),
                                                cellCoordinate(-g, h, half));
                    u(i, j, nz - 1 + g) = exactSolution(
                        cellCoordinate(i, h, half), cellCoordinate(j, h, half),
                        cellCoordinate(nz - 1 + g, h, half));
                }
            }
        }
    };

    ysq::Grid3D<double> u(cellCount, cellCount, cellCount, spacing, ghostCells);
    const ysq::MultigridResult result =
        ysq::solveFAS(u, spacing, applyOperator, relaxPoint, applyBoundary);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.vCyclesUsed, 20);

    const std::size_t halfCells = cellCount / 2;
    const double half = static_cast<double>(halfCells) * spacing;
    double maxError = 0.0;
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(cellCount); ++i) {
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(cellCount); ++j) {
            for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(cellCount); ++k) {
                const double expected = exactSolution(cellCoordinate(i, spacing, half),
                                                      cellCoordinate(j, spacing, half),
                                                      cellCoordinate(k, spacing, half));
                maxError = std::max(maxError, std::abs(u(i, j, k) - expected));
            }
        }
    }
    EXPECT_LT(maxError, 1.0e-8);
}

TEST(Multigrid, RestrictAndProlongateAtLargeNAgreeWithTheComputeCpuReference) {
    // Above Math/Multigrid.hpp's own GPU dispatch threshold, so
    // detail::restrictGrid/prolongateAndAdd exercise whichever compute
    // backend is actually available. ysq::CpuBackend is called directly as
    // an independent reference (its own agreement with every GPU backend is
    // already covered by tests/integration/compute_backends_agree.cpp; this
    // test only checks that Multigrid.hpp's marshaling is wired correctly).
    // 162^3 = 4251528, above kMultigridGpuDispatchThreshold (4194304;
    // measured by benchmarks/compute_thresholds.cpp).
    constexpr std::size_t n = 162;
    constexpr double spacing = 0.1;

    ysq::Grid3D<double> fine(n, n, n, spacing, 0);
    std::vector<float> fineData(n * n * n);
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(n); ++j) {
            for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(n); ++k) {
                const double value = std::sin(0.1 * static_cast<double>(i)) +
                                     static_cast<double>(j) -
                                     static_cast<double>(k) * 0.5;
                fine(i, j, k) = value;
                fineData[(static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)) *
                             n +
                         static_cast<std::size_t>(k)] = static_cast<float>(value);
            }
        }
    }

    const ysq::Grid3D<double> coarse = ysq::detail::restrictGrid(fine);

    const ysq::CpuBackend cpu;
    constexpr std::size_t cn = n / 2;
    std::vector<float> coarseReference(cn * cn * cn);
    cpu.multigridRestrict3D(fineData, n, n, n, coarseReference);

    for (const std::size_t i : {std::size_t{0}, cn / 2, cn - 1}) {
        for (const std::size_t j : {std::size_t{0}, cn / 2, cn - 1}) {
            for (const std::size_t k : {std::size_t{0}, cn / 2, cn - 1}) {
                const std::size_t flat = (i * cn + j) * cn + k;
                EXPECT_NEAR(coarse(static_cast<std::ptrdiff_t>(i),
                                   static_cast<std::ptrdiff_t>(j),
                                   static_cast<std::ptrdiff_t>(k)),
                            coarseReference[flat], 1e-3)
                    << "coarse cell " << i << "," << j << "," << k;
            }
        }
    }

    // Built from coarseReference (not the GPU-computed coarse above), so
    // this stage's own check is isolated from any float32-rounding
    // difference the restrict stage's actual backend introduced.
    ysq::Grid3D<double> coarseFromReference(cn, cn, cn, spacing * 2.0, 0);
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(cn); ++i) {
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(cn); ++j) {
            for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(cn); ++k) {
                const std::size_t flat =
                    (static_cast<std::size_t>(i) * cn + static_cast<std::size_t>(j)) *
                        cn +
                    static_cast<std::size_t>(k);
                coarseFromReference(i, j, k) = static_cast<double>(coarseReference[flat]);
            }
        }
    }

    ysq::Grid3D<double> fineAfterProlongate = fine;
    ysq::detail::prolongateAndAdd(fineAfterProlongate, coarseFromReference);

    std::vector<float> nextFineReference(n * n * n);
    cpu.multigridProlongateAndAdd3D(fineData, coarseReference, n, n, n,
                                    nextFineReference);

    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        for (const std::size_t j : {std::size_t{0}, n / 2, n - 1}) {
            for (const std::size_t k : {std::size_t{0}, n / 2, n - 1}) {
                const std::size_t flat = (i * n + j) * n + k;
                EXPECT_NEAR(fineAfterProlongate(static_cast<std::ptrdiff_t>(i),
                                                static_cast<std::ptrdiff_t>(j),
                                                static_cast<std::ptrdiff_t>(k)),
                            nextFineReference[flat], 1e-3)
                    << "fine cell " << i << "," << j << "," << k;
            }
        }
    }
}

}  // namespace
