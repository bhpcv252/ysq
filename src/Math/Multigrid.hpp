#pragma once

#include <Compute/ComputeBackend.hpp>
#include <Math/Grid3D.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ysq {

/// A geometric multigrid V-cycle for a nonlinear elliptic equation on a
/// `Grid3D`, using the Full Approximation Scheme (FAS) -- the standard way
/// multigrid handles a genuinely nonlinear operator (as opposed to a linear
/// one, where a simpler correction-scheme multigrid would do): both the
/// residual *and* the current solution are restricted to each coarser grid,
/// so the coarse level solves the same nonlinear equation with a
/// FAS-corrected right-hand side, not a linearization of it.
///
/// General numerical method: takes any nonlinear operator, local relaxation
/// rule, and boundary condition over a `Grid3D<double>`, with no assumption
/// about what equation it is. `Physics/Spacetime/PunctureInitialData.hpp`'s
/// Hamiltonian-constraint solve is the first real consumer; nothing here
/// knows that, or needs to.
///
/// **The caller supplies three things:**
///
/// - `applyOperator(u, i, j, k, spacing) -> double`: the equation's own
///   left-hand side, `L(u)`, at one cell -- e.g. a Laplacian plus whatever
///   nonlinear term the equation has. The equation being solved is
///   `L(u) = 0`; fold any external source into `L` itself (see
///   `PunctureInitialData.cpp` for a worked example).
/// - `relaxPoint(u, i, j, k, spacing, target)`: one Gauss-Seidel-style
///   update of `u(i, j, k)` so that `applyOperator(u, i, j, k, spacing)`
///   moves toward `target` (not always exactly zero: the FAS scheme feeds
///   a nonzero `target` to coarser levels, so this cannot assume otherwise).
/// - `applyBoundary(u)`: whatever boundary condition the equation needs,
///   applied to every level's own grid (a Dirichlet condition, for
///   instance, needs no unit conversion between levels, but a caller
///   wanting something resolution-dependent is free to branch on
///   `u.spacing()`).
///
/// **Restriction and prolongation are the simplest correct choices for a
/// cell-centered grid**, not the fastest-converging ones: restriction is
/// the straight average of the 8 fine cells each coarse cell exactly
/// contains (a coarsening by exactly a factor of 2 on every axis), and
/// prolongation is piecewise-constant injection of a coarse correction
/// into those same 8 fine cells, relying on the post-smoothing sweeps to
/// clean up the resulting discontinuity. Trilinear prolongation converges
/// faster and is real, later refinement, not a correctness requirement --
/// FAS multigrid with piecewise-constant transfer operators is a
/// well-established, if not optimal, combination.
struct MultigridSettings {
    int preSmoothingSweeps = 2;
    int postSmoothingSweeps = 2;
    int maxVCycles = 50;
    double residualTolerance = 1.0e-10;
    /// Coarsening stops once any axis reaches this many cells (or stops
    /// dividing evenly by two); the coarsest level is then relaxed with
    /// many more sweeps, standing in for an exact coarse-grid solve.
    std::size_t coarsestCellCount = 4;
};

struct MultigridResult {
    int vCyclesUsed = 0;
    double finalResidual = 0.0;
    bool converged = false;
};

namespace detail {

/// Measured on the development machine (Apple Silicon, Metal backend) by
/// `benchmarks/compute_thresholds.cpp`: restriction is one dispatch,
/// straight 8-cell-average, memory-bound and trivially cheap per cell, so
/// the GPU path never won even at 128^3 (~2.1M cells) fine-grid cells, the
/// largest size tried -- this is a measured floor (double the largest size
/// tried), not an observed crossover, and higher than the original
/// provisional guess of 200000 turned out to be (O(n) work per cell stays
/// CPU-competitive even longer than that guess assumed). Keeps
/// tests/unit/multigrid.cpp's 16^3 grid on the exact double-precision CPU
/// path either way. Re-run the benchmark and update this if the reference
/// machine or backend ever changes.
inline constexpr std::size_t kMultigridGpuDispatchThreshold = 4194304;

/// The straight average of the 8 fine cells each coarse cell exactly
/// contains, for a cell-centered grid coarsened by exactly a factor of 2 on
/// every axis -- every cell count must be even, asserted below rather than
/// silently handled, since an odd count has no well-defined coarsening at
/// all (both paths below would otherwise silently drop the last plane on
/// that axis instead). Above kMultigridGpuDispatchThreshold, dispatches
/// through Compute::defaultBackend() (equation-independent, so this is the
/// same kernel regardless of which elliptic equation the caller is
/// solving); below it, the plain CPU path.
inline Grid3D<double> restrictGrid(const Grid3D<double>& fine) {
    assert(fine.cellCountX() % 2 == 0 && fine.cellCountY() % 2 == 0 &&
           fine.cellCountZ() % 2 == 0);
    const std::size_t nx = fine.cellCountX() / 2;
    const std::size_t ny = fine.cellCountY() / 2;
    const std::size_t nz = fine.cellCountZ() / 2;
    Grid3D<double> coarse(nx, ny, nz, fine.spacing() * 2.0, fine.ghostCells());

    const std::size_t fineNx = fine.cellCountX();
    const std::size_t fineNy = fine.cellCountY();
    const std::size_t fineNz = fine.cellCountZ();

    if (fineNx * fineNy * fineNz >= kMultigridGpuDispatchThreshold) {
        std::vector<float> fineData(fineNx * fineNy * fineNz);
        for (std::size_t i = 0; i < fineNx; ++i) {
            for (std::size_t j = 0; j < fineNy; ++j) {
                for (std::size_t k = 0; k < fineNz; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    fineData[(i * fineNy + j) * fineNz + k] =
                        static_cast<float>(fine(pi, pj, pk));
                }
            }
        }

        std::vector<float> coarseData(nx * ny * nz);
        defaultBackend().multigridRestrict3D(fineData, fineNx, fineNy, fineNz,
                                             coarseData);

        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    coarse(pi, pj, pk) =
                        static_cast<double>(coarseData[(i * ny + j) * nz + k]);
                }
            }
        }
        return coarse;
    }

    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nx); ++i) {
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(ny); ++j) {
            for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(nz); ++k) {
                double sum = 0.0;
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        for (int dk = 0; dk < 2; ++dk) {
                            sum += fine(2 * i + di, 2 * j + dj, 2 * k + dk);
                        }
                    }
                }
                coarse(i, j, k) = sum / 8.0;
            }
        }
    }
    return coarse;
}

/// Adds each coarse cell's correction to the 8 fine cells it came from.
/// `coarseCorrection`'s own dimensions must be exactly `fine`'s halved on
/// every axis (so `fine`'s must be even), asserted below rather than
/// silently handled: both paths index into `coarseCorrection` computed
/// from `fine`'s cell counts (the GPU kernel does so internally, matching
/// `multigridProlongateAndAdd3D`'s own documented layout convention), so a
/// mismatched `coarseCorrection` would read or write out of bounds rather
/// than merely produce a wrong answer. Same dispatch policy as
/// restrictGrid: above kMultigridGpuDispatchThreshold (on the fine grid's
/// own cell count), routes through Compute::defaultBackend(); below it,
/// the plain CPU path.
inline void prolongateAndAdd(Grid3D<double>& fine,
                             const Grid3D<double>& coarseCorrection) {
    assert(fine.cellCountX() % 2 == 0 && fine.cellCountY() % 2 == 0 &&
           fine.cellCountZ() % 2 == 0);
    assert(coarseCorrection.cellCountX() == fine.cellCountX() / 2 &&
           coarseCorrection.cellCountY() == fine.cellCountY() / 2 &&
           coarseCorrection.cellCountZ() == fine.cellCountZ() / 2);

    const auto nx = static_cast<std::ptrdiff_t>(coarseCorrection.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(coarseCorrection.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(coarseCorrection.cellCountZ());

    const std::size_t fineNx = fine.cellCountX();
    const std::size_t fineNy = fine.cellCountY();
    const std::size_t fineNz = fine.cellCountZ();

    if (fineNx * fineNy * fineNz >= kMultigridGpuDispatchThreshold) {
        std::vector<float> fineData(fineNx * fineNy * fineNz);
        for (std::size_t i = 0; i < fineNx; ++i) {
            for (std::size_t j = 0; j < fineNy; ++j) {
                for (std::size_t k = 0; k < fineNz; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    fineData[(i * fineNy + j) * fineNz + k] =
                        static_cast<float>(fine(pi, pj, pk));
                }
            }
        }

        const auto cnx = static_cast<std::size_t>(nx);
        const auto cny = static_cast<std::size_t>(ny);
        const auto cnz = static_cast<std::size_t>(nz);
        std::vector<float> coarseData(cnx * cny * cnz);
        for (std::size_t i = 0; i < cnx; ++i) {
            for (std::size_t j = 0; j < cny; ++j) {
                for (std::size_t k = 0; k < cnz; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    coarseData[(i * cny + j) * cnz + k] =
                        static_cast<float>(coarseCorrection(pi, pj, pk));
                }
            }
        }

        std::vector<float> nextFineData(fineNx * fineNy * fineNz);
        defaultBackend().multigridProlongateAndAdd3D(fineData, coarseData, fineNx, fineNy,
                                                     fineNz, nextFineData);

        for (std::size_t i = 0; i < fineNx; ++i) {
            for (std::size_t j = 0; j < fineNy; ++j) {
                for (std::size_t k = 0; k < fineNz; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    fine(pi, pj, pk) =
                        static_cast<double>(nextFineData[(i * fineNy + j) * fineNz + k]);
                }
            }
        }
        return;
    }

    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                const double correction = coarseCorrection(i, j, k);
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        for (int dk = 0; dk < 2; ++dk) {
                            fine(2 * i + di, 2 * j + dj, 2 * k + dk) += correction;
                        }
                    }
                }
            }
        }
    }
}

template <class Operator, class Relax, class Boundary>
void smooth(Grid3D<double>& u, const Grid3D<double>& target, double spacing, int sweeps,
            Operator applyOperator, Relax relaxPoint, Boundary applyBoundary) {
    (void)applyOperator;
    const auto nx = static_cast<std::ptrdiff_t>(u.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(u.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(u.cellCountZ());
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        applyBoundary(u);
        for (std::ptrdiff_t i = 0; i < nx; ++i) {
            for (std::ptrdiff_t j = 0; j < ny; ++j) {
                for (std::ptrdiff_t k = 0; k < nz; ++k) {
                    relaxPoint(u, i, j, k, spacing, target(i, j, k));
                }
            }
        }
    }
    applyBoundary(u);
}

template <class Operator, class Relax, class Boundary>
void vCycle(Grid3D<double>& u, const Grid3D<double>& target, double spacing,
            Operator applyOperator, Relax relaxPoint, Boundary applyBoundary,
            const MultigridSettings& settings) {
    const std::size_t nx = u.cellCountX();
    const std::size_t ny = u.cellCountY();
    const std::size_t nz = u.cellCountZ();
    const bool canCoarsen = nx % 2 == 0 && ny % 2 == 0 && nz % 2 == 0 &&
                            nx / 2 >= settings.coarsestCellCount &&
                            ny / 2 >= settings.coarsestCellCount &&
                            nz / 2 >= settings.coarsestCellCount;

    if (!canCoarsen) {
        // The coarsest level: many relaxation sweeps stand in for an exact
        // solve, the standard cheap substitute when the problem is small
        // enough that this converges well past what the finer levels need.
        smooth(u, target, spacing, settings.preSmoothingSweeps * 20, applyOperator,
               relaxPoint, applyBoundary);
        return;
    }

    smooth(u, target, spacing, settings.preSmoothingSweeps, applyOperator, relaxPoint,
           applyBoundary);

    const auto nxi = static_cast<std::ptrdiff_t>(nx);
    const auto nyi = static_cast<std::ptrdiff_t>(ny);
    const auto nzi = static_cast<std::ptrdiff_t>(nz);

    // Fine-grid defect: target - L(u).
    Grid3D<double> defect(nx, ny, nz, spacing, u.ghostCells());
    for (std::ptrdiff_t i = 0; i < nxi; ++i) {
        for (std::ptrdiff_t j = 0; j < nyi; ++j) {
            for (std::ptrdiff_t k = 0; k < nzi; ++k) {
                defect(i, j, k) = target(i, j, k) - applyOperator(u, i, j, k, spacing);
            }
        }
    }

    Grid3D<double> uCoarse = restrictGrid(u);
    const Grid3D<double> uCoarseBeforeCorrection = uCoarse;
    const Grid3D<double> defectCoarse = restrictGrid(defect);
    const double coarseSpacing = spacing * 2.0;
    applyBoundary(uCoarse);

    const auto cnx = static_cast<std::ptrdiff_t>(uCoarse.cellCountX());
    const auto cny = static_cast<std::ptrdiff_t>(uCoarse.cellCountY());
    const auto cnz = static_cast<std::ptrdiff_t>(uCoarse.cellCountZ());

    // FAS: the coarse level solves L(uCoarse) = L(restrict(u)) + defectCoarse,
    // not L(uCoarse) = 0 -- the correction that makes multigrid correct for
    // a genuinely nonlinear L rather than only a linear one.
    Grid3D<double> targetCoarse(uCoarse.cellCountX(), uCoarse.cellCountY(),
                                uCoarse.cellCountZ(), coarseSpacing,
                                uCoarse.ghostCells());
    for (std::ptrdiff_t i = 0; i < cnx; ++i) {
        for (std::ptrdiff_t j = 0; j < cny; ++j) {
            for (std::ptrdiff_t k = 0; k < cnz; ++k) {
                targetCoarse(i, j, k) = applyOperator(uCoarse, i, j, k, coarseSpacing) +
                                        defectCoarse(i, j, k);
            }
        }
    }

    vCycle(uCoarse, targetCoarse, coarseSpacing, applyOperator, relaxPoint, applyBoundary,
           settings);

    Grid3D<double> correction(uCoarse.cellCountX(), uCoarse.cellCountY(),
                              uCoarse.cellCountZ(), coarseSpacing, uCoarse.ghostCells());
    for (std::ptrdiff_t i = 0; i < cnx; ++i) {
        for (std::ptrdiff_t j = 0; j < cny; ++j) {
            for (std::ptrdiff_t k = 0; k < cnz; ++k) {
                correction(i, j, k) = uCoarse(i, j, k) - uCoarseBeforeCorrection(i, j, k);
            }
        }
    }

    prolongateAndAdd(u, correction);
    applyBoundary(u);

    smooth(u, target, spacing, settings.postSmoothingSweeps, applyOperator, relaxPoint,
           applyBoundary);
}

}  // namespace detail

/// Solves `applyOperator(u, ...) = 0` in place, starting from whatever `u`
/// already holds, returning how many V-cycles were used and the final
/// residual. `u`'s own dimensions and ghost-cell count are used at every
/// coarser level too (only the finest level's are the caller's concern).
template <class Operator, class Relax, class Boundary>
[[nodiscard]] MultigridResult
solveFAS(Grid3D<double>& u, double spacing, Operator applyOperator, Relax relaxPoint,
         Boundary applyBoundary, const MultigridSettings& settings = {}) {
    const Grid3D<double> zeroTarget(u.cellCountX(), u.cellCountY(), u.cellCountZ(),
                                    spacing, u.ghostCells());

    const auto nx = static_cast<std::ptrdiff_t>(u.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(u.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(u.cellCountZ());

    MultigridResult result;
    applyBoundary(u);
    for (int cycle = 0; cycle < settings.maxVCycles; ++cycle) {
        detail::vCycle(u, zeroTarget, spacing, applyOperator, relaxPoint, applyBoundary,
                       settings);

        double maxResidual = 0.0;
        for (std::ptrdiff_t i = 0; i < nx; ++i) {
            for (std::ptrdiff_t j = 0; j < ny; ++j) {
                for (std::ptrdiff_t k = 0; k < nz; ++k) {
                    maxResidual = std::max(maxResidual,
                                           std::abs(applyOperator(u, i, j, k, spacing)));
                }
            }
        }

        result.vCyclesUsed = cycle + 1;
        result.finalResidual = maxResidual;
        if (maxResidual < settings.residualTolerance) {
            result.converged = true;
            break;
        }
    }
    return result;
}

}  // namespace ysq
