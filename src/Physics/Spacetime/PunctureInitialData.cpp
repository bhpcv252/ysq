#include <Physics/Spacetime/PunctureInitialData.hpp>

#include <Math/Tensor.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace ysq {

namespace {

/// The physical coordinate of a grid cell's own center: cell-centered
/// (`i + 0.5`, never an integer), so a puncture placed at any coordinate
/// that is itself a multiple of `spacing` -- the coordinate origin, in
/// particular -- never coincides exactly with a grid point and the 1/r
/// singularity there is never sampled directly. `halfX/Y/Z` shift the
/// domain's interior to be roughly centered on the origin; they need not be
/// exact for this non-coincidence property to hold (see the derivation in
/// this function's caller).
[[nodiscard]] Vec3 cellCenter(std::ptrdiff_t i, std::ptrdiff_t j, std::ptrdiff_t k,
                              double spacing, double halfX, double halfY, double halfZ) {
    return Vec3{(static_cast<double>(i) + 0.5) * spacing - halfX,
               (static_cast<double>(j) + 0.5) * spacing - halfY,
               (static_cast<double>(k) + 0.5) * spacing - halfZ};
}

/// The flat-space Bowen-York extrinsic curvature (Bowen & York, Phys. Rev. D
/// 21, 2047 (1980)), summed over every puncture: the closed-form solution
/// that satisfies the (flat-background) momentum constraint identically for
/// any mass/momentum/spin, which is what lets the Hamiltonian constraint be
/// the only equation this module solves numerically.
///
///   AbarIJ = (3 / 2 r^2) [ P^i n^j + P^j n^i - (delta^ij - n^i n^j) P.n ]
///          + (3 / r^3) [ (S x n)^i n^j + (S x n)^j n^i ]
[[nodiscard]] Tensor<double, 2, 3> bowenYorkAt(const std::vector<PunctureSpec>& punctures,
                                              const Vec3& point) {
    Tensor<double, 2, 3> total{};
    for (const PunctureSpec& puncture : punctures) {
        const Vec3 delta = point - puncture.position;
        const double r = length(delta);
        // Cell-centered placement (see cellCenter) keeps this from firing in
        // practice; guarding it anyway is cheap and avoids a division by
        // zero if a caller ever places a puncture exactly on a grid point.
        if (r < 1.0e-12) {
            continue;
        }
        const Vec3 n = delta / r;
        const double pDotN = dot(puncture.momentum, n);
        const Vec3 spinCrossN = cross(puncture.spin, n);

        const std::array<double, 3> p{puncture.momentum.x, puncture.momentum.y,
                                      puncture.momentum.z};
        const std::array<double, 3> nArr{n.x, n.y, n.z};
        const std::array<double, 3> sxn{spinCrossN.x, spinCrossN.y, spinCrossN.z};

        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const double delta_ab = (a == b) ? 1.0 : 0.0;
                const double linear =
                    (3.0 / (2.0 * r * r)) *
                    (p[static_cast<std::size_t>(a)] * nArr[static_cast<std::size_t>(b)] +
                     p[static_cast<std::size_t>(b)] * nArr[static_cast<std::size_t>(a)] -
                     (delta_ab - nArr[static_cast<std::size_t>(a)] *
                                     nArr[static_cast<std::size_t>(b)]) *
                         pDotN);
                const double spinTerm =
                    (3.0 / (r * r * r)) *
                    (sxn[static_cast<std::size_t>(a)] * nArr[static_cast<std::size_t>(b)] +
                     sxn[static_cast<std::size_t>(b)] * nArr[static_cast<std::size_t>(a)]);
                total(a, b) += linear + spinTerm;
            }
        }
    }
    return total;
}

/// The Brill-Lindquist superposition, the singular part of the conformal
/// factor factored out analytically so the numerically solved correction
/// `u` is smooth (including at every puncture): psi_BL = 1 + sum m_i / (2 r_i).
[[nodiscard]] double brillLindquistPsi(const std::vector<PunctureSpec>& punctures,
                                      const Vec3& point) {
    double psi = 1.0;
    for (const PunctureSpec& puncture : punctures) {
        const double r = length(point - puncture.position);
        psi += puncture.mass / (2.0 * r);
    }
    return psi;
}

}  // namespace

PunctureInitialDataResult solvePunctureInitialData(const std::vector<PunctureSpec>& punctures,
                                                   std::size_t cellCountX,
                                                   std::size_t cellCountY,
                                                   std::size_t cellCountZ, double spacing,
                                                   std::size_t ghostCells,
                                                   const MultigridSettings& settings) {
    // Deliberately integer division, truncated before the cast: an odd cell
    // count centers the domain within half a cell of the origin rather than
    // failing to compile, and cellCenter's own +0.5 offset is what actually
    // guarantees no cell center lands exactly on the origin regardless.
    // Fixed once here and captured by every callable below: half-extent is
    // a physical constant describing the domain, the same at every
    // multigrid level regardless of that level's own (coarser) cell count,
    // since coarsening halves cell count and doubles spacing together.
    const std::size_t halfCellsX = cellCountX / 2;
    const std::size_t halfCellsY = cellCountY / 2;
    const std::size_t halfCellsZ = cellCountZ / 2;
    const double halfX = static_cast<double>(halfCellsX) * spacing;
    const double halfY = static_cast<double>(halfCellsY) * spacing;
    const double halfZ = static_cast<double>(halfCellsZ) * spacing;

    Grid3D<double> u(cellCountX, cellCountY, cellCountZ, spacing, ghostCells);

    const auto nx = static_cast<std::ptrdiff_t>(cellCountX);
    const auto ny = static_cast<std::ptrdiff_t>(cellCountY);
    const auto nz = static_cast<std::ptrdiff_t>(cellCountZ);

    // The Hamiltonian constraint for conformally flat, momentarily-static
    // (maximal slicing, K = 0) puncture data (Brandt & Brügmann 1997),
    // written as L(u) = 0 for Math::solveFAS's convention:
    //
    //   L(u) := flatLaplacian(u) + (1/8) AbarIJ AbarIJ (psi_BL + u)^{-7} = 0
    //
    // `punctureTerm` below is that second term alone, evaluated with
    // whatever `u` value is passed in (the caller decides whether that's
    // the current, not-yet-updated value -- a lagged/Picard nonlinearity,
    // the same treatment the original relaxation solver used).
    const auto punctureTerm = [&punctures](const Vec3& point, double uValue) {
        const Tensor<double, 2, 3> aBar = bowenYorkAt(punctures, point);
        double aBarSquared = 0.0;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                aBarSquared += aBar(a, b) * aBar(a, b);
            }
        }
        const double psiTotal = brillLindquistPsi(punctures, point) + uValue;
        return (1.0 / 8.0) * aBarSquared * std::pow(psiTotal, -7.0);
    };

    const auto applyOperator = [&](const Grid3D<double>& field, std::ptrdiff_t i,
                                  std::ptrdiff_t j, std::ptrdiff_t k, double h) {
        const Vec3 point = cellCenter(i, j, k, h, halfX, halfY, halfZ);
        const double laplacian = (field(i + 1, j, k) + field(i - 1, j, k) +
                                 field(i, j + 1, k) + field(i, j - 1, k) +
                                 field(i, j, k + 1) + field(i, j, k - 1) -
                                 6.0 * field(i, j, k)) /
                                (h * h);
        return laplacian + punctureTerm(point, field(i, j, k));
    };

    const auto relaxPoint = [&](Grid3D<double>& field, std::ptrdiff_t i, std::ptrdiff_t j,
                               std::ptrdiff_t k, double h, double target) {
        const Vec3 point = cellCenter(i, j, k, h, halfX, halfY, halfZ);
        const double term = punctureTerm(point, field(i, j, k));
        const double neighborSum = field(i + 1, j, k) + field(i - 1, j, k) +
                                  field(i, j + 1, k) + field(i, j - 1, k) +
                                  field(i, j, k + 1) + field(i, j, k - 1);
        field(i, j, k) = (neighborSum - h * h * (target - term)) / 6.0;
    };

    // u = 0 held fixed at the domain's outer boundary: a simple Dirichlet
    // condition, adequate for a domain large enough that u itself has
    // decayed there; a true asymptotic falloff condition is a refinement
    // this stage does not need (for a single, zero-momentum puncture, u = 0
    // is the exact answer everywhere, boundary included).
    const auto applyBoundary = [](Grid3D<double>& field) {
        const auto fnx = static_cast<std::ptrdiff_t>(field.cellCountX());
        const auto fny = static_cast<std::ptrdiff_t>(field.cellCountY());
        const auto fnz = static_cast<std::ptrdiff_t>(field.cellCountZ());
        const auto ghost = static_cast<std::ptrdiff_t>(field.ghostCells());
        for (std::ptrdiff_t g = 1; g <= ghost; ++g) {
            for (std::ptrdiff_t j = -ghost; j < fny + ghost; ++j) {
                for (std::ptrdiff_t k = -ghost; k < fnz + ghost; ++k) {
                    field(-g, j, k) = 0.0;
                    field(fnx - 1 + g, j, k) = 0.0;
                }
            }
        }
        for (std::ptrdiff_t g = 1; g <= ghost; ++g) {
            for (std::ptrdiff_t i = -ghost; i < fnx + ghost; ++i) {
                for (std::ptrdiff_t k = -ghost; k < fnz + ghost; ++k) {
                    field(i, -g, k) = 0.0;
                    field(i, fny - 1 + g, k) = 0.0;
                }
            }
        }
        for (std::ptrdiff_t g = 1; g <= ghost; ++g) {
            for (std::ptrdiff_t i = -ghost; i < fnx + ghost; ++i) {
                for (std::ptrdiff_t j = -ghost; j < fny + ghost; ++j) {
                    field(i, j, -g) = 0.0;
                    field(i, j, fnz - 1 + g) = 0.0;
                }
            }
        }
    };

    const MultigridResult multigridResult =
        solveFAS(u, spacing, applyOperator, relaxPoint, applyBoundary, settings);
    const int iterationsUsed = multigridResult.vCyclesUsed;
    const double finalResidual = multigridResult.finalResidual;
    const bool converged = multigridResult.converged;

    AdmData adm(cellCountX, cellCountY, cellCountZ, spacing, ghostCells);
    const auto ig = static_cast<std::ptrdiff_t>(ghostCells);
    for (std::ptrdiff_t i = -ig; i < nx + ig; ++i) {
        for (std::ptrdiff_t j = -ig; j < ny + ig; ++j) {
            for (std::ptrdiff_t k = -ig; k < nz + ig; ++k) {
                const Vec3 point = cellCenter(i, j, k, spacing, halfX, halfY, halfZ);
                const double uValue = (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                                         ? u(i, j, k)
                                         : 0.0;
                const double psi = brillLindquistPsi(punctures, point) + uValue;
                const double conformalFactor4 = psi * psi * psi * psi;

                Tensor<double, 2, 3> gamma{};
                Tensor<double, 2, 3> kExtrinsic{};
                const Tensor<double, 2, 3> aBar = bowenYorkAt(punctures, point);
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        gamma(a, b) = (a == b) ? conformalFactor4 : 0.0;
                        kExtrinsic(a, b) = aBar(a, b) / (psi * psi);
                    }
                }

                adm.spatialMetric.set(i, j, k, gamma);
                adm.extrinsicCurvature.set(i, j, k, kExtrinsic);
                adm.lapse(i, j, k) = 1.0;
                adm.shift.x(i, j, k) = 0.0;
                adm.shift.y(i, j, k) = 0.0;
                adm.shift.z(i, j, k) = 0.0;
            }
        }
    }

    return PunctureInitialDataResult{std::move(adm), iterationsUsed, finalResidual, converged};
}

double newtonianCircularMomentum(double mass1, double mass2, double separation) {
    const double totalMass = mass1 + mass2;
    const double reducedMass = mass1 * mass2 / totalMass;
    return reducedMass * std::sqrt(totalMass / separation);
}

}  // namespace ysq
