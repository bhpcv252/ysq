#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ysq {

/// The time-independent Schrödinger equation in three spatial dimensions,
/// `-hbar^2/(2m) (d^2/dx^2 + d^2/dy^2 + d^2/dz^2) psi + V psi = E psi`, the
/// direct generalization of `Schrodinger.hpp`'s tridiagonal eigenvalue
/// problem to the 7-point 3D Laplacian stencil.
///
/// `hbar` and `mass` are always explicit, the same standing
/// `Schrodinger.hpp` has.
///
/// **Scope: dense eigensolver, so only modest grids.** The Hamiltonian is
/// `N x N` for `N = nx ny nz`, and `Math/Eigen.hpp` has no sparse/iterative
/// eigensolver (Lanczos, say) -- building one is separate scope well
/// beyond a 1D-to-3D extension. `jacobiEigenSymmetric` is `O(N^3)`, so this
/// is correct and general but only practical for small grids (validated
/// up to `8x8x8 = 512`), not a production-scale simulation.

/// Every eigenstate `solveTimeIndependentSchrodinger3D` finds, ascending by
/// energy. `wavefunctions[n][idx(i, j, k)]` is the n-th eigenstate's
/// amplitude at grid point `(i, j, k)`, `idx(i, j, k) = (i * ny + j) * nz +
/// k` (row-major, `x` slowest, `z` fastest, matching `Math/FFT.hpp`'s 3D
/// convention), normalized so
/// `sum_idx wavefunctions[n][idx]^2 * spacing^3 == 1`.
struct QuantumEigenstates3D {
    std::vector<double> energies;
    std::vector<std::vector<double>> wavefunctions;
};

/// Discretizes the Hamiltonian on an `nx * ny * nz` grid via the standard
/// 7-point stencil for the 3D Laplacian:
///
/// ```
/// H[idx][idx]           = 3 hbar^2 / (m spacing^2) + V(x_i, y_j, z_k)
/// H[idx][idx of a neighbor] = -hbar^2 / (2 m spacing^2)
/// ```
///
/// (Dirichlet boundary, the same "particle confined to this domain"
/// condition `Schrodinger.hpp` uses: a neighbor past the domain edge is
/// simply not a term), then diagonalizes with `Math/Eigen.hpp`'s
/// `jacobiEigenSymmetric`, the exact same solver `Schrodinger.hpp` uses,
/// since this discretizes to the same kind of matrix, only bigger.
/// `potential` is flat, row-major, size `nx * ny * nz`.
[[nodiscard]] QuantumEigenstates3D
solveTimeIndependentSchrodinger3D(std::span<const double> potential, std::size_t nx,
                                  std::size_t ny, std::size_t nz, double spacing,
                                  double mass, double hbar);

}  // namespace ysq
