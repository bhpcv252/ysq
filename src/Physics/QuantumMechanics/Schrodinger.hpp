#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ysq {

/// The time-independent Schrödinger equation, `-hbar^2/(2m) d^2(psi)/dx^2 +
/// V(x) psi = E psi`, as a symmetric matrix eigenvalue problem.
///
/// `hbar` and `mass` are always explicit, never defaulted: real SI quantum
/// mechanics has extremely small numbers (`hbar ~ 1e-34`), and
/// natural/atomic units (`hbar = 1`) are just as valid a choice a caller
/// might make instead, so nothing here assumes one over the other.

/// Every eigenstate `solveTimeIndependentSchrodinger` finds, ascending by
/// energy. `wavefunctions[n][i]` is the n-th eigenstate's amplitude at grid
/// point `i`, normalized so `sum_i wavefunctions[n][i]^2 * spacing == 1`.
struct QuantumEigenstates {
    std::vector<double> energies;
    std::vector<std::vector<double>> wavefunctions;
};

/// Discretizes the Hamiltonian on a grid of `potential.size()` points via
/// the standard 3-point central-difference stencil for the second
/// derivative:
///
/// ```
/// H[i][i]   = hbar^2 / (m spacing^2) + V(x_i)
/// H[i][i-1] = H[i][i+1] = -hbar^2 / (2 m spacing^2)
/// ```
///
/// (Dirichlet boundary: `psi = 0` just outside the grid, the standard
/// "particle confined to this domain" condition), then diagonalizes the
/// resulting real symmetric matrix with `Math/Eigen.hpp`'s
/// `jacobiEigenSymmetric` -- the Hamiltonian discretizes to exactly the
/// kind of matrix that solver exists for.
[[nodiscard]] QuantumEigenstates
solveTimeIndependentSchrodinger(std::span<const double> potential, double spacing,
                                double mass, double hbar);

}  // namespace ysq
