#include <Physics/QuantumMechanics/Schrodinger3D.hpp>

#include <Math/Eigen.hpp>
#include <Math/LinearSolve.hpp>

#include <cmath>

namespace ysq {

QuantumEigenstates3D solveTimeIndependentSchrodinger3D(std::span<const double> potential,
                                                       std::size_t nx, std::size_t ny,
                                                       std::size_t nz, double spacing,
                                                       double mass, double hbar) {
    const std::size_t n = nx * ny * nz;
    const double offDiagonal = -(hbar * hbar) / (2.0 * mass * spacing * spacing);
    const double kineticDiagonal = 3.0 * (hbar * hbar) / (mass * spacing * spacing);

    const auto idx = [ny, nz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * ny + j) * nz + k;
    };

    MatrixN<double> hamiltonian(n, n, 0.0);
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t here = idx(i, j, k);
                hamiltonian(here, here) = kineticDiagonal + potential[here];

                if (i + 1 < nx) {
                    const std::size_t other = idx(i + 1, j, k);
                    hamiltonian(here, other) = offDiagonal;
                    hamiltonian(other, here) = offDiagonal;
                }
                if (j + 1 < ny) {
                    const std::size_t other = idx(i, j + 1, k);
                    hamiltonian(here, other) = offDiagonal;
                    hamiltonian(other, here) = offDiagonal;
                }
                if (k + 1 < nz) {
                    const std::size_t other = idx(i, j, k + 1);
                    hamiltonian(here, other) = offDiagonal;
                    hamiltonian(other, here) = offDiagonal;
                }
            }
        }
    }

    const EigenDecomposition<double> decomposition = jacobiEigenSymmetric(hamiltonian);

    QuantumEigenstates3D result;
    result.energies.reserve(n);
    result.wavefunctions.reserve(n);

    const double cellVolume = spacing * spacing * spacing;
    for (std::size_t state = 0; state < n; ++state) {
        result.energies.push_back(decomposition.eigenvalues[state]);

        std::vector<double> wavefunction(n);
        double normSquared = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            wavefunction[p] = decomposition.eigenvectors(p, state);
            normSquared += wavefunction[p] * wavefunction[p];
        }
        const double norm = std::sqrt(normSquared * cellVolume);
        for (double& amplitude : wavefunction) {
            amplitude /= norm;
        }
        result.wavefunctions.push_back(std::move(wavefunction));
    }

    return result;
}

}  // namespace ysq
