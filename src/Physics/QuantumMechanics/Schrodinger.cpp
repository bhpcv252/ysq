#include <Physics/QuantumMechanics/Schrodinger.hpp>

#include <Math/Eigen.hpp>
#include <Math/LinearSolve.hpp>

#include <cmath>

namespace ysq {

QuantumEigenstates solveTimeIndependentSchrodinger(std::span<const double> potential,
                                                   double spacing, double mass,
                                                   double hbar) {
    const std::size_t n = potential.size();
    const double offDiagonal = -(hbar * hbar) / (2.0 * mass * spacing * spacing);
    const double kineticDiagonal = (hbar * hbar) / (mass * spacing * spacing);

    MatrixN<double> hamiltonian(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        hamiltonian(i, i) = kineticDiagonal + potential[i];
        if (i + 1 < n) {
            hamiltonian(i, i + 1) = offDiagonal;
            hamiltonian(i + 1, i) = offDiagonal;
        }
    }

    const EigenDecomposition<double> decomposition = jacobiEigenSymmetric(hamiltonian);

    QuantumEigenstates result;
    result.energies.reserve(n);
    result.wavefunctions.reserve(n);

    for (std::size_t state = 0; state < n; ++state) {
        result.energies.push_back(decomposition.eigenvalues[state]);

        std::vector<double> wavefunction(n);
        double normSquared = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            wavefunction[i] = decomposition.eigenvectors(i, state);
            normSquared += wavefunction[i] * wavefunction[i];
        }
        const double norm = std::sqrt(normSquared * spacing);
        for (double& amplitude : wavefunction) {
            amplitude /= norm;
        }
        result.wavefunctions.push_back(std::move(wavefunction));
    }

    return result;
}

}  // namespace ysq
