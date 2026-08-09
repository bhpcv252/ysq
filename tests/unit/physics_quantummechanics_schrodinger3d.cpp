#include <Physics/QuantumMechanics/Schrodinger3D.hpp>

#include <Physics/QuantumMechanics/Schrodinger.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using ysq::QuantumEigenstates;
using ysq::QuantumEigenstates3D;

}  // namespace

TEST(PhysicsQuantumMechanicsSchrodinger3D,
     InfiniteWellEigenvaluesAreSumsOfTheOneDimensionalOnes) {
    // The discretized 3D infinite well is exactly separable (its
    // Hamiltonian is the Kronecker sum of three copies of the identical 1D
    // discretized Hamiltonian, since the potential is zero on every axis),
    // so its eigenvalues are exactly every sum of three of the 1D
    // problem's own eigenvalues -- an exact cross-check against
    // Schrodinger.hpp's already-validated 1D solver, not a resolution-
    // dependent comparison to the continuum closed form.
    constexpr std::size_t n = 6;
    constexpr double length = 1.0;
    constexpr double mass = 1.0;
    constexpr double hbar = 1.0;
    const double spacing = length / static_cast<double>(n + 1);

    const std::vector<double> potential1D(n, 0.0);
    const QuantumEigenstates states1D =
        ysq::solveTimeIndependentSchrodinger(potential1D, spacing, mass, hbar);

    const std::vector<double> potential3D(n * n * n, 0.0);
    const QuantumEigenstates3D states3D =
        ysq::solveTimeIndependentSchrodinger3D(potential3D, n, n, n, spacing, mass, hbar);

    std::vector<double> expected;
    expected.reserve(n * n * n);
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = 0; b < n; ++b) {
            for (std::size_t c = 0; c < n; ++c) {
                expected.push_back(states1D.energies[a] + states1D.energies[b] +
                                   states1D.energies[c]);
            }
        }
    }
    std::sort(expected.begin(), expected.end());

    ASSERT_EQ(states3D.energies.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(states3D.energies[i], expected[i],
                    std::abs(expected[i]) * 1e-6 + 1e-9)
            << "i = " << i;
    }
}

TEST(PhysicsQuantumMechanicsSchrodinger3D,
     HarmonicOscillatorEigenvaluesAreSumsOfTheOneDimensionalOnes) {
    // Separable for the same reason as the infinite well above, plus the
    // isotropic 3D harmonic potential itself already being a sum of three
    // identical 1D potentials: V(x,y,z) = Vx(x) + Vy(y) + Vz(z).
    constexpr std::size_t n = 6;
    constexpr double mass = 1.0;
    constexpr double hbar = 1.0;
    constexpr double omega = 1.0;
    constexpr double domainHalfWidth = 4.0;
    const double spacing = (2.0 * domainHalfWidth) / static_cast<double>(n + 1);

    std::vector<double> potential1D(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -domainHalfWidth + static_cast<double>(i + 1) * spacing;
        potential1D[i] = 0.5 * mass * omega * omega * x * x;
    }

    const QuantumEigenstates states1D =
        ysq::solveTimeIndependentSchrodinger(potential1D, spacing, mass, hbar);

    std::vector<double> potential3D(n * n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                potential3D[(i * n + j) * n + k] =
                    potential1D[i] + potential1D[j] + potential1D[k];
            }
        }
    }

    const QuantumEigenstates3D states3D =
        ysq::solveTimeIndependentSchrodinger3D(potential3D, n, n, n, spacing, mass, hbar);

    std::vector<double> expected;
    expected.reserve(n * n * n);
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = 0; b < n; ++b) {
            for (std::size_t c = 0; c < n; ++c) {
                expected.push_back(states1D.energies[a] + states1D.energies[b] +
                                   states1D.energies[c]);
            }
        }
    }
    std::sort(expected.begin(), expected.end());

    ASSERT_EQ(states3D.energies.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(states3D.energies[i], expected[i],
                    std::abs(expected[i]) * 1e-6 + 1e-9)
            << "i = " << i;
    }

    // And the ground state itself should be roughly near the closed-form
    // 1.5 hbar omega -- only a loose sanity bound, since n = 6 is coarse
    // enough on its own (chosen for the dense eigensolver's O(N^3) cost)
    // that discretization error dominates here; the sums-of-1D-eigenvalues
    // check above is the real, tight validation.
    EXPECT_NEAR(states3D.energies[0], 1.5 * hbar * omega, 0.2);
}

TEST(PhysicsQuantumMechanicsSchrodinger3D, EigenstatesAreOrthonormal) {
    constexpr std::size_t n = 4;
    const double spacing = 0.2;
    const std::vector<double> potential(n * n * n, 0.0);

    const QuantumEigenstates3D states =
        ysq::solveTimeIndependentSchrodinger3D(potential, n, n, n, spacing, 1.0, 1.0);
    const double cellVolume = spacing * spacing * spacing;

    for (std::size_t a = 0; a < 5; ++a) {
        for (std::size_t b = 0; b < 5; ++b) {
            double overlap = 0.0;
            for (std::size_t p = 0; p < n * n * n; ++p) {
                overlap += states.wavefunctions[a][p] * states.wavefunctions[b][p];
            }
            overlap *= cellVolume;
            EXPECT_NEAR(overlap, (a == b) ? 1.0 : 0.0, 1e-8) << "a=" << a << " b=" << b;
        }
    }
}

TEST(PhysicsQuantumMechanicsSchrodinger3D, EnergiesAreReturnedInAscendingOrder) {
    constexpr std::size_t n = 4;
    const std::vector<double> potential(n * n * n, 0.0);
    const QuantumEigenstates3D states =
        ysq::solveTimeIndependentSchrodinger3D(potential, n, n, n, 0.1, 1.0, 1.0);

    for (std::size_t i = 1; i < states.energies.size(); ++i) {
        EXPECT_LE(states.energies[i - 1], states.energies[i]);
    }
}
