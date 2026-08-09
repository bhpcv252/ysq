#include <Math/Scalar.hpp>
#include <Physics/QuantumMechanics/Schrodinger.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace {

using ysq::QuantumEigenstates;

}  // namespace

TEST(PhysicsQuantumMechanicsSchrodinger, InfiniteSquareWellMatchesTheClosedFormEnergies) {
    // A particle confined to (0, L) by V = 0 inside and the Dirichlet
    // boundary itself acting as infinite walls just outside the grid.
    constexpr std::size_t n = 400;
    constexpr double length = 1.0;
    constexpr double mass = 1.0;
    constexpr double hbar = 1.0;
    const double spacing = length / static_cast<double>(n + 1);

    const std::vector<double> potential(n, 0.0);
    const QuantumEigenstates states =
        ysq::solveTimeIndependentSchrodinger(potential, spacing, mass, hbar);

    for (int level = 1; level <= 4; ++level) {
        const double expected = (static_cast<double>(level * level) * ysq::kPi<double> *
                                 ysq::kPi<double> * hbar * hbar) /
                                (2.0 * mass * length * length);
        const double actual = states.energies[static_cast<std::size_t>(level - 1)];
        EXPECT_NEAR(actual, expected, expected * 0.01) << "level = " << level;
    }
}

TEST(PhysicsQuantumMechanicsSchrodinger, HarmonicOscillatorMatchesTheClosedFormEnergies) {
    constexpr std::size_t n = 500;
    constexpr double mass = 1.0;
    constexpr double hbar = 1.0;
    constexpr double omega = 1.0;
    constexpr double domainHalfWidth =
        8.0;  // characteristic length is sqrt(hbar/(m omega)) = 1
    const double spacing = (2.0 * domainHalfWidth) / static_cast<double>(n + 1);

    std::vector<double> potential(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -domainHalfWidth + static_cast<double>(i + 1) * spacing;
        potential[i] = 0.5 * mass * omega * omega * x * x;
    }

    const QuantumEigenstates states =
        ysq::solveTimeIndependentSchrodinger(potential, spacing, mass, hbar);

    for (int level = 0; level <= 3; ++level) {
        const double expected = (static_cast<double>(level) + 0.5) * hbar * omega;
        const double actual = states.energies[static_cast<std::size_t>(level)];
        EXPECT_NEAR(actual, expected, 0.02) << "level = " << level;
    }
}

TEST(PhysicsQuantumMechanicsSchrodinger, EigenstatesAreOrthonormal) {
    constexpr std::size_t n = 60;
    constexpr double spacing = 0.05;
    const std::vector<double> potential(n, 0.0);

    const QuantumEigenstates states =
        ysq::solveTimeIndependentSchrodinger(potential, spacing, 1.0, 1.0);

    for (std::size_t a = 0; a < 5; ++a) {
        for (std::size_t b = 0; b < 5; ++b) {
            double overlap = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                overlap += states.wavefunctions[a][i] * states.wavefunctions[b][i];
            }
            overlap *= spacing;
            EXPECT_NEAR(overlap, (a == b) ? 1.0 : 0.0, 1e-9) << "a=" << a << " b=" << b;
        }
    }
}

TEST(PhysicsQuantumMechanicsSchrodinger, EnergiesAreReturnedInAscendingOrder) {
    constexpr std::size_t n = 100;
    const std::vector<double> potential(n, 0.0);
    const QuantumEigenstates states =
        ysq::solveTimeIndependentSchrodinger(potential, 0.02, 1.0, 1.0);

    for (std::size_t i = 1; i < states.energies.size(); ++i) {
        EXPECT_LE(states.energies[i - 1], states.energies[i]);
    }
}
