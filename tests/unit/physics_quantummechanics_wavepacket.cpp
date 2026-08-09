#include <Math/Complex.hpp>
#include <Math/Scalar.hpp>
#include <Physics/QuantumMechanics/Schrodinger.hpp>
#include <Physics/QuantumMechanics/WavePacket.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using ysq::Complex;
using ysq::TimeDependentWavefunction1D;

constexpr std::size_t kGridSize = 256;  // a power of two, Math/FFT.hpp's precondition
constexpr double kMass = 1.0;
constexpr double kHbar = 1.0;
constexpr double kOmega = 1.0;
constexpr double kDomainHalfWidth = 10.0;

double spacingFor(std::size_t n) {
    return (2.0 * kDomainHalfWidth) / static_cast<double>(n + 1);
}

std::vector<double> harmonicOscillatorPotential(std::size_t n, double spacing) {
    std::vector<double> potential(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -kDomainHalfWidth + static_cast<double>(i + 1) * spacing;
        potential[i] = 0.5 * kMass * kOmega * kOmega * x * x;
    }
    return potential;
}

std::vector<Complex<double>> gaussianWavePacket(std::size_t n, double spacing,
                                                double center, double width) {
    std::vector<Complex<double>> psi(n);
    double normSquared = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -kDomainHalfWidth + static_cast<double>(i + 1) * spacing;
        const double envelope =
            std::exp(-((x - center) * (x - center)) / (2.0 * width * width));
        psi[i] = Complex<double>{envelope, 0.0};
        normSquared += envelope * envelope;
    }
    const double norm = std::sqrt(normSquared * spacing);
    for (Complex<double>& amplitude : psi) {
        amplitude *= (1.0 / norm);
    }
    return psi;
}

}  // namespace

TEST(PhysicsQuantumMechanicsWavePacket, TotalProbabilityStaysNormalizedOverManySteps) {
    const double spacing = spacingFor(kGridSize);
    const std::vector<double> potential = harmonicOscillatorPotential(kGridSize, spacing);
    TimeDependentWavefunction1D psi(gaussianWavePacket(kGridSize, spacing, 0.0, 1.0),
                                    potential, spacing, kMass, kHbar);

    ASSERT_NEAR(psi.totalProbability(), 1.0, 1e-9);

    for (int step = 0; step < 500; ++step) {
        psi.step(0.01);
    }

    EXPECT_NEAR(psi.totalProbability(), 1.0, 1e-6);
}

TEST(PhysicsQuantumMechanicsWavePacket,
     EnergyIsConservedOverManyStepsForATimeIndependentPotential) {
    const double spacing = spacingFor(kGridSize);
    const std::vector<double> potential = harmonicOscillatorPotential(kGridSize, spacing);
    TimeDependentWavefunction1D psi(gaussianWavePacket(kGridSize, spacing, 2.0, 1.0),
                                    potential, spacing, kMass, kHbar);

    const double initialEnergy = psi.expectationEnergy();
    ASSERT_GT(initialEnergy, 0.0);

    for (int step = 0; step < 500; ++step) {
        psi.step(0.01);
    }

    // expectationEnergy() measures with a 3-point finite-difference
    // kinetic operator, while step() propagates with the FFT's spectral
    // one -- two different discretizations of the same continuous
    // operator, exact for each other's own eigenstates only in the
    // continuum limit. A small, non-accumulating offset between the two
    // readings is the expected signature of that mismatch, not drift; 1%
    // comfortably covers it while still catching genuine energy drift,
    // which would keep growing over further steps rather than saturate.
    EXPECT_NEAR(psi.expectationEnergy(), initialEnergy, std::abs(initialEnergy) * 1e-2);
}

TEST(PhysicsQuantumMechanicsWavePacket, AnEigenstateStaysStationaryUnderEvolution) {
    // A stationary state (an eigenstate of a time-independent Hamiltonian)
    // only picks up an overall phase exp(-i E t / hbar) under evolution --
    // invisible in the probability density -- which holds only if
    // Schrodinger.hpp's eigensolver and this class's propagator agree on
    // the same physics.
    const double spacing = spacingFor(kGridSize);
    const std::vector<double> potential = harmonicOscillatorPotential(kGridSize, spacing);

    const ysq::QuantumEigenstates states =
        ysq::solveTimeIndependentSchrodinger(potential, spacing, kMass, kHbar);

    std::vector<Complex<double>> groundState(kGridSize);
    for (std::size_t i = 0; i < kGridSize; ++i) {
        groundState[i] = Complex<double>{states.wavefunctions[0][i], 0.0};
    }

    TimeDependentWavefunction1D psi(groundState, potential, spacing, kMass, kHbar);

    std::vector<double> densityBefore(kGridSize);
    for (std::size_t i = 0; i < kGridSize; ++i) {
        densityBefore[i] = psi.probabilityDensity(i);
    }

    for (int step = 0; step < 200; ++step) {
        psi.step(0.005);
    }

    // The ground state above is exact for Schrodinger.hpp's 3-point
    // finite-difference Hamiltonian, not for the FFT's spectral kinetic
    // operator step() actually propagates with -- two different
    // discretizations of the same continuous operator, so it is only
    // approximately stationary here, with the residual reflecting that
    // discretization mismatch (shrinking as the grid refines) rather than
    // a numerical bug. A fixed floor keeps the check meaningful at the
    // wavefunction's small-amplitude tails, where a purely relative
    // comparison would be dominated by noise rather than signal.
    const double peakDensity =
        *std::max_element(densityBefore.begin(), densityBefore.end());
    for (std::size_t i = 0; i < kGridSize; ++i) {
        EXPECT_NEAR(psi.probabilityDensity(i), densityBefore[i], peakDensity * 0.01)
            << "grid point " << i;
    }
}

TEST(PhysicsQuantumMechanicsWavePacket,
     ProbabilityDensityMatchesTheAmplitudesSquaredMagnitude) {
    const double spacing = spacingFor(kGridSize);
    const std::vector<double> potential = harmonicOscillatorPotential(kGridSize, spacing);
    const TimeDependentWavefunction1D psi(
        gaussianWavePacket(kGridSize, spacing, 0.0, 1.0), potential, spacing, kMass,
        kHbar);

    for (std::size_t i = 0; i < kGridSize; i += 17) {
        const Complex<double> amplitude = psi.amplitude(i);
        const double expected = amplitude.re * amplitude.re + amplitude.im * amplitude.im;
        EXPECT_NEAR(psi.probabilityDensity(i), expected, 1e-15);
    }
}
