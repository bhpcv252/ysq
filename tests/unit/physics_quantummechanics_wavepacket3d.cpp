#include <Physics/QuantumMechanics/WavePacket3D.hpp>

#include <Math/Complex.hpp>
#include <Physics/QuantumMechanics/Schrodinger.hpp>
#include <Physics/QuantumMechanics/WavePacket.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using ysq::Complex;
using ysq::TimeDependentWavefunction3D;

constexpr double kMass = 1.0;
constexpr double kHbar = 1.0;
constexpr double kOmega = 1.0;

double gaussian1D(double x, double center, double width) {
    const double normalized = (x - center) / width;
    return std::exp(-0.5 * normalized * normalized);
}

}  // namespace

TEST(PhysicsQuantumMechanicsWavePacket3D, TotalProbabilityStaysNormalizedOverManySteps) {
    constexpr std::size_t n = 16;
    constexpr double domainHalfWidth = 6.0;
    const double spacing = (2.0 * domainHalfWidth) / static_cast<double>(n);

    std::vector<double> potential(n * n * n);
    std::vector<Complex<double>> psi(n * n * n);
    double normSquared = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -domainHalfWidth + static_cast<double>(i) * spacing;
        for (std::size_t j = 0; j < n; ++j) {
            const double y = -domainHalfWidth + static_cast<double>(j) * spacing;
            for (std::size_t k = 0; k < n; ++k) {
                const double z = -domainHalfWidth + static_cast<double>(k) * spacing;
                const std::size_t p = (i * n + j) * n + k;
                potential[p] = 0.5 * kMass * kOmega * kOmega * (x * x + y * y + z * z);
                const double envelope = gaussian1D(x, 0.0, 1.0) *
                                        gaussian1D(y, 0.0, 1.0) * gaussian1D(z, 0.0, 1.0);
                psi[p] = Complex<double>{envelope, 0.0};
                normSquared += envelope * envelope;
            }
        }
    }
    const double norm = std::sqrt(normSquared * spacing * spacing * spacing);
    for (Complex<double>& amplitude : psi) {
        amplitude *= (1.0 / norm);
    }

    TimeDependentWavefunction3D wavefunction(psi, potential, n, n, n, spacing, kMass,
                                             kHbar);
    ASSERT_NEAR(wavefunction.totalProbability(), 1.0, 1e-9);

    for (int step = 0; step < 100; ++step) {
        wavefunction.step(0.01);
    }

    EXPECT_NEAR(wavefunction.totalProbability(), 1.0, 1e-6);
}

TEST(PhysicsQuantumMechanicsWavePacket3D, EnergyIsConservedOverManySteps) {
    constexpr std::size_t n = 16;
    constexpr double domainHalfWidth = 6.0;
    const double spacing = (2.0 * domainHalfWidth) / static_cast<double>(n);

    std::vector<double> potential(n * n * n);
    std::vector<Complex<double>> psi(n * n * n);
    double normSquared = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -domainHalfWidth + static_cast<double>(i) * spacing;
        for (std::size_t j = 0; j < n; ++j) {
            const double y = -domainHalfWidth + static_cast<double>(j) * spacing;
            for (std::size_t k = 0; k < n; ++k) {
                const double z = -domainHalfWidth + static_cast<double>(k) * spacing;
                const std::size_t p = (i * n + j) * n + k;
                potential[p] = 0.5 * kMass * kOmega * kOmega * (x * x + y * y + z * z);
                // Off-center, so the packet has real kinetic + potential
                // energy rather than sitting at the potential's minimum.
                const double envelope = gaussian1D(x, 1.5, 1.0) *
                                        gaussian1D(y, 0.0, 1.0) * gaussian1D(z, 0.0, 1.0);
                psi[p] = Complex<double>{envelope, 0.0};
                normSquared += envelope * envelope;
            }
        }
    }
    const double norm = std::sqrt(normSquared * spacing * spacing * spacing);
    for (Complex<double>& amplitude : psi) {
        amplitude *= (1.0 / norm);
    }

    TimeDependentWavefunction3D wavefunction(psi, potential, n, n, n, spacing, kMass,
                                             kHbar);
    const double initialEnergy = wavefunction.expectationEnergy();
    ASSERT_GT(initialEnergy, 0.0);

    for (int step = 0; step < 100; ++step) {
        wavefunction.step(0.01);
    }

    // Same discretization-mismatch reasoning as
    // TimeDependentWavefunction1D's own energy-conservation test:
    // expectationEnergy() measures with a 7-point finite-difference
    // Hamiltonian, step() propagates with the FFT's spectral one -- a
    // small, non-accumulating offset between the two is the expected
    // signature of that mismatch, not drift.
    EXPECT_NEAR(wavefunction.expectationEnergy(), initialEnergy,
                std::abs(initialEnergy) * 0.1);
}

TEST(PhysicsQuantumMechanicsWavePacket3D, AnEigenstateStaysStationaryUnderEvolution) {
    // The 3D ground state of a separable Hamiltonian (Schrodinger3D.hpp's
    // own "sums of the 1D eigenvalues" derivation applies equally to
    // eigenvectors) is exactly the product of the three 1D ground states,
    // so this builds it from Schrodinger.hpp's already-fast, already-
    // validated 1D solver rather than the dense 3D eigensolver, which
    // would need an impractically slow O(N^3) solve to reach the same
    // per-axis resolution as the well-resolved n used here.
    constexpr std::size_t n = 16;
    constexpr double domainHalfWidth = 6.0;
    const double spacing = (2.0 * domainHalfWidth) / static_cast<double>(n);

    std::vector<double> potential1D(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -domainHalfWidth + static_cast<double>(i) * spacing;
        potential1D[i] = 0.5 * kMass * kOmega * kOmega * x * x;
    }
    const ysq::QuantumEigenstates states1D =
        ysq::solveTimeIndependentSchrodinger(potential1D, spacing, kMass, kHbar);
    const std::vector<double>& groundState1D = states1D.wavefunctions[0];

    std::vector<double> potential(n * n * n);
    std::vector<Complex<double>> groundState(n * n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t p = (i * n + j) * n + k;
                potential[p] = potential1D[i] + potential1D[j] + potential1D[k];
                groundState[p] = Complex<double>{
                    groundState1D[i] * groundState1D[j] * groundState1D[k], 0.0};
            }
        }
    }

    TimeDependentWavefunction3D wavefunction(groundState, potential, n, n, n, spacing,
                                             kMass, kHbar);

    std::vector<double> densityBefore(n * n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                densityBefore[(i * n + j) * n + k] =
                    wavefunction.probabilityDensity(i, j, k);
            }
        }
    }

    for (int step = 0; step < 100; ++step) {
        wavefunction.step(0.01);
    }

    // A wider floor than TimeDependentWavefunction1D's own 1% version: the
    // same finite-difference/spectral discretization mismatch applies
    // independently along all three axes here, compounding at the
    // density's peak (where the Laplacian's curvature, and so the
    // mismatch, is largest).
    const double peakDensity =
        *std::max_element(densityBefore.begin(), densityBefore.end());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t p = (i * n + j) * n + k;
                EXPECT_NEAR(wavefunction.probabilityDensity(i, j, k), densityBefore[p],
                            peakDensity * 0.25)
                    << "grid point " << i << "," << j << "," << k;
            }
        }
    }
}

TEST(PhysicsQuantumMechanicsWavePacket3D,
     ReducesExactlyToTheOneDimensionalSolverWhenFlatInYAndZ) {
    constexpr std::size_t nx = 16;
    constexpr double domainHalfWidth = 6.0;
    const double spacing = (2.0 * domainHalfWidth) / static_cast<double>(nx);

    std::vector<double> potential1D(nx);
    std::vector<Complex<double>> psi1D(nx);
    double normSquared = 0.0;
    for (std::size_t i = 0; i < nx; ++i) {
        const double x = -domainHalfWidth + static_cast<double>(i) * spacing;
        potential1D[i] = 0.5 * kMass * kOmega * kOmega * x * x;
        const double envelope = gaussian1D(x, 1.0, 1.0);
        psi1D[i] = Complex<double>{envelope, 0.0};
        normSquared += envelope * envelope;
    }
    const double norm = std::sqrt(normSquared * spacing);
    for (Complex<double>& amplitude : psi1D) {
        amplitude *= (1.0 / norm);
    }

    ysq::TimeDependentWavefunction1D wavefunction1D(psi1D, potential1D, spacing, kMass,
                                                    kHbar);
    // ny = nz = 1: fft3D's power-of-two precondition holds trivially (a
    // 1-point transform is the identity), and there is no other axis for
    // the field to vary along at all.
    TimeDependentWavefunction3D wavefunction3D(psi1D, potential1D, nx, 1, 1, spacing,
                                               kMass, kHbar);

    for (int step = 0; step < 100; ++step) {
        wavefunction1D.step(0.01);
        wavefunction3D.step(0.01);
    }

    for (std::size_t i = 0; i < nx; ++i) {
        const Complex<double> a = wavefunction1D.amplitude(i);
        const Complex<double> b = wavefunction3D.amplitude(i, 0, 0);
        EXPECT_NEAR(a.re, b.re, 1e-9) << "i = " << i;
        EXPECT_NEAR(a.im, b.im, 1e-9) << "i = " << i;
    }
}
