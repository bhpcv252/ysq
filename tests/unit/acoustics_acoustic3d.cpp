#include <Physics/Acoustics/Acoustic3D.hpp>

#include <Math/Scalar.hpp>
#include <Physics/Acoustics/Acoustic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr std::size_t kCellCount = 16;
constexpr double kSpacing = 0.01;
constexpr double kMediumDensity = 1.2;
constexpr double kSoundSpeed = 343.0;

double gaussian(double x, double center, double width) {
    const double normalized = (x - center) / width;
    return std::exp(-normalized * normalized);
}

}  // namespace

TEST(AcousticsAcoustic3D, StableTimeStepMatchesTheThreeDimensionalCflLimit) {
    const double expected = kSpacing / (kSoundSpeed * std::sqrt(3.0));
    EXPECT_NEAR(ysq::AcousticField3D(4, 4, 4, kSpacing, kMediumDensity, kSoundSpeed)
                    .stableTimeStep(1.0),
                expected, 1e-15);
}

TEST(AcousticsAcoustic3D, AZeroFieldStaysZero) {
    ysq::AcousticField3D field(kCellCount, kCellCount, kCellCount, kSpacing,
                               kMediumDensity, kSoundSpeed);
    const double dt = field.stableTimeStep(0.5);
    for (int s = 0; s < 20; ++s) {
        field.step(dt);
    }
    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kCellCount; ++j) {
            for (std::size_t k = 0; k < kCellCount; ++k) {
                EXPECT_NEAR(field.pressure(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field.velocityX(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field.velocityY(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field.velocityZ(i, j, k), 0.0, 1e-30);
            }
        }
    }
}

TEST(AcousticsAcoustic3D, EnergyIsConservedOverManySteps) {
    // Unlike AcousticField1D at its own magic timestep -- exact for any
    // initial data, per Acoustic3D.hpp's own doc comment on why no such
    // step exists in 3D -- a sub-CFL 3D step has genuine, small numerical
    // dispersion, so energy is only bounded here, not exactly constant.
    // A larger, better-resolved grid than the other tests here keeps that
    // dispersion small.
    constexpr std::size_t cellCount = 24;
    ysq::AcousticField3D field(cellCount, cellCount, cellCount, kSpacing, kMediumDensity,
                               kSoundSpeed);
    const double center = static_cast<double>(cellCount / 2) * kSpacing;
    const double width = 5.0 * kSpacing;
    const double impedance = kMediumDensity * kSoundSpeed;

    for (std::size_t i = 0; i < cellCount; ++i) {
        for (std::size_t j = 0; j < cellCount; ++j) {
            for (std::size_t k = 0; k < cellCount; ++k) {
                const double x = static_cast<double>(i) * kSpacing;
                const double p = gaussian(x, center, width);
                field.setPressure(i, j, k, p);
                field.setVelocityX(
                    i, j, k, gaussian(x + kSpacing / 2.0, center, width) / impedance);
            }
        }
    }

    const double initialEnergy = field.totalEnergy();
    ASSERT_GT(initialEnergy, 0.0);

    const double dt = field.stableTimeStep(0.3);
    for (int s = 0; s < 200; ++s) {
        field.step(dt);
    }

    EXPECT_NEAR(field.totalEnergy(), initialEnergy, initialEnergy * 1e-3);
}

TEST(AcousticsAcoustic3D, ReducesExactlyToTheOneDimensionalSolverWhenFlatInYAndZ) {
    constexpr std::size_t kFlatCount = 4;
    ysq::AcousticField3D field3D(kCellCount, kFlatCount, kFlatCount, kSpacing,
                                 kMediumDensity, kSoundSpeed);
    ysq::AcousticField1D field1D(kCellCount, kSpacing, kMediumDensity, kSoundSpeed);

    const double center = static_cast<double>(kCellCount / 2) * kSpacing;
    const double width = 3.0 * kSpacing;
    const double impedance = kMediumDensity * kSoundSpeed;

    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double x = static_cast<double>(i) * kSpacing;
        const double p = gaussian(x, center, width);
        const double u = gaussian(x + kSpacing / 2.0, center, width) / impedance;
        field1D.setPressure(i, p);
        field1D.setVelocity(i, u);
        for (std::size_t j = 0; j < kFlatCount; ++j) {
            for (std::size_t k = 0; k < kFlatCount; ++k) {
                field3D.setPressure(i, j, k, p);
                field3D.setVelocityX(i, j, k, u);
            }
        }
    }

    const double dt = ysq::magicAcousticTimeStep(kSpacing, kSoundSpeed);
    for (int s = 0; s < 100; ++s) {
        field1D.step(dt);
        field3D.step(dt);
    }

    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kFlatCount; ++j) {
            for (std::size_t k = 0; k < kFlatCount; ++k) {
                EXPECT_NEAR(field3D.pressure(i, j, k), field1D.pressure(i), 1e-9)
                    << "cell " << i << "," << j << "," << k;
                EXPECT_NEAR(field3D.velocityX(i, j, k), field1D.velocity(i), 1e-9)
                    << "cell " << i << "," << j << "," << k;
                EXPECT_NEAR(field3D.velocityY(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field3D.velocityZ(i, j, k), 0.0, 1e-30);
            }
        }
    }
}

TEST(AcousticsAcoustic3D,
     ADiagonalPlaneWavePropagatesAtApproximatelyTheContinuumSoundSpeed) {
    // A genuinely 3D case the dimensional-reduction test above cannot
    // exercise: a plane wave travelling along the grid diagonal, all three
    // velocity components active at once. A second-order scheme's phase
    // velocity is not exactly c off-axis on a Cartesian grid (see
    // Acoustic3D.hpp's own doc comment on why no "magic timestep" exists
    // in 3D), but converges to it as O((k h)^2) for a well-resolved wave,
    // so this checks the *measured* angular frequency against the
    // continuum prediction to a tolerance sized for that truncation order,
    // rather than a hand-derived exact discrete dispersion formula.
    constexpr std::size_t n = 32;
    const double h = kSpacing;
    const double k1 =
        ysq::kTau<double> / (static_cast<double>(n) * h);  // one full period per axis
    const double kMagnitude = k1 * std::sqrt(3.0);
    const double omegaContinuum = kSoundSpeed * kMagnitude;

    ysq::AcousticField3D field(n, n, n, h, kMediumDensity, kSoundSpeed);
    const double impedance = kMediumDensity * kSoundSpeed;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const double x = static_cast<double>(i) * h;
                const double y = static_cast<double>(j) * h;
                const double z = static_cast<double>(k) * h;
                field.setPressure(i, j, k, std::cos(k1 * x + k1 * y + k1 * z));
                // Longitudinal velocity along the propagation direction,
                // each component evaluated at its own staggered position.
                const double component = (k1 / kMagnitude) / impedance;
                field.setVelocityX(
                    i, j, k, component * std::cos(k1 * (x + h / 2.0) + k1 * y + k1 * z));
                field.setVelocityY(
                    i, j, k, component * std::cos(k1 * x + k1 * (y + h / 2.0) + k1 * z));
                field.setVelocityZ(
                    i, j, k, component * std::cos(k1 * x + k1 * y + k1 * (z + h / 2.0)));
            }
        }
    }

    const double dt = field.stableTimeStep(0.5);
    constexpr int steps = 15;
    for (int s = 0; s < steps; ++s) {
        field.step(dt);
    }
    const double t = steps * dt;

    double cosProjection = 0.0;
    double sinProjection = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const double phase = k1 * static_cast<double>(i) * h +
                                     k1 * static_cast<double>(j) * h +
                                     k1 * static_cast<double>(k) * h;
                const double p = field.pressure(i, j, k);
                cosProjection += p * std::cos(phase);
                sinProjection += p * std::sin(phase);
            }
        }
    }
    const double omegaMeasured = std::atan2(sinProjection, cosProjection) / t;

    EXPECT_NEAR(omegaMeasured, omegaContinuum, omegaContinuum * 0.05);
}
