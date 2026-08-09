#include <Physics/Electromagnetism/Maxwell3D.hpp>

#include <Math/Scalar.hpp>
#include <Physics/Electromagnetism/Maxwell.hpp>
#include <Units/Constants.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

namespace {

constexpr std::size_t kCellCount = 16;
constexpr double kSpacing = 0.01;

double gaussian(double x, double center, double width) {
    const double normalized = (x - center) / width;
    return std::exp(-normalized * normalized);
}

/// A transverse plane wave travelling along the grid diagonal
/// `(1,1,1)/sqrt(3)`: E polarized along `(1,-1,0)/sqrt(2)` (perpendicular
/// to the propagation direction), B along `(1,1,-2)/sqrt(6)` (the vacuum
/// relation `B = (k_hat x E_hat)/c`), one full period across the domain
/// along each axis (`k1 = 2 pi / (n h)`) so the field is exactly periodic.
/// A single Fourier mode like this is an exact eigenmode of the periodic
/// finite-difference curl operator regardless of any numerical dispersion
/// in its time evolution, unlike a spatially localized pulse, which is a
/// superposition of many modes each dispersing at a slightly different
/// rate -- the reason this (not a pulse) is the right initial condition
/// for a genuinely 3D energy-conservation check.
ysq::MaxwellField3D makeDiagonalPlaneWave(std::size_t n, double h) {
    const double c = ysq::constants::speedOfLight.value();
    const double k1 = ysq::kTau<double> / (static_cast<double>(n) * h);

    const double exAmplitude = 1.0 / std::sqrt(2.0);
    const double eyAmplitude = -1.0 / std::sqrt(2.0);
    const double bxAmplitude = 1.0 / (c * std::sqrt(6.0));
    const double byAmplitude = 1.0 / (c * std::sqrt(6.0));
    const double bzAmplitude = -2.0 / (c * std::sqrt(6.0));

    ysq::MaxwellField3D field(n, n, n, h);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const double x = static_cast<double>(i) * h;
                const double y = static_cast<double>(j) * h;
                const double z = static_cast<double>(k) * h;
                field.setElectricFieldX(
                    i, j, k,
                    exAmplitude * std::cos(k1 * (x + h / 2.0) + k1 * y + k1 * z));
                field.setElectricFieldY(
                    i, j, k,
                    eyAmplitude * std::cos(k1 * x + k1 * (y + h / 2.0) + k1 * z));
                field.setMagneticFieldX(
                    i, j, k,
                    bxAmplitude *
                        std::cos(k1 * x + k1 * (y + h / 2.0) + k1 * (z + h / 2.0)));
                field.setMagneticFieldY(
                    i, j, k,
                    byAmplitude *
                        std::cos(k1 * (x + h / 2.0) + k1 * y + k1 * (z + h / 2.0)));
                field.setMagneticFieldZ(
                    i, j, k,
                    bzAmplitude *
                        std::cos(k1 * (x + h / 2.0) + k1 * (y + h / 2.0) + k1 * z));
            }
        }
    }
    return field;
}

}  // namespace

TEST(ElectromagnetismMaxwell3D, StableTimeStepMatchesTheThreeDimensionalCflLimit) {
    const double c = ysq::constants::speedOfLight.value();
    const double expected = kSpacing / (c * std::sqrt(3.0));
    EXPECT_NEAR(ysq::MaxwellField3D(4, 4, 4, kSpacing).stableTimeStep(1.0), expected,
                1e-25);
}

TEST(ElectromagnetismMaxwell3D, AZeroFieldStaysZero) {
    ysq::MaxwellField3D field(kCellCount, kCellCount, kCellCount, kSpacing);
    const double dt = field.stableTimeStep(0.5);
    for (int s = 0; s < 20; ++s) {
        field.step(dt);
    }
    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kCellCount; ++j) {
            for (std::size_t k = 0; k < kCellCount; ++k) {
                EXPECT_NEAR(field.electricFieldX(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field.electricFieldY(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field.electricFieldZ(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field.magneticFieldX(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field.magneticFieldY(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field.magneticFieldZ(i, j, k), 0.0, 1e-30);
            }
        }
    }
}

TEST(ElectromagnetismMaxwell3D, EnergyIsBoundedOverManyStepsForADiagonalPlaneWave) {
    // Off the (nonexistent, in 3D) magic step, even a single Fourier mode's
    // discrete energy is not an exact invariant of this leapfrog scheme --
    // only MaxwellField1D at its own magic step gets that -- but it stays
    // bounded, oscillating within a fixed band rather than drifting away,
    // confirmed here over ten times the step count the tolerance below is
    // sized from: the measured oscillation band is a few percent wide and
    // does not grow with more steps.
    constexpr std::size_t n = 32;
    ysq::MaxwellField3D field = makeDiagonalPlaneWave(n, kSpacing);

    const double initialEnergy = field.totalEnergy();
    ASSERT_GT(initialEnergy, 0.0);

    const double dt = field.stableTimeStep(0.5);
    double maxRelativeDeviation = 0.0;
    for (int s = 0; s < 2000; ++s) {
        field.step(dt);
        const double deviation =
            std::abs(field.totalEnergy() - initialEnergy) / initialEnergy;
        maxRelativeDeviation = std::max(maxRelativeDeviation, deviation);
    }

    EXPECT_LT(maxRelativeDeviation, 0.05);
}

TEST(ElectromagnetismMaxwell3D, ReducesExactlyToTheOneDimensionalSolverWhenFlatInYAndZ) {
    constexpr std::size_t kFlatCount = 4;
    ysq::MaxwellField3D field3D(kCellCount, kFlatCount, kFlatCount, kSpacing);
    ysq::MaxwellField1D field1D(kCellCount, kSpacing);
    const double c = ysq::constants::speedOfLight.value();

    const double center = static_cast<double>(kCellCount / 2) * kSpacing;
    const double width = 3.0 * kSpacing;

    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double x = static_cast<double>(i) * kSpacing;
        const double ey = gaussian(x, center, width);
        const double bz = gaussian(x + kSpacing / 2.0, center, width) / c;
        field1D.setElectricField(i, ey);
        field1D.setMagneticField(i, bz);
        for (std::size_t j = 0; j < kFlatCount; ++j) {
            for (std::size_t k = 0; k < kFlatCount; ++k) {
                field3D.setElectricFieldY(i, j, k, ey);
                field3D.setMagneticFieldZ(i, j, k, bz);
            }
        }
    }

    const double dt = ysq::magicTimeStep(kSpacing);
    for (int s = 0; s < 100; ++s) {
        field1D.step(dt);
        field3D.step(dt);
    }

    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kFlatCount; ++j) {
            for (std::size_t k = 0; k < kFlatCount; ++k) {
                EXPECT_NEAR(field3D.electricFieldY(i, j, k), field1D.electricField(i),
                            1e-9)
                    << "cell " << i << "," << j << "," << k;
                EXPECT_NEAR(field3D.magneticFieldZ(i, j, k), field1D.magneticField(i),
                            1e-9)
                    << "cell " << i << "," << j << "," << k;
                EXPECT_NEAR(field3D.electricFieldX(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field3D.electricFieldZ(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field3D.magneticFieldX(i, j, k), 0.0, 1e-30);
                EXPECT_NEAR(field3D.magneticFieldY(i, j, k), 0.0, 1e-30);
            }
        }
    }
}

TEST(ElectromagnetismMaxwell3D, ADiagonalPlaneWavePropagatesAtApproximatelyC) {
    // A genuinely 3D case (transverse E, B both fully populated, along a
    // grid diagonal) the dimensional-reduction test above cannot exercise.
    // Measures the actual angular frequency the simulation produces and
    // compares it to the continuum omega = c|k|, to a tolerance sized for
    // a second-order scheme's O((k h)^2) dispersion error, rather than a
    // hand-derived exact discrete dispersion formula.
    constexpr std::size_t n = 32;
    const double h = kSpacing;
    const double c = ysq::constants::speedOfLight.value();
    const double k1 = ysq::kTau<double> / (static_cast<double>(n) * h);
    const double omegaContinuum = c * k1 * std::sqrt(3.0);

    ysq::MaxwellField3D field = makeDiagonalPlaneWave(n, h);

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
                const double x = static_cast<double>(i) * h;
                const double y = static_cast<double>(j) * h;
                const double z = static_cast<double>(k) * h;
                const double phase = k1 * (x + h / 2.0) + k1 * y + k1 * z;
                const double ex = field.electricFieldX(i, j, k);
                cosProjection += ex * std::cos(phase);
                sinProjection += ex * std::sin(phase);
            }
        }
    }
    const double omegaMeasured = std::atan2(sinProjection, cosProjection) / t;

    EXPECT_NEAR(omegaMeasured, omegaContinuum, omegaContinuum * 0.05);
}
