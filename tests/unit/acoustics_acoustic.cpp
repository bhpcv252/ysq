#include <Physics/Acoustics/Acoustic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr std::size_t kCellCount = 100;
constexpr double kSpacing = 0.01;
constexpr double kMediumDensity = 1.2;  // roughly air, kg/m^3
constexpr double kSoundSpeed = 343.0;   // roughly air at room temperature, m/s

double gaussian(double x, double center, double width) {
    const double normalized = (x - center) / width;
    return std::exp(-normalized * normalized);
}

/// A right-moving acoustic pulse: velocity = pressure / (rho0 c) is the
/// plane-wave relation for a wave whose energy flux points in +x, the
/// acoustic analog of Maxwell's Bz = Ey / c.
ysq::AcousticField1D makeRightMovingPulse(double center, double width) {
    ysq::AcousticField1D field(kCellCount, kSpacing, kMediumDensity, kSoundSpeed);
    const double impedance = kMediumDensity * kSoundSpeed;

    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double x = static_cast<double>(i) * kSpacing;
        field.setPressure(i, gaussian(x, center, width));
        field.setVelocity(i, gaussian(x + kSpacing / 2.0, center, width) / impedance);
    }
    return field;
}

}  // namespace

TEST(AcousticsAcoustic, MagicTimeStepIsSpacingOverSoundSpeed) {
    EXPECT_NEAR(ysq::magicAcousticTimeStep(2.0, kSoundSpeed), 2.0 / kSoundSpeed, 1e-15);
}

TEST(AcousticsAcoustic, ARightMovingPulsePeakTravelsAtExactlySoundSpeed) {
    constexpr std::size_t startCell = 30;
    ysq::AcousticField1D field =
        makeRightMovingPulse(static_cast<double>(startCell) * kSpacing, 5.0 * kSpacing);

    const double dt = ysq::magicAcousticTimeStep(kSpacing, kSoundSpeed);
    constexpr std::size_t steps = 10;
    for (std::size_t s = 0; s < steps; ++s) {
        field.step(dt);
    }

    std::size_t peakCell = 0;
    double peakValue = field.pressure(0);
    for (std::size_t i = 1; i < kCellCount; ++i) {
        if (field.pressure(i) > peakValue) {
            peakValue = field.pressure(i);
            peakCell = i;
        }
    }

    // At the magic time step, each step advances the wave by exactly one
    // spacing, so the peak should have moved by exactly `steps` cells.
    EXPECT_EQ(peakCell, startCell + steps);
}

TEST(AcousticsAcoustic, EnergyIsConservedOverManySteps) {
    ysq::AcousticField1D field = makeRightMovingPulse(50.0 * kSpacing, 5.0 * kSpacing);
    const double initialEnergy = field.totalEnergy();
    ASSERT_GT(initialEnergy, 0.0);

    const double dt = ysq::magicAcousticTimeStep(kSpacing, kSoundSpeed);
    for (std::size_t s = 0; s < 500; ++s) {
        field.step(dt);
    }

    EXPECT_NEAR(field.totalEnergy(), initialEnergy, initialEnergy * 1e-6);
}

TEST(AcousticsAcoustic, AZeroFieldStaysZero) {
    ysq::AcousticField1D field(kCellCount, kSpacing, kMediumDensity, kSoundSpeed);
    const double dt = ysq::magicAcousticTimeStep(kSpacing, kSoundSpeed);
    for (std::size_t s = 0; s < 50; ++s) {
        field.step(dt);
    }
    for (std::size_t i = 0; i < kCellCount; ++i) {
        EXPECT_NEAR(field.pressure(i), 0.0, 1e-30);
        EXPECT_NEAR(field.velocity(i), 0.0, 1e-30);
    }
}
