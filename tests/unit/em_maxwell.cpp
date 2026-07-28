#include <Physics/Electromagnetism/Maxwell.hpp>
#include <Units/Constants.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr std::size_t kCellCount = 100;
constexpr double kSpacing = 0.01;

double gaussian(double x, double center, double width) {
    const double normalized = (x - center) / width;
    return std::exp(-normalized * normalized);
}

/// A right-moving TEM pulse: Bz = Ey / c is the vacuum relation for a wave
/// whose Poynting vector, E x B, points in +x.
ysq::MaxwellField1D makeRightMovingPulse(double center, double width) {
    ysq::MaxwellField1D field(kCellCount, kSpacing);
    const double c = ysq::constants::speedOfLight.value();

    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double x = static_cast<double>(i) * kSpacing;
        field.setElectricField(i, gaussian(x, center, width));
        field.setMagneticField(i, gaussian(x + kSpacing / 2.0, center, width) / c);
    }
    return field;
}

TEST(ElectromagnetismMaxwell, MagicTimeStepIsSpacingOverC) {
    const double c = ysq::constants::speedOfLight.value();
    EXPECT_NEAR(ysq::magicTimeStep(2.0), 2.0 / c, 1e-25);
}

TEST(ElectromagnetismMaxwell, ARightMovingPulsePeakTravelsAtExactlyC) {
    constexpr std::size_t startCell = 30;
    ysq::MaxwellField1D field =
        makeRightMovingPulse(static_cast<double>(startCell) * kSpacing, 5.0 * kSpacing);

    const double dt = ysq::magicTimeStep(kSpacing);
    constexpr std::size_t steps = 10;
    for (std::size_t s = 0; s < steps; ++s) {
        field.step(dt);
    }

    std::size_t peakCell = 0;
    double peakValue = field.electricField(0);
    for (std::size_t i = 1; i < kCellCount; ++i) {
        if (field.electricField(i) > peakValue) {
            peakValue = field.electricField(i);
            peakCell = i;
        }
    }

    // At the magic time step, each step advances the wave by exactly one
    // spacing, so the peak should have moved by exactly `steps` cells.
    EXPECT_EQ(peakCell, startCell + steps);
}

TEST(ElectromagnetismMaxwell, EnergyIsConservedOverManySteps) {
    ysq::MaxwellField1D field = makeRightMovingPulse(50.0 * kSpacing, 5.0 * kSpacing);
    const double initialEnergy = field.totalEnergy();
    ASSERT_GT(initialEnergy, 0.0);

    const double dt = ysq::magicTimeStep(kSpacing);
    for (std::size_t s = 0; s < 500; ++s) {
        field.step(dt);
    }

    EXPECT_NEAR(field.totalEnergy(), initialEnergy, initialEnergy * 1e-6);
}

TEST(ElectromagnetismMaxwell, AZeroFieldStaysZero) {
    ysq::MaxwellField1D field(kCellCount, kSpacing);
    const double dt = ysq::magicTimeStep(kSpacing);
    for (std::size_t s = 0; s < 50; ++s) {
        field.step(dt);
    }
    for (std::size_t i = 0; i < kCellCount; ++i) {
        EXPECT_NEAR(field.electricField(i), 0.0, 1e-30);
        EXPECT_NEAR(field.magneticField(i), 0.0, 1e-30);
    }
}

}  // namespace
