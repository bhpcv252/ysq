#include <Physics/Thermodynamics/Thermodynamics.hpp>
#include <Units/Constants.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Unit.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

TEST(Thermodynamics, StefanBoltzmannConstantMatchesTheKnownValue) {
    // CODATA: 5.670374419...e-8 W/(m^2 K^4), exact since the 2019
    // redefinition fixed h, k and c.
    EXPECT_NEAR(ysq::constants::stefanBoltzmann.value(), 5.670374419e-8, 1e-16);
}

TEST(Thermodynamics, IdealGasPressureMatchesTheDirectFormula) {
    // ISA sea-level density, exactly what the standard atmosphere is
    // defined to produce at these values.
    const ysq::Density density{1.225};
    const ysq::SpecificGasConstant airConstant{287.0};  // dry air
    const ysq::Temperature temperature{288.15};         // 15 C

    const ysq::Pressure pressure =
        ysq::idealGasPressure(density, airConstant, temperature);

    EXPECT_NEAR(pressure.value(), 1.225 * 287.0 * 288.15, 1e-6);
    EXPECT_NEAR(pressure.value(), ysq::units::atmosphere.value(),
                ysq::units::atmosphere.value() * 0.005);
}

TEST(Thermodynamics, AdiabaticCompressionToHalfVolumeRaisesPressure) {
    constexpr double gamma = 1.4;  // diatomic ideal gas
    const ysq::Pressure p1{1.0e5};
    const ysq::Volume v1{2.0};
    const ysq::Volume v2{1.0};

    const ysq::Pressure p2 = ysq::adiabaticPressure(p1, v1, v2, gamma);
    EXPECT_NEAR(p2.value(), p1.value() * std::pow(2.0, gamma), p1.value() * 1e-9);
    EXPECT_GT(p2, p1);
}

TEST(Thermodynamics, AdiabaticRelationRoundTrips) {
    constexpr double gamma = 5.0 / 3.0;  // monatomic
    const ysq::Pressure p1{2.0e5};
    const ysq::Volume v1{0.5};
    const ysq::Volume v2{0.3};

    const ysq::Pressure p2 = ysq::adiabaticPressure(p1, v1, v2, gamma);
    const ysq::Pressure roundTripped = ysq::adiabaticPressure(p2, v2, v1, gamma);
    EXPECT_QUANTITY_NEAR(roundTripped, p1, p1 * 1e-9);
}

TEST(Thermodynamics, BlackBodyLuminosityOfASolarLikeStarIsOfTheRightOrderOfMagnitude) {
    // Not the IAU nominal L_sun, which fixes a specific reference
    // temperature and radius by convention (docs/units.md): this checks
    // the formula against realistic solar parameters instead, to within
    // the accuracy those parameters themselves carry.
    const ysq::Length radius = ysq::units::solarRadius;
    const ysq::Temperature temperature{5772.0};

    const ysq::Power luminosity = ysq::blackBodyLuminosity(radius, temperature);
    EXPECT_NEAR(luminosity.value(), 3.828e26, 3.828e26 * 0.02);
}

TEST(Thermodynamics, WienPeakWavelengthOfTheSunIsInVisibleLight) {
    const ysq::Temperature temperature{5772.0};
    const ysq::Length peak = ysq::wienPeakWavelength(temperature);

    // Visible light spans roughly 380-750 nm.
    EXPECT_GT(peak.value(), 380.0e-9);
    EXPECT_LT(peak.value(), 750.0e-9);
}

TEST(Thermodynamics, WiensLawIsExactlyItsOwnDefinition) {
    const ysq::Temperature temperature{1000.0};
    const ysq::Length peak = ysq::wienPeakWavelength(temperature);
    EXPECT_NEAR(peak.value() * temperature.value(),
                ysq::constants::wienDisplacementConstant.value(), 1e-15);
}

}  // namespace
