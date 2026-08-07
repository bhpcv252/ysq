/// Unit conversion, and which conversions are exact.
///
/// A unit is not part of a quantity's type. It is a quantity of magnitude one
/// unit, so converting in is multiplication and converting out is division,
/// and there is no separate mechanism to get wrong.
///
/// Exactness is asserted rather than assumed. Some of these round-trip
/// bit-for-bit and some cannot, and which is which is a property of the
/// definitions, not of the code: a kilometre is exactly a thousand metres, the
/// astronomical unit has been an exact number of metres since IAU 2012, and a
/// parsec carries a factor of pi and therefore cannot be. Asserting equality
/// where equality holds is what would catch a factor quietly rounded when
/// someone retypes it. src/Units/README.md carries the values and their sources.

#include <Units/Acceleration.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Luminosity.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using namespace ysq;
using namespace ysq::literals;

}  // namespace

TEST(UnitsConversions, AUnitIsAQuantityOfMagnitudeOneUnit) {
    static_assert(units::metre.value() == 1.0);
    static_assert(units::second.value() == 1.0);
    static_assert(units::kilogram.value() == 1.0);
    static_assert(units::kelvin.value() == 1.0);
    static_assert(units::candela.value() == 1.0);
    static_assert(units::newton.value() == 1.0);
    static_assert(units::joule.value() == 1.0);
    SUCCEED();
}

TEST(UnitsConversions, DecimalPrefixesRoundTripExactly) {
    // Every factor here is a power of ten with an exact binary representation
    // at the magnitudes involved, so these are equalities, not tolerances.
    static_assert(5.0 * units::kilometre / units::kilometre == 5.0);
    static_assert(5.0 * units::centimetre / units::centimetre == 5.0);
    static_assert((5.0 * units::kilometre).value() == 5000.0);
    static_assert((250.0 * units::millimetre).value() == 0.25);
    static_assert((1.0 * units::tonne).value() == 1000.0);
    static_assert((1.0 * units::gram).value() == 1.0e-3);

    // And the accessor form reads the same as the division.
    static_assert((5.0 * units::kilometre).in(units::kilometre) == 5.0);
    static_assert((5.0 * units::kilometre).in(units::metre) == 5000.0);
    SUCCEED();
}

TEST(UnitsConversions, TimeUnitsAreExactMultiplesOfTheSecond) {
    static_assert((1.0 * units::minute).value() == 60.0);
    static_assert((1.0 * units::hour).value() == 3600.0);
    static_assert((1.0 * units::day).value() == 86400.0);
    static_assert((1.0 * units::week).value() == 7.0 * 86400.0);

    // The Julian year, exactly 365.25 days. Not the tropical year and not a
    // calendar year, neither of which is a fixed number of seconds.
    static_assert((1.0 * units::year).value() == 365.25 * 86400.0);
    static_assert((1.0 * units::year).value() == 31557600.0);
    static_assert(units::megayear == units::year * 1.0e6);
    static_assert(units::gigayear == units::year * 1.0e9);

    // A twelfth of the Julian year, not a calendar month.
    static_assert((1.0 * units::month).value() == 31557600.0 / 12.0);
    SUCCEED();
}

TEST(UnitsConversions, AstronomicalLengthsCarryTheirDefinitions) {
    // Exact since IAU 2012 Resolution B2: the au is a defined number of
    // metres, and 149597870700 is an integer well inside the range a double
    // represents exactly.
    static_assert(units::astronomicalUnit.value() == 149597870700.0);
    static_assert(3.0 * units::astronomicalUnit / units::astronomicalUnit == 3.0);

    // The light-year is exact for a less obvious reason: it is the product of
    // two exact integers, and although 9460730472580800 is larger than 2^53,
    // it has enough trailing zero bits that the product is still representable
    // to the last bit.
    static_assert(units::lightYear == constants::speedOfLight * units::year);
    static_assert(units::lightYear.value() == 9460730472580800.0);

    // The parsec is (648000 / pi) au, so it is irrational and this one is a
    // tolerance rather than an equality. It sits between three and four
    // light-years, which is the sanity check worth having.
    EXPECT_GT(units::parsec, units::lightYear * 3.0);
    EXPECT_LT(units::parsec, units::lightYear * 4.0);
    EXPECT_QUANTITY_NEAR(units::parsec, 206264.806 * units::astronomicalUnit,
                         units::metre * 1.0e8);
}

TEST(UnitsConversions, LiteralsAgreeWithTheirUnitConstants) {
    static_assert(1.0_m == units::metre);
    static_assert(1_m == units::metre);
    static_assert(2.5_km == units::kilometre * 2.5);
    static_assert(3_au == units::astronomicalUnit * 3.0);
    static_assert(1.0_pc == units::parsec);
    static_assert(1.0_ly == units::lightYear);

    static_assert(1.0_kg == units::kilogram);
    static_assert(1.0_Msun == units::solarMass);

    static_assert(1.0_s == units::second);
    static_assert(1.0_min == units::minute);
    static_assert(1.0_h == units::hour);
    static_assert(1.0_yr == units::year);

    static_assert(1.0_N == units::newton);
    static_assert(1.0_J == units::joule);
    static_assert(1.0_eV == units::electronvolt);
    static_assert(1.0_K == units::kelvin);
    static_assert(1.0_Lsun == units::solarLuminosity);
    SUCCEED();
}

TEST(UnitsConversions, DerivedUnitsFallOutOfTheAlgebra) {
    // Nothing here is a hand-written factor. Each is the product the
    // definition says it is, which is the point of having the algebra.
    static_assert(units::newton == units::kilogram * units::metrePerSecondSquared);
    static_assert(units::joule == units::newton * units::metre);
    static_assert(units::watt == units::joule / units::second);
    static_assert(units::pascal == units::newton / units::squareMetre);
    static_assert(units::hertz == 1.0 / units::second);
    static_assert(units::metrePerSecond == units::metre / units::second);

    // Non-SI ones that are exact by definition.
    static_assert(units::dyne ==
                  units::gram * units::centimetre / (units::second * units::second));
    static_assert(units::erg.value() == 1.0e-7);
    static_assert(units::atmosphere.value() == 101325.0);
    static_assert((1.0 * units::bar).value() == 1.0e5);
    SUCCEED();
}

TEST(UnitsConversions, SpeedConversionsAreConsistent) {
    static_assert(units::kilometrePerSecond == units::kilometre / units::second);
    static_assert(1.0 * units::kilometrePerHour == units::kilometre / units::hour);
    static_assert(units::speedOfLight == constants::speedOfLight);

    EXPECT_QUANTITY_APPROX(7.8_kmps, 7800.0 * units::metrePerSecond);
    EXPECT_QUANTITY_NEAR(1.0 * units::kilometrePerHour, 0.2777778 * units::metrePerSecond,
                         1.0e-6 * units::metrePerSecond);
}

TEST(UnitsConversions, TheDefiningConstantsAreTheirDefinedValues) {
    // Since the 2019 redefinition these are exact by fiat, not measured, and
    // every SI base unit is derived from them. If one of these is ever wrong,
    // everything above it is wrong in a way no physics test would localize.
    static_assert(constants::caesiumHyperfineFrequency.value() == 9192631770.0);
    static_assert(constants::speedOfLight.value() == 299792458.0);
    static_assert(constants::planckConstant.value() == 6.62607015e-34);
    static_assert(constants::elementaryCharge.value() == 1.602176634e-19);
    static_assert(constants::boltzmannConstant.value() == 1.380649e-23);
    static_assert(constants::avogadroConstant.value() == 6.02214076e23);
    static_assert(constants::luminousEfficacy.value() == 683.0);

    // The electronvolt is the elementary charge times one volt, so it is
    // exact for the same reason.
    static_assert(units::electronvolt.value() == constants::elementaryCharge.value());

    EXPECT_QUANTITY_NEAR(constants::reducedPlanckConstant,
                         1.054571817e-34 * units::joule * units::second,
                         1.0e-42 * units::joule * units::second);
}

TEST(UnitsConversions, NominalSolarValuesAreConventionsNotMeasurements) {
    // What the IAU fixes exactly is GM, not M. The mass in kilograms is GM
    // divided by a measured G, so it is the lossy form; src/Units/README.md
    // records the G used. This test pins the relationship so the two cannot drift
    // apart when either is retyped.
    constexpr double gravitationalConstant = 6.67430e-11;
    const Mass reconstructed{constants::nominalSolarMassParameter.value() /
                             gravitationalConstant};

    EXPECT_QUANTITY_NEAR(reconstructed, units::solarMass, units::solarMass * 1.0e-12);

    const Mass earthReconstructed{constants::nominalEarthMassParameter.value() /
                                  gravitationalConstant};
    EXPECT_QUANTITY_NEAR(earthReconstructed, units::earthMass,
                         units::earthMass * 1.0e-12);

    // Sanity against the values everyone knows.
    EXPECT_QUANTITY_NEAR(units::solarMass, 1.989e30 * units::kilogram,
                         1.0e27 * units::kilogram);
    EXPECT_QUANTITY_NEAR(units::earthMass, 5.972e24 * units::kilogram,
                         1.0e21 * units::kilogram);
    EXPECT_GT(units::solarMass / units::earthMass, 332000.0);
    EXPECT_LT(units::solarMass / units::earthMass, 334000.0);
}

TEST(UnitsConversions, TemperatureScalesAreAffineAndSoAreFunctions) {
    // Not unit constants, because there is no factor that converts between
    // scales with different origins. Unit.hpp explains why that distinction
    // has to be visible in the API rather than hidden in a number.
    static_assert(fromCelsius(0.0) == Temperature{273.15});
    static_assert(fromCelsius(100.0) == Temperature{373.15});
    static_assert(fromFahrenheit(32.0) == fromCelsius(0.0));

    EXPECT_DOUBLE_EQ(toCelsius(fromCelsius(21.5)), 21.5);
    EXPECT_DOUBLE_EQ(toFahrenheit(fromFahrenheit(98.6)), 98.6);
    EXPECT_NEAR(toFahrenheit(fromCelsius(100.0)), 212.0, 1e-9);

    // The two scales cross at -40, which is the one value that catches a
    // slope and an offset that are both wrong in compensating ways.
    EXPECT_NEAR(toFahrenheit(fromCelsius(-40.0)), -40.0, 1e-9);

    // Absolute zero, and the fact that nothing stops a nonsensical one being
    // constructed. Units does not police physics; it polices dimensions.
    EXPECT_DOUBLE_EQ(toCelsius(Temperature{0.0}), -273.15);
}

TEST(UnitsConversions, LuminosityUsesTheNominalSolarValue) {
    static_assert(units::solarLuminosity.value() == 3.828e26);

    // Radiometric power is power. There is no separate radiant watt, and the
    // type system agrees, which is why Luminosity.hpp defines no unit constant
    // of its own for it.
    static_assert(std::is_same_v<RadiantPower, Power>);

    EXPECT_QUANTITY_APPROX(1.0_Lsun, units::solarLuminosity);
    EXPECT_QUANTITY_NEAR(4.0_Lsun, 1.5312e27 * units::watt, 1.0e20 * units::watt);
}
