/// Formatting: base-unit powers, and nothing invented.
///
/// The formatter never prints a named derived symbol. An energy renders as
/// `m^2 kg s^-2` rather than `J`, and so does a torque, because the two share
/// a dimension and this module cannot tell them apart. Printing `J` would
/// assert a distinction the type system does not carry and would be wrong
/// half the time it mattered. Base-unit powers are mechanical, unambiguous,
/// and claim exactly as much as is known.

#include <Units/Acceleration.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Format.hpp>
#include <Units/Length.hpp>
#include <Units/Luminosity.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace {

using namespace ysq;

}  // namespace

TEST(UnitsFormat, BaseUnitsPrintTheirSymbol) {
    EXPECT_EQ(std::format("{}", Length{5.0}), "5 m");
    EXPECT_EQ(std::format("{}", Mass{2.0}), "2 kg");
    EXPECT_EQ(std::format("{}", Time{3.0}), "3 s");
    EXPECT_EQ(std::format("{}", Temperature{300.0}), "300 K");
    EXPECT_EQ(std::format("{}", LuminousIntensity{1.0}), "1 cd");
    EXPECT_EQ(std::format("{}", ElectricCharge{1.0}), "1 s A");
}

TEST(UnitsFormat, DerivedUnitsPrintAsBaseUnitPowers) {
    // SI order, m kg s A K mol cd, with signed exponents and no named
    // symbols.
    EXPECT_EQ(std::format("{}", Area{4.0}), "4 m^2");
    EXPECT_EQ(std::format("{}", Volume{8.0}), "8 m^3");
    EXPECT_EQ(std::format("{}", Speed{10.0}), "10 m s^-1");
    EXPECT_EQ(std::format("{}", Acceleration{9.8}), "9.8 m s^-2");
    EXPECT_EQ(std::format("{}", Force{1.0}), "1 m kg s^-2");
    EXPECT_EQ(std::format("{}", Energy{1.0}), "1 m^2 kg s^-2");
    EXPECT_EQ(std::format("{}", Power{1.0}), "1 m^2 kg s^-3");
    EXPECT_EQ(std::format("{}", Pressure{1.0}), "1 m^-1 kg s^-2");
    EXPECT_EQ(std::format("{}", Frequency{50.0}), "50 s^-1");
    EXPECT_EQ(std::format("{}", Density{1.0}), "1 m^-3 kg");
}

TEST(UnitsFormat, QuantitiesThatShareADimensionPrintIdentically) {
    // Not a defect of the formatter. It is the honest rendering of two things
    // the type system genuinely cannot separate; see src/Units/README.md.
    EXPECT_EQ(std::format("{}", Torque{1.0}), std::format("{}", Energy{1.0}));
    EXPECT_EQ(std::format("{}", Entropy{1.0}), std::format("{}", HeatCapacity{1.0}));
    EXPECT_EQ(std::format("{}", AngularVelocity{1.0}),
              std::format("{}", Frequency{1.0}));
}

TEST(UnitsFormat, DimensionlessCarriesNoSuffix) {
    EXPECT_EQ(std::format("{}", Dimensionless{0.5}), "0.5");
    EXPECT_EQ(std::format("{:.3f}", Length{1.0} / Length{4.0}), "0.250");
}

TEST(UnitsFormat, TheSpecIsForwardedToTheValue) {
    EXPECT_EQ(std::format("{:.3f}", Speed{7800.0}), "7800.000 m s^-1");
    EXPECT_EQ(std::format("{:.2e}", Mass{1.989e30}), "1.99e+30 kg");
    EXPECT_EQ(std::format("{:+.1f}", Length{2.5}), "+2.5 m");
    EXPECT_EQ(std::format("{:.4g}", Energy{1.602176634e-19}),
              "1.602e-19 m^2 kg s^-2");
}

TEST(UnitsFormat, VectorQuantitiesFormatLikeMathVectorsWithAUnit) {
    const Length3 position{Vec3{1.5e11, 0.0, 0.0}};
    const Velocity3 velocity{Vec3{0.0, 29780.0, 0.0}};

    EXPECT_EQ(std::format("{:.2e}", position), "(1.50e+11, 0.00e+00, 0.00e+00) m");
    EXPECT_EQ(std::format("{:.1f}", velocity), "(0.0, 29780.0, 0.0) m s^-1");

    const Quantity<dim::Length, Vec2> planar{Vec2{1.0, 2.0}};
    EXPECT_EQ(std::format("{}", planar), "(1, 2) m");
}

TEST(UnitsFormat, LargeAndNegativeExponentsRenderCorrectly) {
    // Two-digit exponents exercise the integer writer's loop rather than its
    // first iteration, which is the only part of it that could be wrong.
    using Odd = Quantity<dim::Dim<12, -11, 3>>;
    EXPECT_EQ(std::format("{}", Odd{1.0}), "1 m^12 kg^-11 s^3");

    using AllSeven = Quantity<dim::Dim<1, 1, 1, 1, 1, 1, 1>>;
    EXPECT_EQ(std::format("{}", AllSeven{1.0}), "1 m kg s A K mol cd");
}

TEST(UnitsFormat, ConstantsPrintWithTheirDimensions) {
    EXPECT_EQ(std::format("{:.6e}", constants::speedOfLight), "2.997925e+08 m s^-1");
    EXPECT_EQ(std::format("{:.4e}", constants::nominalSolarMassParameter),
              "1.3271e+20 m^3 s^-2");
}
