#include <Math/Calculus.hpp>
#include <Math/Scalar.hpp>
#include <Physics/Thermodynamics/StatisticalMechanics.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Velocity.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ysq::Mass;
using ysq::Speed;
using ysq::Temperature;

constexpr double kNitrogenMoleculeMass = 4.6518e-26;  // kg, N2
constexpr double kRoomTemperature = 300.0;            // K

}  // namespace

TEST(PhysicsThermodynamicsStatisticalMechanics, DensityIntegratesToOne) {
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};

    const auto density = [&](double v) {
        return ysq::maxwellBoltzmannSpeedDensity(Speed{v}, mass, temperature);
    };

    // Integrate well past where the density has become negligible.
    const double upperBound =
        20.0 * ysq::maxwellBoltzmannScale(mass, temperature).value();
    const double integral = ysq::simpson(density, 0.0, upperBound, 20000);

    EXPECT_NEAR(integral, 1.0, 1e-6);
}

TEST(PhysicsThermodynamicsStatisticalMechanics,
     CdfAtZeroIsZeroAndApproachesOneAtLargeSpeed) {
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};

    EXPECT_NEAR(ysq::maxwellBoltzmannSpeedCdf(Speed{0.0}, mass, temperature), 0.0, 1e-12);

    const double largeSpeed =
        20.0 * ysq::maxwellBoltzmannScale(mass, temperature).value();
    EXPECT_NEAR(ysq::maxwellBoltzmannSpeedCdf(Speed{largeSpeed}, mass, temperature), 1.0,
                1e-9);
}

TEST(PhysicsThermodynamicsStatisticalMechanics, CdfMatchesTheIntegralOfTheDensity) {
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};
    const double v = 400.0;

    const auto density = [&](double speed) {
        return ysq::maxwellBoltzmannSpeedDensity(Speed{speed}, mass, temperature);
    };
    const double integrated = ysq::simpson(density, 0.0, v, 20000);
    const double closedForm = ysq::maxwellBoltzmannSpeedCdf(Speed{v}, mass, temperature);

    EXPECT_NEAR(integrated, closedForm, 1e-6);
}

TEST(PhysicsThermodynamicsStatisticalMechanics, MostProbableSpeedIsWhereDensityPeaks) {
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};
    const double vp = ysq::maxwellBoltzmannMostProbableSpeed(mass, temperature).value();

    const double atPeak = ysq::maxwellBoltzmannSpeedDensity(Speed{vp}, mass, temperature);
    const double justBelow =
        ysq::maxwellBoltzmannSpeedDensity(Speed{vp - 5.0}, mass, temperature);
    const double justAbove =
        ysq::maxwellBoltzmannSpeedDensity(Speed{vp + 5.0}, mass, temperature);

    EXPECT_GT(atPeak, justBelow);
    EXPECT_GT(atPeak, justAbove);
}

TEST(PhysicsThermodynamicsStatisticalMechanics,
     MeanSpeedMatchesDirectNumericalIntegration) {
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};

    const auto integrand = [&](double v) {
        return v * ysq::maxwellBoltzmannSpeedDensity(Speed{v}, mass, temperature);
    };
    const double upperBound =
        20.0 * ysq::maxwellBoltzmannScale(mass, temperature).value();
    const double numericMean = ysq::simpson(integrand, 0.0, upperBound, 20000);

    EXPECT_NEAR(ysq::maxwellBoltzmannMeanSpeed(mass, temperature).value(), numericMean,
                1e-3);
}

TEST(PhysicsThermodynamicsStatisticalMechanics,
     RmsSpeedMatchesDirectNumericalIntegration) {
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};

    const auto integrand = [&](double v) {
        return v * v * ysq::maxwellBoltzmannSpeedDensity(Speed{v}, mass, temperature);
    };
    const double upperBound =
        20.0 * ysq::maxwellBoltzmannScale(mass, temperature).value();
    const double numericMeanSquare = ysq::simpson(integrand, 0.0, upperBound, 20000);

    EXPECT_NEAR(ysq::maxwellBoltzmannRmsSpeed(mass, temperature).value(),
                std::sqrt(numericMeanSquare), 1e-3);
}

TEST(PhysicsThermodynamicsStatisticalMechanics,
     TheThreeCharacteristicSpeedsHaveTheirTextbookRatios) {
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};

    const double vp = ysq::maxwellBoltzmannMostProbableSpeed(mass, temperature).value();
    const double vMean = ysq::maxwellBoltzmannMeanSpeed(mass, temperature).value();
    const double vRms = ysq::maxwellBoltzmannRmsSpeed(mass, temperature).value();

    // v_p : v_mean : v_rms = sqrt(2) : sqrt(8/pi) : sqrt(3), the standard
    // textbook ratios among the three characteristic speeds.
    EXPECT_NEAR(vMean / vp, std::sqrt(8.0 / ysq::kPi<double>) / std::sqrt(2.0), 1e-9);
    EXPECT_NEAR(vRms / vp, std::sqrt(3.0) / std::sqrt(2.0), 1e-9);
}

TEST(PhysicsThermodynamicsStatisticalMechanics,
     NitrogenAtRoomTemperatureMatchesTheKnownRmsSpeed) {
    // A well-known real-world number: N2 at 300 K has an RMS speed of
    // about 517 m/s.
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};

    const double vRms = ysq::maxwellBoltzmannRmsSpeed(mass, temperature).value();
    EXPECT_NEAR(vRms, 517.0, 5.0);
}

TEST(PhysicsThermodynamicsStatisticalMechanics, MomentTwoEqualsRmsSpeedSquared) {
    const Mass mass{kNitrogenMoleculeMass};
    const Temperature temperature{kRoomTemperature};

    const double momentTwo = ysq::maxwellBoltzmannMoment(2, mass, temperature);
    const double vRms = ysq::maxwellBoltzmannRmsSpeed(mass, temperature).value();

    EXPECT_NEAR(momentTwo, vRms * vRms, 1e-6);
}
