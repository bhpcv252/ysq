#include <Physics/Optics/RayleighScattering.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

TEST(OpticsRayleighScattering, CrossSectionOfAirAtGreenIsTheRealOrderOfMagnitude) {
    // Real air at STP: n about 1.000293, Loschmidt number N about
    // 2.6868e25 per cubic metre. The commonly cited cross-section at 532 nm
    // is of order 5e-31 m^2.
    const double crossSection = ysq::rayleighCrossSection(532.0e-9, 1.000293, 2.6868e25);

    EXPECT_GT(crossSection, 3.0e-31);
    EXPECT_LT(crossSection, 7.0e-31);
}

TEST(OpticsRayleighScattering, CrossSectionFollowsTheInverseFourthPowerOfWavelength) {
    const double n = 1.000293;
    const double numberDensity = 2.6868e25;
    const double blue = ysq::rayleighCrossSection(450.0e-9, n, numberDensity);
    const double red = ysq::rayleighCrossSection(650.0e-9, n, numberDensity);

    // Blue scatters far more than red over the same path: the reason a
    // grazing atmospheric path reddens whatever light gets through.
    EXPECT_GT(blue, red);

    const double expectedRatio = std::pow(650.0 / 450.0, 4.0);
    EXPECT_NEAR(blue / red, expectedRatio, expectedRatio * 1e-9);
}

TEST(OpticsRayleighScattering, CrossSectionVanishesForAVacuum) {
    // n = 1 exactly: nothing to scatter off.
    EXPECT_NEAR(ysq::rayleighCrossSection(500.0e-9, 1.0, 2.6868e25), 0.0, 1e-40);
}

TEST(OpticsRayleighScattering, NumberDensityIsSurfaceValueAtZeroAltitude) {
    EXPECT_NEAR(ysq::exponentialNumberDensity(6.371e6, 2.6868e25, 6.371e6, 8434.5),
                2.6868e25, 2.6868e25 * 1e-9);
}

TEST(OpticsRayleighScattering, NumberDensityFallsByOneOverEPerScaleHeight) {
    const double radius = 6.371e6;
    const double scaleHeight = 8434.5;
    const double surface = 2.6868e25;

    const double atOneScaleHeight =
        ysq::exponentialNumberDensity(radius + scaleHeight, surface, radius, scaleHeight);
    EXPECT_NEAR(atOneScaleHeight, surface / std::exp(1.0), surface * 1e-9);
}

TEST(OpticsRayleighScattering,
     TransmissionIsOneAtZeroOpticalDepthAndDecreasesMonotonically) {
    EXPECT_NEAR(ysq::transmission(0.0), 1.0, 1e-15);

    const double shallow = ysq::transmission(0.1);
    const double deep = ysq::transmission(2.0);
    EXPECT_GT(shallow, deep);
    EXPECT_GT(deep, 0.0);
    EXPECT_LT(shallow, 1.0);
}

}  // namespace
