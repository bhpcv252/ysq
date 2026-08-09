#include <Math/Scalar.hpp>
#include <Physics/Thermodynamics/RadiativeTransfer.hpp>
#include <Physics/Thermodynamics/Thermodynamics.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Temperature.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Area;
using ysq::Length;
using ysq::Power;
using ysq::Temperature;

}  // namespace

TEST(PhysicsThermodynamicsRadiativeTransfer,
     CoaxialDiskViewFactorApproachesOneWhenDisksCoincide) {
    // Equal radii, separated by a distance tiny compared to that radius:
    // nearly all of one disk's radiation reaches the other.
    const double viewFactor =
        ysq::coaxialDiskViewFactor(Length{1.0}, Length{1.0}, Length{1e-6});
    EXPECT_NEAR(viewFactor, 1.0, 1e-3);
}

TEST(PhysicsThermodynamicsRadiativeTransfer,
     CoaxialDiskViewFactorVanishesAtLargeSeparation) {
    const double viewFactor =
        ysq::coaxialDiskViewFactor(Length{1.0}, Length{1.0}, Length{1000.0});
    EXPECT_NEAR(viewFactor, 0.0, 1e-6);
}

TEST(PhysicsThermodynamicsRadiativeTransfer, CoaxialDiskViewFactorStaysWithinZeroAndOne) {
    for (double distance : {0.1, 0.5, 1.0, 2.0, 5.0, 20.0}) {
        const double viewFactor =
            ysq::coaxialDiskViewFactor(Length{1.0}, Length{2.0}, Length{distance});
        EXPECT_GE(viewFactor, 0.0);
        EXPECT_LE(viewFactor, 1.0);
    }
}

TEST(PhysicsThermodynamicsRadiativeTransfer, CoaxialDiskViewFactorSatisfiesReciprocity) {
    const Length r1{1.0};
    const Length r2{3.0};
    const Length distance{4.0};

    const double f12 = ysq::coaxialDiskViewFactor(r1, r2, distance);
    const double f21 = ysq::coaxialDiskViewFactor(r2, r1, distance);

    const Area a1{ysq::kPi<double> * r1.value() * r1.value()};
    const Area a2{ysq::kPi<double> * r2.value() * r2.value()};

    // The reciprocity relation A1 F12 = A2 F21 must hold for any valid
    // pair of view factors, independent of the closed form used to
    // compute either one.
    EXPECT_NEAR((a1 * f12).value(), (a2 * f21).value(), 1e-9);
}

TEST(PhysicsThermodynamicsRadiativeTransfer, NetExchangeIsZeroAtThermalEquilibrium) {
    const Power exchange =
        ysq::netRadiativeExchange(Area{1.0}, 0.5, Temperature{300.0}, Temperature{300.0});
    EXPECT_NEAR(exchange.value(), 0.0, 1e-9);
}

TEST(PhysicsThermodynamicsRadiativeTransfer, NetExchangeFlowsFromHotToCold) {
    const Power exchange =
        ysq::netRadiativeExchange(Area{1.0}, 1.0, Temperature{400.0}, Temperature{300.0});
    EXPECT_GT(exchange.value(), 0.0);
}

TEST(PhysicsThermodynamicsRadiativeTransfer,
     NetExchangeIsAntisymmetricUnderSwappingTemperatures) {
    const Power forward =
        ysq::netRadiativeExchange(Area{2.0}, 0.7, Temperature{500.0}, Temperature{250.0});
    const Power backward =
        ysq::netRadiativeExchange(Area{2.0}, 0.7, Temperature{250.0}, Temperature{500.0});
    EXPECT_NEAR(forward.value(), -backward.value(), 1e-9);
}

TEST(PhysicsThermodynamicsRadiativeTransfer,
     NetExchangeReducesToBlackBodyLuminosityIntoFreeSpace) {
    // View factor 1 and a sink at absolute zero is exactly "radiating into
    // free space", Thermodynamics.hpp's own Stefan-Boltzmann law.
    const Length radius{2.0};
    const Temperature temperature{5000.0};
    const Area surfaceArea{4.0 * ysq::kPi<double> * radius.value() * radius.value()};

    const Power viaExchange =
        ysq::netRadiativeExchange(surfaceArea, 1.0, temperature, Temperature{0.0});
    const Power viaBlackBody = ysq::blackBodyLuminosity(radius, temperature);

    EXPECT_NEAR(viaExchange.value(), viaBlackBody.value(), viaBlackBody.value() * 1e-9);
}
