#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Optics/RadiationPressure.hpp>
#include <Units/Constants.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Luminosity.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Area;
using ysq::Force3;
using ysq::Irradiance;
using ysq::Length;
using ysq::Length3;
using ysq::RadiantPower;
using ysq::Vec3;

}  // namespace

TEST(PhysicsOpticsRadiationPressure, IrradianceFromTheSunAtOneAUMatchesTheSolarConstant) {
    // The real solar luminosity and one real astronomical unit should
    // reproduce the measured solar constant, about 1361 W/m^2, to within
    // the couple of percent real solar variability allows for.
    const RadiantPower solarLuminosity{3.828e26};
    const Length oneAU{1.495978707e11};

    const Irradiance irradiance = ysq::irradianceFromPointSource(solarLuminosity, oneAU);
    EXPECT_NEAR(irradiance.value(), 1361.0, 1361.0 * 0.02);
}

TEST(PhysicsOpticsRadiationPressure, IrradianceFollowsTheInverseSquareLaw) {
    const RadiantPower luminosity{1.0e26};
    const Irradiance atOneUnit = ysq::irradianceFromPointSource(luminosity, Length{1.0});
    const Irradiance atTwoUnits = ysq::irradianceFromPointSource(luminosity, Length{2.0});

    EXPECT_NEAR(atOneUnit.value() / atTwoUnits.value(), 4.0, 1e-9);
}

TEST(PhysicsOpticsRadiationPressure, ForcePushesAwayFromTheSource) {
    const Irradiance irradiance{1000.0};
    const Area area{1.0};
    const Vec3 direction{1.0, 0.0, 0.0};

    const Force3 force = ysq::radiationPressureForce(irradiance, area, 1.0, direction);
    EXPECT_GT(force.value().x, 0.0);
    EXPECT_NEAR(force.value().y, 0.0, 1e-15);
    EXPECT_NEAR(force.value().z, 0.0, 1e-15);
}

TEST(PhysicsOpticsRadiationPressure, ForceMagnitudeMatchesTheClosedForm) {
    const Irradiance irradiance{1361.0};
    const Area area{2.5};
    const double cr = 1.3;
    const Vec3 direction{0.0, 1.0, 0.0};

    const Force3 force = ysq::radiationPressureForce(irradiance, area, cr, direction);
    const double expected =
        irradiance.value() * area.value() * cr / ysq::constants::speedOfLight.value();

    EXPECT_NEAR(ysq::length(force).value(), expected, expected * 1e-9);
}

TEST(PhysicsOpticsRadiationPressure,
     APerfectReflectorFeelsTwiceTheForceOfAPerfectAbsorber) {
    const Irradiance irradiance{1361.0};
    const Area area{1.0};
    const Vec3 direction{1.0, 0.0, 0.0};

    const Force3 absorber = ysq::radiationPressureForce(irradiance, area, 1.0, direction);
    const Force3 reflector =
        ysq::radiationPressureForce(irradiance, area, 2.0, direction);

    EXPECT_NEAR(ysq::length(reflector).value(), 2.0 * ysq::length(absorber).value(),
                1e-12);
}

TEST(PhysicsOpticsRadiationPressure, CannonballModelMatchesTheTwoPieceComposition) {
    const RadiantPower luminosity{3.828e26};
    const Length3 sourcePosition{Vec3{0.0, 0.0, 0.0}};
    const Length3 bodyPosition{Vec3{1.495978707e11, 0.0, 0.0}};
    const Length bodyRadius{6.378e6};
    const double cr = 1.2;

    const Force3 viaCannonball = ysq::radiationPressureForce(
        luminosity, sourcePosition, bodyPosition, bodyRadius, cr);

    const Length distance = ysq::length(bodyPosition - sourcePosition);
    const Irradiance irradiance = ysq::irradianceFromPointSource(luminosity, distance);
    const Area crossSection{ysq::kPi<double> * bodyRadius.value() * bodyRadius.value()};
    const Force3 viaPieces =
        ysq::radiationPressureForce(irradiance, crossSection, cr, Vec3{1.0, 0.0, 0.0});

    EXPECT_QUANTITY_VEC_NEAR(viaCannonball, viaPieces, ysq::Force{1e-20});
}

TEST(PhysicsOpticsRadiationPressure,
     CannonballModelIsZeroWhenBodyCoincidesWithTheSource) {
    const RadiantPower luminosity{3.828e26};
    const Length3 position{Vec3{5.0, 5.0, 5.0}};

    const Force3 force =
        ysq::radiationPressureForce(luminosity, position, position, Length{1.0}, 1.0);
    EXPECT_QUANTITY_VEC_NEAR(force, Force3::zero(), ysq::Force{1e-30});
}
