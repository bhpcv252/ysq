#include <Applications/LunarEclipse/Scenario.hpp>

#include <Math/Quaternion.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Gravity/Kepler.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Thermodynamics/Thermodynamics.hpp>
#include <Units/Acceleration.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>

namespace ysq::lunar_eclipse {

namespace {

using ysq::KeplerStateVector;
using ysq::OrbitalElements;
using ysq::stateVectorFromElements;

}  // namespace

Scenario makeScenario() {
    Scenario scenario;

    // --- Sun ---------------------------------------------------------
    scenario.sun.mass = units::solarMass;
    scenario.sun.radius = units::solarRadius;

    // --- Earth: real mass, radius, oblateness, spin, atmosphere ------
    const double earthMassKg = units::earthMass.value();
    const double earthRadiusM = 6.371e6;

    scenario.earth.mass = Mass{earthMassKg};
    scenario.earth.radius = Length{earthRadiusM};
    scenario.earth.j2 = 1.08263e-3;

    // Real published values, C (polar) and A (equatorial) moments of
    // inertia; C - A is what the J2 term and the gravity-gradient torque
    // both ultimately trace back to.
    const double earthPolarMomentOfInertia = 8.034e37;
    const double earthEquatorialMomentOfInertia = 8.008e37;
    scenario.earth.principalMomentsOfInertia =
        MomentOfInertia3{Vec3{earthEquatorialMomentOfInertia,
                              earthEquatorialMomentOfInertia, earthPolarMomentOfInertia}};

    // Real axial tilt (obliquity), 23.44 degrees from the ecliptic normal
    // (+Z here, since Earth's own orbit is set in the z = 0 plane below).
    // Tilted about the x axis; which azimuth the tilt points at is not a
    // physically distinguished choice (there is no real historical epoch
    // being reproduced here), so this one is as good as any.
    const double obliquity = radians(23.44);
    scenario.earth.orientation = Quat::fromAxisAngle(Vec3::unitX(), obliquity);

    // Real sidereal rotation period; angular momentum = (polar moment) *
    // (spin rate), about Earth's own tilted polar axis.
    const double siderealDay = 23.9344696 * units::hour.value();
    const double spinRate = kTau<double> / siderealDay;
    const Vec3 earthPolarAxis = rotate(scenario.earth.orientation, Vec3::unitZ());
    scenario.earth.angularMomentum =
        AngularMomentum3{earthPolarAxis * (earthPolarMomentOfInertia * spinRate)};

    // Real dry-air numbers: scale height derived the same way
    // Thermodynamics::isothermalScaleHeight documents, surface refractivity
    // and number density at standard conditions.
    const double airSpecificGasConstant = 287.0;
    const double surfaceTemperature = 288.15;
    const double surfaceGravity = 9.80665;
    const double scaleHeight =
        isothermalScaleHeight(SpecificGasConstant{airSpecificGasConstant},
                              Temperature{surfaceTemperature},
                              Acceleration{surfaceGravity})
            .value();
    const double surfaceRefractivity = 2.9e-4;
    scenario.earthAtmosphere =
        RefractiveMedium{earthRadiusM, surfaceRefractivity, scaleHeight};
    scenario.earthSurfaceNumberDensity = 2.6868e25;
    scenario.earthScatteringScaleHeight = scaleHeight;

    // --- Moon: real mass, radius, real elliptical, inclined orbit ----
    scenario.moon.mass = Mass{7.342e22};
    scenario.moon.radius = Length{1.7374e6};

    // --- Jupiter: real mass; a gravitational perturber only ----------
    scenario.jupiter.mass = Mass{1.8982e27};
    scenario.jupiter.radius = Length{6.9911e7};

    // --- Orbits: real elements, an arbitrary (non-historical) epoch --
    const double gmSun = constants::G.value() * units::solarMass.value();
    const double gmEarth = constants::G.value() * earthMassKg;

    const OrbitalElements earthOrbit{
        units::astronomicalUnit.value(), 0.0167086, 0.0, 0.0, 0.0, 0.0};
    const OrbitalElements jupiterOrbit{
        5.2044 * units::astronomicalUnit.value(), 0.0489, radians(1.303), 0.0, 0.0, 0.0};
    // Real eccentricity and the real 5.145 degree inclination to the
    // ecliptic; longitude of ascending node and argument of periapsis are
    // arbitrary (no real historical epoch is being matched), but the shape
    // and tilt of the ellipse are real.
    const OrbitalElements moonOrbit{3.84399e8, 0.0549006, radians(5.145), 0.0, 0.0, 0.0};

    const KeplerStateVector earthState = stateVectorFromElements(earthOrbit, gmSun);
    scenario.earth.position = Length3{earthState.position};
    scenario.earth.momentum = Momentum3{earthState.velocity * earthMassKg};

    const KeplerStateVector jupiterState = stateVectorFromElements(jupiterOrbit, gmSun);
    scenario.jupiter.position = Length3{jupiterState.position};
    scenario.jupiter.momentum =
        Momentum3{jupiterState.velocity * scenario.jupiter.mass.value()};

    const KeplerStateVector moonRelative = stateVectorFromElements(moonOrbit, gmEarth);
    scenario.moon.position = Length3{earthState.position + moonRelative.position};
    scenario.moon.momentum = Momentum3{(earthState.velocity + moonRelative.velocity) *
                                       scenario.moon.mass.value()};

    // The Sun's momentum balances every other body's, so the system's
    // total momentum is exactly zero, the same center-of-mass convention
    // SolarSystem's own scenario uses.
    Vec3 sunMomentum = Vec3::zero();
    sunMomentum -= scenario.earth.momentum.value();
    sunMomentum -= scenario.moon.momentum.value();
    sunMomentum -= scenario.jupiter.momentum.value();
    scenario.sun.momentum = Momentum3{sunMomentum};

    return scenario;
}

std::vector<Body> Scenario::allBodies() const {
    return {sun, earth, moon, jupiter};
}

Vec3f toRenderPosition(const Length3& position) {
    const Vec3 meters = position.value();
    const double scale =
        static_cast<double>(kRenderUnitsPerAu) / units::astronomicalUnit.value();
    return Vec3f{static_cast<float>(meters.x * scale),
                 static_cast<float>(meters.y * scale),
                 static_cast<float>(meters.z * scale)};
}

float toRenderRadius(Length radius) {
    const double scale =
        static_cast<double>(kRenderUnitsPerAu) / units::astronomicalUnit.value();
    return static_cast<float>(radius.value() * scale);
}

}  // namespace ysq::lunar_eclipse
