#include <Applications/SolarSystem/Scenario.hpp>

#include <Math/Vector3.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>

#include <cmath>
#include <utility>

namespace ysq::solar_system {

namespace {

struct PlanetData {
    std::string name;
    double massKg;
    double semiMajorAxisAu;
    Vec3f color;
    float renderRadius;
};

// Masses (kg) and semi-major axes (AU) are standard published approximate
// values. Orbits are circular at that radius rather than each planet's true
// (slightly eccentric) one; see Scenario's own doc comment.
const std::vector<PlanetData>& planetTable() {
    static const std::vector<PlanetData> table = {
        {"Mercury", 3.3011e23, 0.387, Vec3f{0.62f, 0.59f, 0.55f}, 0.12f},
        {"Venus", 4.8675e24, 0.723, Vec3f{0.92f, 0.77f, 0.42f}, 0.18f},
        {"Earth", units::earthMass.value(), 1.000, Vec3f{0.25f, 0.45f, 0.85f}, 0.19f},
        {"Mars", 6.4171e23, 1.524, Vec3f{0.80f, 0.35f, 0.20f}, 0.15f},
        {"Jupiter", 1.8982e27, 5.203, Vec3f{0.82f, 0.68f, 0.52f}, 0.50f},
    };
    return table;
}

}  // namespace

Scenario makeScenario() {
    Scenario scenario;
    scenario.sun.mass = units::solarMass;
    scenario.sunColor = Vec3f{1.0f, 0.85f, 0.3f};

    const double gmSun = constants::G.value() * units::solarMass.value();

    Vec3 sunMomentum = Vec3::zero();

    for (const PlanetData& data : planetTable()) {
        const double radius = data.semiMajorAxisAu * units::astronomicalUnit.value();
        const double speed = std::sqrt(gmSun / radius);

        Planet planet;
        planet.name = data.name;
        planet.body.mass = Mass{data.massKg};
        planet.body.position = Length3{Vec3{radius, 0.0, 0.0}};
        planet.body.momentum = Momentum3{Vec3{0.0, speed, 0.0} * data.massKg};
        planet.color = data.color;
        planet.renderRadius = data.renderRadius;

        sunMomentum -= planet.body.momentum.value();
        scenario.planets.push_back(std::move(planet));
    }

    scenario.sun.momentum = Momentum3{sunMomentum};
    return scenario;
}

std::vector<Body> Scenario::allBodies() const {
    std::vector<Body> bodies;
    bodies.reserve(planets.size() + 1);
    bodies.push_back(sun);
    for (const Planet& planet : planets) {
        bodies.push_back(planet.body);
    }
    return bodies;
}

Vec3f toRenderPosition(const Length3& position) {
    const Vec3 meters = position.value();
    const double scale =
        static_cast<double>(kRenderUnitsPerAu) / units::astronomicalUnit.value();
    return Vec3f{static_cast<float>(meters.x * scale),
                 static_cast<float>(meters.y * scale),
                 static_cast<float>(meters.z * scale)};
}

}  // namespace ysq::solar_system
