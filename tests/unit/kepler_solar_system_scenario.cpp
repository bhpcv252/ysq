#include <Applications/KeplerSolarSystem/Scenario.hpp>

#include <Units/Unit.hpp>

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

/// KeplerSolarSystem's own scenario invariants: every body's elements
/// retained (not collapsed to one fixed state, unlike
/// Applications::SolarSystem), the 5 dwarf planets present with real
/// values, both procedural populations within their real ranges, and every
/// ring naming a real planet. tests/e2e/kepler_solar_system.cpp covers the
/// heavier "does propagation actually behave correctly over time" checks;
/// this is the cheap, isolated "did the data load and compose correctly"
/// check.

namespace {

using namespace ysq::kepler_solar_system;

const ysq::applications::KeplerCatalogBody& findByName(const Scenario& scenario,
                                                   const std::string& name) {
    for (const auto& body : scenario.bodies) {
        if (body.name == name) {
            return body;
        }
    }
    ADD_FAILURE() << "no body named '" << name << "'";
    return scenario.bodies.front();
}

}  // namespace

TEST(KeplerSolarSystemScenario, LoadsTheRealDataFileSuccessfully) {
    std::string error;
    const std::optional<Scenario> scenario = makeScenario(&error);
    ASSERT_TRUE(scenario.has_value()) << error;
    EXPECT_GT(scenario->bodies.size(), 100u)
        << "expected the Sun, 8 planets, every real moon, and the 5 dwarf planets";
}

TEST(KeplerSolarSystemScenario, SunIsTheOneRootAndSitsFirst) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());
    EXPECT_TRUE(scenario->bodies.front().parent.empty()) << "index 0 must be the one root";
    EXPECT_EQ(scenario->bodies.front().name, "Sun");
    EXPECT_EQ(scenario->bodies.front().parentIndex, -1);
    EXPECT_FALSE(scenario->bodies.front().elements.has_value());
}

TEST(KeplerSolarSystemScenario, EveryNonRootBodyHasElementsAndAValidParentIndex) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    for (const auto& body : scenario->bodies) {
        if (body.parent.empty()) {
            continue;
        }
        EXPECT_TRUE(body.elements.has_value()) << body.name;
        ASSERT_GE(body.parentIndex, 0) << body.name;
        EXPECT_LT(static_cast<std::size_t>(body.parentIndex), scenario->bodies.size())
            << body.name;
        EXPECT_EQ(scenario->bodies[static_cast<std::size_t>(body.parentIndex)].name,
                 body.parent)
            << body.name;
    }
}

TEST(KeplerSolarSystemScenario, EveryNonRootBodysParentExistsInTheScenario) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    std::unordered_set<std::string> names;
    for (const auto& body : scenario->bodies) {
        names.insert(body.name);
    }
    for (const auto& body : scenario->bodies) {
        if (body.parent.empty()) {
            continue;
        }
        EXPECT_TRUE(names.contains(body.parent))
            << body.name << " claims parent '" << body.parent << "'";
    }
}

TEST(KeplerSolarSystemScenario, TheFiveDwarfPlanetsArePresentWithRealSemiMajorAxes) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    const double auMeters = ysq::units::astronomicalUnit.value();
    // Real values, not periapsis/apoapsis-swung ones: semiMajorAxis is the
    // orbit's own shape, independent of where each body happens to sit on
    // it right now, so this can check against the real semi-major axis
    // directly rather than needing a wide "instantaneous distance" margin.
    const std::vector<std::pair<std::string, double>> expectedAu{
        {"Ceres", 2.7656}, {"Pluto", 39.59}, {"Haumea", 43.06},
        {"Makemake", 45.57}, {"Eris", 67.93}};

    for (const auto& [name, semiMajorAxisAu] : expectedAu) {
        const ysq::applications::KeplerCatalogBody& body = findByName(*scenario, name);
        EXPECT_EQ(body.parent, "Sun") << name;
        ASSERT_TRUE(body.elements.has_value()) << name;
        EXPECT_NEAR(body.elements->semiMajorAxis, semiMajorAxisAu * auMeters,
                   semiMajorAxisAu * auMeters * 1e-3)
            << name;
    }
}

TEST(KeplerSolarSystemScenario, SunParentedBodiesPrecessAndMoonsDoNot) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    const ysq::applications::KeplerCatalogBody& mercury = findByName(*scenario, "Mercury");
    ASSERT_TRUE(mercury.elements.has_value());
    EXPECT_GT(mercury.elements->precessionRatePerSecond, 0.0)
        << "a Sun-parented body's own real GR precession rate must be positive, not left "
           "at the default zero a fixed ellipse would have";

    const ysq::applications::KeplerCatalogBody& moon = findByName(*scenario, "Moon");
    ASSERT_TRUE(moon.elements.has_value());
    EXPECT_DOUBLE_EQ(moon.elements->precessionRatePerSecond, 0.0)
        << "a moon's own primary is its planet, not the Sun -- solar GR precession does "
           "not apply to it here";
}

TEST(KeplerSolarSystemScenario, AsteroidBeltParticlesAreSunParentedAndWithinTheRealRange) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());
    ASSERT_FALSE(scenario->asteroidBelt.empty());

    const double auMeters = ysq::units::astronomicalUnit.value();
    for (const auto& particle : scenario->asteroidBelt) {
        EXPECT_EQ(particle.parentIndex, 0);
        EXPECT_GE(particle.elements.semiMajorAxis, 2.1 * auMeters);
        EXPECT_LE(particle.elements.semiMajorAxis, 3.3 * auMeters);
    }
}

TEST(KeplerSolarSystemScenario, KuiperBeltParticlesAreSunParentedAndWithinTheRealRange) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());
    ASSERT_FALSE(scenario->kuiperBelt.empty());

    const double auMeters = ysq::units::astronomicalUnit.value();
    for (const auto& particle : scenario->kuiperBelt) {
        EXPECT_EQ(particle.parentIndex, 0);
        EXPECT_GE(particle.elements.semiMajorAxis, 30.0 * auMeters);
        EXPECT_LE(particle.elements.semiMajorAxis, 50.0 * auMeters);
    }
}

TEST(KeplerSolarSystemScenario, EveryRingNamesARealPlanetAndItsParticlesOrbitThatPlanet) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());
    ASSERT_FALSE(scenario->rings.empty());

    const double auMeters = ysq::units::astronomicalUnit.value();

    std::unordered_set<std::string> names;
    for (const auto& body : scenario->bodies) {
        names.insert(body.name);
    }

    std::unordered_set<std::string> ringedPlanets;
    for (const RingPopulation& ring : scenario->rings) {
        EXPECT_TRUE(names.contains(ring.parent)) << ring.parent;
        ASSERT_FALSE(ring.particles.empty()) << ring.parent;

        const int parentIndex = ring.particles.front().parentIndex;
        for (const auto& particle : ring.particles) {
            // Every particle in one ring orbits the same real planet, not
            // the Sun: a real ring radius is a tiny fraction of an AU,
            // well under any real planet's own distance from the Sun.
            EXPECT_EQ(particle.parentIndex, parentIndex) << ring.parent;
            EXPECT_GT(particle.elements.semiMajorAxis, 0.0) << ring.parent;
            EXPECT_LT(particle.elements.semiMajorAxis, 0.01 * auMeters) << ring.parent;
            EXPECT_GT(particle.realRadiusMeters, 0.0) << ring.parent;
        }
        EXPECT_EQ(scenario->bodies[static_cast<std::size_t>(parentIndex)].name, ring.parent);
        EXPECT_NE(parentIndex, 0) << ring.parent << " must orbit its own planet, not the Sun";
        ringedPlanets.insert(ring.parent);
    }
    for (const char* planet : {"Jupiter", "Saturn", "Uranus", "Neptune"}) {
        EXPECT_TRUE(ringedPlanets.contains(planet)) << planet << " should have a ring";
    }
}
