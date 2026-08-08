#include <Applications/SolarSystem/Scenario.hpp>

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

/// main.cpp indexes bodies[i] against scenario.bodies[i] directly for
/// rendering, trails and the POV/Focus lists; nothing else pins that
/// contract down. This is the cheap, isolated check for it, separate from
/// tests/e2e/solar_system.cpp's much heavier integrated conservation run.

namespace {

using namespace ysq::solar_system;

}  // namespace

TEST(SolarSystemScenario, LoadsTheRealDataFileSuccessfully) {
    std::string error;
    const std::optional<Scenario> scenario = makeScenario(&error);
    ASSERT_TRUE(scenario.has_value()) << error;
    EXPECT_GT(scenario->bodies.size(), 100u)
        << "expected the Sun, 8 planets and every real moon (175 total)";
}

TEST(SolarSystemScenario, AllBodiesIsSunFirstThenEveryOtherBodyInScenarioOrder) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());
    const std::vector<ysq::Body> bodies = scenario->allBodies();

    ASSERT_EQ(bodies.size(), scenario->bodies.size());
    EXPECT_TRUE(scenario->bodies.front().parent.empty()) << "index 0 must be the one root";
    EXPECT_EQ(scenario->bodies.front().name, "Sun");

    for (std::size_t i = 0; i < scenario->bodies.size(); ++i) {
        EXPECT_DOUBLE_EQ(bodies[i].mass.value(), scenario->bodies[i].body.mass.value())
            << "body " << i << " (" << scenario->bodies[i].name << ")";
    }
}

TEST(SolarSystemScenario, PlanetsAreOrderedByIncreasingSemiMajorAxis) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    const std::array<std::string, 8> planetsInOrder{"Mercury", "Venus",  "Earth",  "Mars",
                                                     "Jupiter", "Saturn", "Uranus", "Neptune"};

    const auto findByName = [&](const std::string& name) -> const ysq::applications::CatalogBody& {
        for (const auto& body : scenario->bodies) {
            if (body.name == name) {
                return body;
            }
        }
        ADD_FAILURE() << "no body named '" << name << "'";
        return scenario->bodies.front();
    };

    double previous = 0.0;
    for (const std::string& name : planetsInOrder) {
        const ysq::applications::CatalogBody& planet = findByName(name);
        EXPECT_EQ(planet.parent, "Sun") << name;
        const double distance = length(planet.body.position.value());
        EXPECT_GT(distance, previous) << name << " should orbit farther out than the previous planet";
        previous = distance;
    }
}

TEST(SolarSystemScenario, TotalMomentumIsZeroByConstruction) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());
    const std::vector<ysq::Body> bodies = scenario->allBodies();

    ysq::Vec3 total = ysq::Vec3::zero();
    double earthMomentumScale = 0.0;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        total += bodies[i].momentum.value();
        if (scenario->bodies[i].name == "Earth") {
            earthMomentumScale = length(bodies[i].momentum.value());
        }
    }

    ASSERT_GT(earthMomentumScale, 0.0);
    // Characteristic scale for a quantity that is exactly zero by
    // construction: Earth's own orbital momentum.
    EXPECT_LT(length(total), earthMomentumScale * 1e-9);
}

TEST(SolarSystemScenario, EveryNonRootBodysParentExistsInTheScenario) {
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
