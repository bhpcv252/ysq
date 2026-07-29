#include <Applications/SolarSystem/Scenario.hpp>

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

/// main.cpp indexes bodies[i + 1] against scenario.planets[i] for both
/// rendering and trails; nothing else pins that contract down. This is the
/// cheap, isolated check for it, separate from tests/e2e/solar_system.cpp's
/// much heavier integrated conservation run.

namespace {

using namespace ysq::solar_system;

}  // namespace

TEST(SolarSystemScenario, AllBodiesIsSunFirstThenPlanetsInOrder) {
    const Scenario scenario = makeScenario();
    const std::vector<ysq::Body> bodies = scenario.allBodies();

    ASSERT_EQ(bodies.size(), scenario.planets.size() + 1);
    EXPECT_DOUBLE_EQ(bodies[0].mass.value(), scenario.sun.mass.value());

    for (std::size_t i = 0; i < scenario.planets.size(); ++i) {
        EXPECT_DOUBLE_EQ(bodies[i + 1].mass.value(),
                         scenario.planets[i].body.mass.value())
            << "body " << (i + 1) << " should be " << scenario.planets[i].name;
    }
}

TEST(SolarSystemScenario, PlanetsAreOrderedByIncreasingSemiMajorAxis) {
    const Scenario scenario = makeScenario();
    for (std::size_t i = 1; i < scenario.planets.size(); ++i) {
        const double previous = length(scenario.planets[i - 1].body.position.value());
        const double current = length(scenario.planets[i].body.position.value());
        EXPECT_LT(previous, current)
            << scenario.planets[i - 1].name << " should orbit closer than "
            << scenario.planets[i].name;
    }
}

TEST(SolarSystemScenario, TotalMomentumIsZeroByConstruction) {
    const Scenario scenario = makeScenario();
    const std::vector<ysq::Body> bodies = scenario.allBodies();

    ysq::Vec3 total = ysq::Vec3::zero();
    for (const ysq::Body& body : bodies) {
        total += body.momentum.value();
    }

    // Characteristic scale for a quantity that is exactly zero by
    // construction: Earth's own orbital momentum.
    const double scale = length(scenario.planets[2].body.momentum.value());
    EXPECT_LT(length(total), scale * 1e-9);
}
