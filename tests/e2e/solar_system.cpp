#include <Applications/SolarSystem/Scenario.hpp>

#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Statistics.hpp>
#include <Math/Vector3.hpp>

#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Physics/Mechanics/Hermite.hpp>

#include <Units/Length.hpp>
#include <Units/Mass.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

/// solar-system's own scenario (the real Sun, 8 planets, and every real
/// moon, zero-momentum frame), integrated headless and checked against the
/// invariants a correct Newtonian N-body integration must preserve.
///
/// tests/integration/orbit_stability.cpp and nbody_energy.cpp already cover
/// Gravity + the symplectic integrator generically; this instead asks
/// whether the specific data Scenario.cpp ships conserves them, so a wrong
/// mass, a unit slip, or a momentum-frame mistake there shows up as a test
/// failure rather than only looking wrong on screen.

namespace {

using namespace ysq::solar_system;

double totalEnergy(const std::vector<ysq::Body>& bodies, ysq::Length softening) {
    double kinetic = 0.0;
    for (const ysq::Body& body : bodies) {
        const double speed = length(body.velocity().value());
        kinetic += 0.5 * body.mass.value() * speed * speed;
    }
    return kinetic + ysq::newtonianPotentialEnergy(bodies, softening).value();
}

ysq::Vec3 totalMomentum(const std::vector<ysq::Body>& bodies) {
    ysq::Vec3 total = ysq::Vec3::zero();
    for (const ysq::Body& body : bodies) {
        total += body.momentum.value();
    }
    return total;
}

}  // namespace

TEST(SolarSystemE2E, TheSunAndEightPlanetsConserveEnergyAndMomentumOverTwoJupiterOrbits) {
    // The Sun + 8 planets specifically, not the full ~175-body catalog:
    // this is a long run (until 2 Jupiter orbits) at a step size set by
    // Mercury's own period, and the full catalog's fastest moon (under a
    // third of a day) and slowest irregular moon (tens of years) would
    // force either a step size or a total step count neither this test's
    // own runtime budget nor a meaningful conservation claim about
    // gravitationally negligible captured moons affords. See
    // AllBodiesRemainFiniteOverAShortRun below for the full-catalog check.
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    std::vector<ysq::Body> bodies;
    bodies.push_back(scenario->bodies.front().body);  // the Sun
    for (const auto& catalogBody : scenario->bodies) {
        if (catalogBody.parent == "Sun") {
            bodies.push_back(catalogBody.body);
        }
    }
    ASSERT_EQ(bodies.size(), 9u) << "expected the Sun plus exactly 8 planets";

    const ysq::Length softening{1.0e8};

    const double gmSun = ysq::constants::G.value() * ysq::units::solarMass.value();
    const auto orbitalPeriod = [&](const ysq::Body& planet) {
        const double radius = length(planet.position.value());
        return 2.0 * ysq::kPi<double> * std::sqrt(radius * radius * radius / gmSun);
    };

    // bodies[0] is the Sun; Mercury (index 1) is the fastest-moving body and
    // sets the step size an order-2 symplectic method needs to stay
    // accurate, Jupiter (index 5) is the slowest of the inner five and sets
    // how long a run has to be to say anything about the outer orbits (the
    // same span the original 6-body version of this test used, before the
    // scenario grew to the full real system).
    const double mercuryPeriod = orbitalPeriod(bodies[1]);
    const double jupiterPeriod = orbitalPeriod(bodies[5]);

    const double step = mercuryPeriod / 2000.0;
    const double untilTime = 2.0 * jupiterPeriod;
    const int totalSteps = static_cast<int>(untilTime / step);

    const double initialEnergy = totalEnergy(bodies, softening);
    const ysq::Vec3 initialMomentum = totalMomentum(bodies);
    // Characteristic momentum scale for a quantity that is exactly zero by
    // construction (the Sun's momentum is set to cancel the planets'):
    // Earth's own orbital momentum.
    const double momentumScale = length(bodies[3].momentum.value());

    ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
    ysq::RunningStatistics<double> energyDeviation;
    ysq::RunningStatistics<double> momentumDeviation;

    for (int i = 0; i < totalSteps; ++i) {
        const ysq::NewtonianField field(bodies, softening);
        const ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                                     ysq::velocitiesOf(bodies)};
        ysq::PhaseState<ysq::NBodyState> next;
        stepper.step(field, static_cast<double>(i) * step, state, step, next);
        ysq::applyState(bodies, next.position, next.velocity);

        const double energy = totalEnergy(bodies, softening);
        energyDeviation.add(std::abs((energy - initialEnergy) / initialEnergy));

        const ysq::Vec3 momentum = totalMomentum(bodies);
        momentumDeviation.add(length(momentum - initialMomentum));
    }

    EXPECT_LT(energyDeviation.maximum(), 1e-7)
        << "a symplectic integrator's energy error must stay bounded, not grow, "
           "over many orbits";
    EXPECT_LT(momentumDeviation.maximum(), momentumScale * 1e-9)
        << "total momentum is conserved structurally: Newtonian's pairwise "
           "accelerations are exactly antisymmetric under exchange";
}

TEST(SolarSystemE2E, TheSunAndEightPlanetsConserveEnergyAndMomentumUnderIndividualTimesteps) {
    // Same 9-body dataset and duration as the symplectic test above, run
    // instead through IndividualTimestepScheduler + NewtonianJerkField --
    // the actual physics path src/Applications/SolarSystem/main.cpp now
    // uses. Not symplectic (see Physics/Mechanics/Hermite.hpp's own header
    // comment), and unlike a lockstep integrator, a pair's two bodies are
    // generally updated at different times against each other's predicted
    // (not simultaneously corrected) state, so neither energy nor momentum
    // is structurally exact here -- both tolerances are looser than the
    // symplectic test's own, the same "bounded, not diverging" standard
    // nbody_energy.cpp's own Barnes-Hut test already applies for the
    // analogous reason (its forces are not exactly antisymmetric either).
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    std::vector<ysq::Body> bodies;
    bodies.push_back(scenario->bodies.front().body);
    for (const auto& catalogBody : scenario->bodies) {
        if (catalogBody.parent == "Sun") {
            bodies.push_back(catalogBody.body);
        }
    }
    ASSERT_EQ(bodies.size(), 9u) << "expected the Sun plus exactly 8 planets";

    const ysq::Length softening{1.0e8};
    const double gmSun = ysq::constants::G.value() * ysq::units::solarMass.value();
    const auto orbitalPeriod = [&](const ysq::Body& planet) {
        const double radius = length(planet.position.value());
        return 2.0 * ysq::kPi<double> * std::sqrt(radius * radius * radius / gmSun);
    };

    double slowestPeriod = 0.0;
    for (std::size_t i = 1; i < bodies.size(); ++i) {
        slowestPeriod = std::max(slowestPeriod, orbitalPeriod(bodies[i]));
    }
    const double baseInterval = slowestPeriod / 2000.0;
    const double untilTime = 2.0 * orbitalPeriod(bodies[5]);  // 2 Jupiter orbits

    const double initialEnergy = totalEnergy(bodies, softening);
    const ysq::Vec3 initialMomentum = totalMomentum(bodies);
    const double momentumScale = length(bodies[3].momentum.value());

    const ysq::NewtonianJerkField jerkField(bodies, softening);
    const ysq::NBodyState positions = ysq::positionsOf(bodies);
    const ysq::NBodyState velocities = ysq::velocitiesOf(bodies);
    ysq::NBodyState accelerations(bodies.size());
    ysq::NBodyState jerks(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const auto [acceleration, jerk] = jerkField(i, positions, velocities);
        accelerations[i] = acceleration;
        jerks[i] = jerk;
    }

    ysq::IndividualTimestepScheduler scheduler(positions, velocities, accelerations, jerks,
                                               0.0, 0.01, baseInterval);

    ysq::RunningStatistics<double> energyDeviation;
    ysq::RunningStatistics<double> momentumDeviation;
    constexpr int samples = 200;
    for (int sample = 1; sample <= samples; ++sample) {
        const double targetTime = untilTime * static_cast<double>(sample) / samples;
        scheduler.advanceTo(jerkField, targetTime, 2000000);

        ysq::NBodyState sampledPositions(bodies.size());
        ysq::NBodyState sampledVelocities(bodies.size());
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            const auto [position, velocity] =
                scheduler.predictedState(i, scheduler.currentTime());
            sampledPositions[i] = position;
            sampledVelocities[i] = velocity;
        }
        ysq::applyState(bodies, sampledPositions, sampledVelocities);

        const double energy = totalEnergy(bodies, softening);
        energyDeviation.add(std::abs((energy - initialEnergy) / initialEnergy));

        const ysq::Vec3 momentum = totalMomentum(bodies);
        momentumDeviation.add(length(momentum - initialMomentum));
    }

    EXPECT_LT(energyDeviation.maximum(), 1e-2)
        << "individual-timestep integration must still keep energy roughly "
           "bounded over many orbits";
    EXPECT_LT(momentumDeviation.maximum(), momentumScale * 1e-2)
        << "momentum must stay roughly conserved even though it is only "
           "approximately, not structurally, exact here";
}

TEST(SolarSystemE2E, AllBodiesRemainFiniteOverAShortRun) {
    // The full real catalog -- Sun, planets, and every real moon, roughly
    // 175 bodies spanning orbital periods from under a day (the innermost
    // ring moons) to decades (the outermost captured irregular moons).
    // Not a tight conservation claim the way the 9-body test above makes:
    // the point here is that the whole hierarchy, loaded and integrated
    // together, does not diverge or produce a non-finite state over a
    // short run -- the load-bearing check that Scenario.cpp's full data
    // set is actually usable as a real n-body initial condition, not just
    // internally self-consistent (already checked by
    // tests/unit/applications_helper_bodycatalog.cpp's own per-body
    // distance-from-parent validation).
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());
    std::vector<ysq::Body> bodies = scenario->allBodies();

    // Softened well below the closest real separation in the catalog (an
    // inner ring moon a little over 1e8 m from its planet), so softening
    // does not itself dominate the force there.
    const ysq::Length softening{1.0e6};

    // Sized to the fastest body in the whole catalog (an inner moon's
    // period, not Mercury's), the same "resolve the fastest thing" rule
    // the 9-body test above applies to its own, much slower, fastest body.
    // T = 2 pi sqrt(a^3 / gm), with `a` each body's *own* distance from
    // its *own* parent (not the Sun -- a moon's distance from the Sun is
    // dominated by its planet's own orbit and says nothing about how fast
    // the moon itself orbits) and gm from that parent's real, already-
    // loaded mass.
    std::unordered_map<std::string, std::size_t> indexByName;
    for (std::size_t i = 0; i < scenario->bodies.size(); ++i) {
        indexByName[scenario->bodies[i].name] = i;
    }

    double fastestPeriod = std::numeric_limits<double>::max();
    for (std::size_t i = 1; i < bodies.size(); ++i) {
        const std::size_t parentIndex = indexByName.at(scenario->bodies[i].parent);
        const double gm = ysq::constants::G.value() * bodies[parentIndex].mass.value();
        const double a =
            length(bodies[i].position.value() - bodies[parentIndex].position.value());
        const double period = ysq::kTau<double> * std::sqrt(a * a * a / gm);
        fastestPeriod = std::min(fastestPeriod, period);
    }
    ASSERT_LT(fastestPeriod, std::numeric_limits<double>::max());

    const double step = fastestPeriod / 200.0;
    const int totalSteps = 500;

    ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
    for (int i = 0; i < totalSteps; ++i) {
        const ysq::NewtonianField field(bodies, softening);
        const ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                                     ysq::velocitiesOf(bodies)};
        ysq::PhaseState<ysq::NBodyState> next;
        stepper.step(field, static_cast<double>(i) * step, state, step, next);
        ysq::applyState(bodies, next.position, next.velocity);

        for (const ysq::Body& body : bodies) {
            ASSERT_TRUE(std::isfinite(length(body.position.value())))
                << "position went non-finite by step " << i;
            ASSERT_TRUE(std::isfinite(length(body.momentum.value())))
                << "momentum went non-finite by step " << i;
        }
    }
}
