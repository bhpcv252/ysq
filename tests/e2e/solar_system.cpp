#include <Applications/SolarSystem/Scenario.hpp>

#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Statistics.hpp>
#include <Math/Vector3.hpp>

#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>

#include <Units/Length.hpp>
#include <Units/Mass.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

/// solar-system's own scenario (Sun + five planets, real masses and orbital
/// radii, zero-momentum frame), integrated headless and checked against the
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

TEST(SolarSystemE2E, ConservesEnergyAndMomentumOverTwoJupiterOrbits) {
    const Scenario scenario = makeScenario();
    std::vector<ysq::Body> bodies = scenario.allBodies();
    const ysq::Length softening{1.0e8};

    const double gmSun = ysq::constants::G.value() * ysq::units::solarMass.value();
    const auto orbitalPeriod = [&](const ysq::Body& planet) {
        const double radius = length(planet.position.value());
        return 2.0 * ysq::kPi<double> * std::sqrt(radius * radius * radius / gmSun);
    };

    // bodies[0] is the Sun; Mercury (index 1) is the fastest-moving body and
    // sets the step size an order-2 symplectic method needs to stay
    // accurate, Jupiter (last) is the slowest and sets how long a run has to
    // be to say anything about the outer orbits.
    const double mercuryPeriod = orbitalPeriod(bodies[1]);
    const double jupiterPeriod = orbitalPeriod(bodies.back());

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
