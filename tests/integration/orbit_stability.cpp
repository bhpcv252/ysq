#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
#include <Math/Statistics.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Units/Constants.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

/// Gravity's Newtonian rung driven by a symplectic integrator, on an actual
/// two-body system built out of Body and NewtonianField rather than a raw
/// acceleration lambda: math_kepler.cpp already established that a symplectic
/// method holds a Kepler orbit at the Math layer. What this asks is whether
/// Physics's own pieces, Body, NBodyState and NewtonianField, compose to the
/// same result.
///
/// A genuine two-body system, not one fixed body and one test particle: both
/// masses move, set up in the center-of-mass frame so total momentum is
/// exactly zero and total position-weighted-by-mass is exactly zero, which
/// gives two more conserved quantities to check than a single-body orbit
/// does.

namespace {

using ysq::Body;
using ysq::NBodyState;
using ysq::PhaseState;
using ysq::Vec3;

using Phase = PhaseState<NBodyState>;

constexpr double kPi = ysq::kPi<double>;

double kineticEnergy(const std::vector<double>& masses, const NBodyState& velocities) {
    double total = 0.0;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        total += 0.5 * masses[i] * lengthSquared(velocities[i]);
    }
    return total;
}

double potentialEnergy(const std::vector<double>& masses, const NBodyState& positions) {
    const double g = ysq::constants::G.value();
    double total = 0.0;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        for (std::size_t j = i + 1; j < masses.size(); ++j) {
            const double r = length(positions[j] - positions[i]);
            total -= g * masses[i] * masses[j] / r;
        }
    }
    return total;
}

Vec3 totalMomentum(const std::vector<double>& masses, const NBodyState& velocities) {
    Vec3 total{};
    for (std::size_t i = 0; i < masses.size(); ++i) {
        total += velocities[i] * masses[i];
    }
    return total;
}

Vec3 totalAngularMomentum(const std::vector<double>& masses, const NBodyState& positions,
                          const NBodyState& velocities) {
    Vec3 total{};
    for (std::size_t i = 0; i < masses.size(); ++i) {
        total += cross(positions[i], velocities[i] * masses[i]);
    }
    return total;
}

struct TwoBodySystem {
    std::vector<Body> bodies;
    std::vector<double> masses;
    double period = 0.0;
    double apoapsis = 0.0;
    double periapsis = 0.0;
};

TwoBodySystem makeEccentricTwoBodySystem() {
    const double m1 = ysq::units::solarMass.value();
    const double m2 = 0.1 * ysq::units::solarMass.value();
    const double gmTotal = ysq::constants::G.value() * (m1 + m2);

    constexpr double semiMajorAxis = 1.0;  // astronomical units
    constexpr double eccentricity = 0.3;
    const double a = semiMajorAxis * ysq::units::astronomicalUnit.value();

    const double rApoapsis = a * (1.0 + eccentricity);
    const double vRelative =
        std::sqrt(gmTotal * (1.0 - eccentricity) / (a * (1.0 + eccentricity)));

    const Vec3 rRel{rApoapsis, 0.0, 0.0};
    const Vec3 vRel{0.0, vRelative, 0.0};

    TwoBodySystem system;
    system.masses = {m1, m2};
    system.period = 2.0 * kPi * std::sqrt(a * a * a / gmTotal);
    system.apoapsis = a * (1.0 + eccentricity);
    system.periapsis = a * (1.0 - eccentricity);

    Body body1{};
    body1.mass = ysq::Mass{m1};
    body1.position = ysq::Length3{rRel * (-m2 / (m1 + m2))};
    body1.momentum = ysq::Momentum3{vRel * (-m2 / (m1 + m2)) * m1};

    Body body2{};
    body2.mass = ysq::Mass{m2};
    body2.position = ysq::Length3{rRel * (m1 / (m1 + m2))};
    body2.momentum = ysq::Momentum3{vRel * (m1 / (m1 + m2)) * m2};

    system.bodies = {body1, body2};
    return system;
}

TEST(OrbitStability, TwoBodyOrbitConservesEnergyMomentumAndAngularMomentum) {
    const TwoBodySystem system = makeEccentricTwoBodySystem();

    ysq::NewtonianField field(system.bodies);
    ysq::VelocityVerletStepper<NBodyState> stepper;

    Phase state{ysq::positionsOf(system.bodies), ysq::velocitiesOf(system.bodies)};

    const double initialEnergy = kineticEnergy(system.masses, state.velocity) +
                                 potentialEnergy(system.masses, state.position);
    const Vec3 initialMomentum = totalMomentum(system.masses, state.velocity);
    const Vec3 initialAngularMomentum =
        totalAngularMomentum(system.masses, state.position, state.velocity);
    const double angularMomentumScale = length(initialAngularMomentum);
    // The characteristic momentum scale, for a tolerance on a quantity whose
    // true value is exactly zero by construction (center-of-mass frame).
    const double momentumScale = system.masses[1] * length(state.velocity[1]);

    ysq::RunningStatistics<double> energyDeviation;
    ysq::RunningStatistics<double> momentumDeviation;
    ysq::RunningStatistics<double> angularMomentumDeviation;
    ysq::RunningStatistics<double> separation;

    const double step = system.period / 8000.0;
    const double untilTime = 30.0 * system.period;

    ysq::integrate(
        stepper, field, state, 0.0, untilTime, step, [&](double, const Phase& s) {
            const double energy = kineticEnergy(system.masses, s.velocity) +
                                  potentialEnergy(system.masses, s.position);
            energyDeviation.add(std::abs((energy - initialEnergy) / initialEnergy));

            const Vec3 momentum = totalMomentum(system.masses, s.velocity);
            momentumDeviation.add(length(momentum - initialMomentum));

            const Vec3 angularMomentum =
                totalAngularMomentum(system.masses, s.position, s.velocity);
            angularMomentumDeviation.add(
                length(angularMomentum - initialAngularMomentum));

            separation.add(length(s.position[1] - s.position[0]));
        });

    EXPECT_LT(energyDeviation.maximum(), 1e-6)
        << "a symplectic integrator's energy error must stay bounded, not grow, "
           "over many orbits";

    EXPECT_LT(momentumDeviation.maximum(), momentumScale * 1e-9)
        << "total momentum is conserved structurally: Newtonian's pairwise "
           "accelerations are exactly antisymmetric under exchange";

    EXPECT_LT(angularMomentumDeviation.maximum(), angularMomentumScale * 1e-9);

    EXPECT_GE(separation.minimum(), system.periapsis * 0.999);
    EXPECT_LE(separation.maximum(), system.apoapsis * 1.001);
}

}  // namespace
