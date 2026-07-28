#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
#include <Math/Statistics.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/BarnesHut.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Units/Constants.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

/// Barnes-Hut against the same conservation checks orbit_stability.cpp runs
/// on direct summation, plus the property that is specifically Barnes-Hut's
/// own: force error falls as the opening angle shrinks, and vanishes at
/// theta = 0.

namespace {

using ysq::Body;
using ysq::NBodyState;
using ysq::PhaseState;
using ysq::Vec3;

using Phase = PhaseState<NBodyState>;

Body randomBody(std::mt19937& rng, std::uniform_real_distribution<double>& positionDist,
                std::uniform_real_distribution<double>& velocityDist,
                std::uniform_real_distribution<double>& massDist) {
    Body body{};
    body.mass = ysq::Mass{massDist(rng)};
    body.position =
        ysq::Length3{Vec3{positionDist(rng), positionDist(rng), positionDist(rng)}};
    body.momentum =
        ysq::Momentum3{Vec3{velocityDist(rng), velocityDist(rng), velocityDist(rng)} *
                       body.mass.value()};
    return body;
}

/// A fixed, reproducible cluster of modest size: enough bodies for the tree
/// to have more than one level, small enough that direct summation stays the
/// cheap reference it is meant to be.
std::vector<Body> randomCluster(std::size_t count) {
    std::mt19937 rng{2024};
    std::uniform_real_distribution<double> positionDist(-5.0e11, 5.0e11);
    // Comparable to the cluster's own characteristic orbital speed,
    // sqrt(G M_total / R): roughly 16 bodies averaging a few times 1e29 kg
    // over a radius of 5e11 m gives on the order of 2e4 m/s. Velocities far
    // below that (as opposed to far above, which just unbinds the cluster)
    // put the bodies in effective free-fall toward each other with no
    // angular momentum to hold them apart, which a fixed step cannot
    // resolve through the resulting close encounters without an
    // energy-conservation failure that is a property of the step size, not
    // of the force law being integrated.
    std::uniform_real_distribution<double> velocityDist(-2.0e4, 2.0e4);
    std::uniform_real_distribution<double> massDist(1.0e28, 5.0e29);

    std::vector<Body> bodies;
    bodies.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        bodies.push_back(randomBody(rng, positionDist, velocityDist, massDist));
    }
    return bodies;
}

double totalEnergy(const std::vector<double>& masses, const NBodyState& positions,
                   const NBodyState& velocities, double softening) {
    const double g = ysq::constants::G.value();
    double energy = 0.0;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        energy += 0.5 * masses[i] * lengthSquared(velocities[i]);
    }
    for (std::size_t i = 0; i < masses.size(); ++i) {
        for (std::size_t j = i + 1; j < masses.size(); ++j) {
            const double r = std::sqrt(distanceSquared(positions[i], positions[j]) +
                                       softening * softening);
            energy -= g * masses[i] * masses[j] / r;
        }
    }
    return energy;
}

std::vector<double> massesOf(const std::vector<Body>& bodies) {
    std::vector<double> masses;
    masses.reserve(bodies.size());
    for (const Body& body : bodies) {
        masses.push_back(body.mass.value());
    }
    return masses;
}

// --- Barnes-Hut error against direct summation --------------------------

TEST(NBodyEnergy, BarnesHutErrorVanishesAtOpeningAngleZero) {
    const std::vector<Body> bodies = randomCluster(40);
    const NBodyState positions = ysq::positionsOf(bodies);

    const std::vector<ysq::Acceleration3> exact = ysq::newtonianAccelerations(bodies);
    const ysq::BarnesHutTree tree(bodies, 0.0);
    const NBodyState approx = tree(0.0, positions);

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        EXPECT_NEAR(distance(approx[i], exact[i].value()), 0.0,
                    length(exact[i].value()) * 1e-9 + 1e-30);
    }
}

TEST(NBodyEnergy, BarnesHutErrorFallsAsTheOpeningAngleShrinks) {
    const std::vector<Body> bodies = randomCluster(60);
    const NBodyState positions = ysq::positionsOf(bodies);
    const std::vector<ysq::Acceleration3> exact = ysq::newtonianAccelerations(bodies);

    const auto rmsError = [&](double openingAngle) {
        const ysq::BarnesHutTree tree(bodies, openingAngle);
        const NBodyState approx = tree(0.0, positions);
        double sumOfSquares = 0.0;
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            sumOfSquares += distanceSquared(approx[i], exact[i].value());
        }
        return std::sqrt(sumOfSquares / static_cast<double>(bodies.size()));
    };

    const double coarse = rmsError(1.2);
    const double fine = rmsError(0.4);

    EXPECT_GT(coarse, 0.0);
    EXPECT_LT(fine, coarse);
}

// --- Conservation under each force law -----------------------------------

TEST(NBodyEnergy, DirectSummationConservesEnergyOverManySteps) {
    const std::vector<Body> bodies = randomCluster(16);
    const std::vector<double> masses = massesOf(bodies);
    constexpr double softening = 1.0e9;

    ysq::NewtonianField field(bodies, ysq::Length{softening});
    ysq::VelocityVerletStepper<NBodyState> stepper;
    Phase state{ysq::positionsOf(bodies), ysq::velocitiesOf(bodies)};

    const double initialEnergy =
        totalEnergy(masses, state.position, state.velocity, softening);

    ysq::RunningStatistics<double> deviation;
    // A characteristic timescale for this cluster's scale and masses, not
    // tied to any one orbit; the point is many steps, not many periods.
    constexpr double totalTime = 5.0e7;
    constexpr std::size_t steps = 60000;

    ysq::integrate(stepper, field, state, 0.0, totalTime, totalTime / steps,
                   [&](double, const Phase& s) {
                       const double energy =
                           totalEnergy(masses, s.position, s.velocity, softening);
                       deviation.add(std::abs((energy - initialEnergy) / initialEnergy));
                   });

    EXPECT_LT(deviation.maximum(), 1e-4)
        << "direct summation under a symplectic integrator must keep energy "
           "bounded";
}

TEST(NBodyEnergy, BarnesHutDrivenIntegrationKeepsEnergyBounded) {
    const std::vector<Body> bodies = randomCluster(16);
    const std::vector<double> masses = massesOf(bodies);
    constexpr double softening = 1.0e9;

    ysq::BarnesHutTree tree(bodies, 0.3, ysq::Length{softening});
    ysq::VelocityVerletStepper<NBodyState> stepper;
    Phase state{ysq::positionsOf(bodies), ysq::velocitiesOf(bodies)};

    const double initialEnergy =
        totalEnergy(masses, state.position, state.velocity, softening);

    ysq::RunningStatistics<double> deviation;
    constexpr double totalTime = 5.0e7;
    constexpr std::size_t steps = 60000;

    ysq::integrate(stepper, tree, state, 0.0, totalTime, totalTime / steps,
                   [&](double, const Phase& s) {
                       const double energy =
                           totalEnergy(masses, s.position, s.velocity, softening);
                       deviation.add(std::abs((energy - initialEnergy) / initialEnergy));
                   });

    // Barnes-Hut's per-step forces are not exactly antisymmetric between
    // pairs the way direct summation's are: two bodies can be resolved
    // individually from one side of a node boundary and approximated as a
    // group from the other, so momentum, and with it energy, is only
    // approximately conserved rather than structurally so. The bound here is
    // an order of magnitude looser than direct summation's for exactly that
    // reason, and theta = 0.3 rather than the 0.5 default, since 0.5 alone
    // measures closer to 6% on this cluster: still bounded, not diverging,
    // but not what "roughly conserved" should mean.
    EXPECT_LT(deviation.maximum(), 2e-2)
        << "a Barnes-Hut-driven integration must still keep energy roughly "
           "conserved, even though the approximation costs it the direct "
           "method's structural exactness";
}

}  // namespace
