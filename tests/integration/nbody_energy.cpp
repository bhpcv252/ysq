#include <Math/Integrators/RK4.hpp>
#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Statistics.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/BarnesHut.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Physics/Mechanics/Hermite.hpp>
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

// --- IndividualTimestepScheduler ------------------------------------------

TEST(NBodyEnergy, IndividualTimestepSchedulerKeepsEnergyBounded) {
    const std::vector<Body> bodies = randomCluster(16);
    const std::vector<double> masses = massesOf(bodies);
    constexpr double softening = 1.0e9;

    const ysq::NewtonianJerkField jerkField(bodies, ysq::Length{softening});
    const NBodyState positions = ysq::positionsOf(bodies);
    const NBodyState velocities = ysq::velocitiesOf(bodies);

    NBodyState accelerations(bodies.size());
    NBodyState jerks(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const auto [acceleration, jerk] = jerkField(i, positions, velocities);
        accelerations[i] = acceleration;
        jerks[i] = jerk;
    }

    constexpr double eta = 0.01;
    // Well above the cluster's own characteristic timescale (a few times
    // 1e4 s, per randomCluster's own comment), so the Aarseth criterion is
    // what actually sets each body's own step, not this ceiling.
    constexpr double baseInterval = 1.0e6;
    ysq::IndividualTimestepScheduler scheduler(positions, velocities, accelerations, jerks,
                                               0.0, eta, baseInterval);

    const double initialEnergy = totalEnergy(masses, positions, velocities, softening);

    ysq::RunningStatistics<double> deviation;
    constexpr double totalTime = 5.0e7;
    constexpr int samples = 200;
    constexpr int maxUpdatesPerSample = 200000;

    for (int sample = 1; sample <= samples; ++sample) {
        const double targetTime = totalTime * static_cast<double>(sample) / samples;
        scheduler.advanceTo(jerkField, targetTime, maxUpdatesPerSample);

        NBodyState sampledPositions(bodies.size());
        NBodyState sampledVelocities(bodies.size());
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            const auto [p, v] = scheduler.predictedState(i, scheduler.currentTime());
            sampledPositions[i] = p;
            sampledVelocities[i] = v;
        }
        const double energy =
            totalEnergy(masses, sampledPositions, sampledVelocities, softening);
        deviation.add(std::abs((energy - initialEnergy) / initialEnergy));
    }

    // Not symplectic (Physics/Mechanics/Hermite.hpp's own header comment):
    // no strict bounded-error guarantee the way VelocityVerletStepper has,
    // so this tolerance is looser than
    // DirectSummationConservesEnergyOverManySteps's -- the point here is
    // "bounded, not diverging", not "as tight as a symplectic method".
    EXPECT_LT(deviation.maximum(), 1e-2)
        << "individual-timestep integration must still keep energy roughly "
           "bounded over many orbits";
}

TEST(NBodyEnergy, IndividualTimestepSchedulerMatchesRk4GroundTruthAcrossVeryDifferentTimescales) {
    // A star with one distant, slow-orbiting body and one close,
    // fast-orbiting body -- the same shape of problem (timescales three
    // orders of magnitude apart) that motivated giving each body its own
    // step size in the first place. Checked against an independent RK4 run
    // stepped uniformly fine enough to resolve the fast body everywhere,
    // the same "ground truth" role src/Applications/SolarSystem/main.cpp's
    // own fixedStep already plays today.
    const double starMass = 2.0e30;
    const double gm = ysq::constants::G.value() * starMass;

    const double slowRadius = 1.5e11;
    const double slowSpeed = std::sqrt(gm / slowRadius);
    const double fastRadius = 1.0e9;
    const double fastSpeed = std::sqrt(gm / fastRadius);
    const double fastPeriod = ysq::kTau<double> * fastRadius / fastSpeed;

    Body star{};
    star.mass = ysq::Mass{starMass};

    Body slow{};
    slow.mass = ysq::Mass{1.0e24};
    slow.position = ysq::Length3{Vec3{slowRadius, 0.0, 0.0}};
    slow.momentum = ysq::Momentum3{Vec3{0.0, slowSpeed, 0.0} * slow.mass.value()};

    Body fast{};
    fast.mass = ysq::Mass{1.0e20};
    fast.position = ysq::Length3{Vec3{fastRadius, 0.0, 0.0}};
    fast.momentum = ysq::Momentum3{Vec3{0.0, fastSpeed, 0.0} * fast.mass.value()};

    const std::vector<Body> bodies{star, slow, fast};

    constexpr std::size_t stepsPerFastPeriod = 2000;
    constexpr std::size_t fastPeriods = 5;
    const double rk4Step = fastPeriod / static_cast<double>(stepsPerFastPeriod);
    const std::size_t rk4StepCount = stepsPerFastPeriod * fastPeriods;
    const double totalTime = rk4Step * static_cast<double>(rk4StepCount);

    // --- Ground truth: RK4, stepped fine enough to resolve the fast body. ---
    ysq::Rk4Stepper<Phase> rk4Stepper;
    ysq::NewtonianField field(bodies);
    const auto rk4System = ysq::asPhaseSystem(field);
    Phase rk4State{ysq::positionsOf(bodies), ysq::velocitiesOf(bodies)};
    Phase rk4Next = rk4State;
    for (std::size_t i = 0; i < rk4StepCount; ++i) {
        rk4Stepper.step(rk4System, static_cast<double>(i) * rk4Step, rk4State, rk4Step,
                        rk4Next);
        rk4State = rk4Next;
    }

    // --- System under test: individual timesteps, each body at its own rate. ---
    const ysq::NewtonianJerkField jerkField(bodies);
    const NBodyState positions = ysq::positionsOf(bodies);
    const NBodyState velocities = ysq::velocitiesOf(bodies);
    NBodyState accelerations(bodies.size());
    NBodyState jerks(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const auto [acceleration, jerk] = jerkField(i, positions, velocities);
        accelerations[i] = acceleration;
        jerks[i] = jerk;
    }

    ysq::IndividualTimestepScheduler scheduler(positions, velocities, accelerations, jerks,
                                               0.0, 0.01, fastPeriod);
    scheduler.advanceTo(jerkField, totalTime, 2000000);

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const auto [predictedPosition, predictedVelocity] =
            scheduler.predictedState(i, totalTime);
        (void)predictedVelocity;
        EXPECT_NEAR(distance(predictedPosition, rk4State.position[i]), 0.0,
                    slowRadius * 1e-4)
            << "body " << i;
    }
}

TEST(NBodyEnergy, PredictedStateStaysBoundedWhenTheSchedulerFallsBehind) {
    // Regression test for the exact failure a real run hit: a caller that
    // keeps asking predictedState() for an ever-advancing target time,
    // without ever checking whether advanceTo() actually reached it, feeds
    // the predictor a gap that grows every call once maxUpdates stops
    // being enough -- and the cubic extrapolation diverges. The fix is for
    // the caller to clamp its own notion of "now" to currentTime() after
    // every advanceTo(), the same way Core::Clock lets simulation time
    // fall behind real time rather than let an unconsumed backlog
    // compound; this proves that pattern actually stays bounded, deliberately
    // forcing the scheduler to fall behind (a tiny maxUpdates) for far
    // longer than any real frame budget would.
    const std::vector<Body> bodies = randomCluster(16);
    const std::vector<double> masses = massesOf(bodies);
    constexpr double softening = 1.0e9;

    const ysq::NewtonianJerkField jerkField(bodies, ysq::Length{softening});
    const NBodyState positions = ysq::positionsOf(bodies);
    const NBodyState velocities = ysq::velocitiesOf(bodies);
    NBodyState accelerations(bodies.size());
    NBodyState jerks(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const auto [acceleration, jerk] = jerkField(i, positions, velocities);
        accelerations[i] = acceleration;
        jerks[i] = jerk;
    }

    constexpr double eta = 0.01;
    constexpr double baseInterval = 1.0e6;
    ysq::IndividualTimestepScheduler scheduler(positions, velocities, accelerations, jerks,
                                               0.0, eta, baseInterval);

    const double initialEnergy = totalEnergy(masses, positions, velocities, softening);

    double simulationTime = 0.0;
    constexpr double requestedPerFrame = 5.0e6;  // far more than a tiny maxUpdates can reach
    constexpr int tinyMaxUpdates = 5;
    constexpr int frames = 500;

    for (int frame = 0; frame < frames; ++frame) {
        simulationTime += requestedPerFrame;
        scheduler.advanceTo(jerkField, simulationTime, tinyMaxUpdates);
        // The fix under test: clamp to what was actually reached.
        simulationTime = scheduler.currentTime();

        NBodyState sampledPositions(bodies.size());
        NBodyState sampledVelocities(bodies.size());
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            const auto [position, velocity] =
                scheduler.predictedState(i, simulationTime);
            ASSERT_TRUE(std::isfinite(length(position)))
                << "frame " << frame << " body " << i;
            ASSERT_TRUE(std::isfinite(length(velocity)))
                << "frame " << frame << " body " << i;
            sampledPositions[i] = position;
            sampledVelocities[i] = velocity;
        }

        const double energy =
            totalEnergy(masses, sampledPositions, sampledVelocities, softening);
        ASSERT_TRUE(std::isfinite(energy)) << "frame " << frame;
        const double drift = std::abs((energy - initialEnergy) / initialEnergy);
        EXPECT_LT(drift, 1e-2) << "frame " << frame
                               << ": energy must stay bounded even when the scheduler "
                                  "can never catch up to the requested rate";
    }
}

}  // namespace
