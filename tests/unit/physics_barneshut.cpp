#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/BarnesHut.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace {

using ysq::Body;
using ysq::Vec3;

Body makeBody(double mass, Vec3 position) {
    Body body{};
    body.mass = ysq::Mass{mass};
    body.position = ysq::Length3{position};
    return body;
}

/// A fixed, reproducible cluster: not physically meaningful, just varied
/// enough (positions spanning several orders of magnitude of separation) to
/// exercise more than one level of the tree.
std::vector<Body> randomCluster(std::size_t count) {
    std::mt19937 rng{12345};
    std::uniform_real_distribution<double> positionDist(-1.0e11, 1.0e11);
    std::uniform_real_distribution<double> massDist(1.0e28, 1.0e30);

    std::vector<Body> bodies;
    bodies.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        bodies.push_back(
            makeBody(massDist(rng),
                     Vec3{positionDist(rng), positionDist(rng), positionDist(rng)}));
    }
    return bodies;
}

double rmsAccelerationError(const std::vector<Body>& bodies, double openingAngle) {
    const std::vector<ysq::Acceleration3> exact = ysq::newtonianAccelerations(bodies);

    const ysq::BarnesHutTree tree(bodies, openingAngle);
    const ysq::NBodyState approx = tree(0.0, ysq::positionsOf(bodies));

    double sumOfSquares = 0.0;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        sumOfSquares += distanceSquared(approx[i], exact[i].value());
    }
    return std::sqrt(sumOfSquares / static_cast<double>(bodies.size()));
}

double rmsAccelerationMagnitude(const std::vector<Body>& bodies) {
    const std::vector<ysq::Acceleration3> exact = ysq::newtonianAccelerations(bodies);
    double sumOfSquares = 0.0;
    for (const ysq::Acceleration3& a : exact) {
        sumOfSquares += lengthSquared(a.value());
    }
    return std::sqrt(sumOfSquares / static_cast<double>(bodies.size()));
}

TEST(PhysicsBarnesHut, TwoBodiesMatchDirectSummationRegardlessOfOpeningAngle) {
    const std::vector<Body> bodies{makeBody(2.0e10, Vec3{0.0, 0.0, 0.0}),
                                   makeBody(3.0e10, Vec3{5.0, 1.0, -2.0})};

    const std::vector<ysq::Acceleration3> exact = ysq::newtonianAccelerations(bodies);
    const ysq::BarnesHutTree tree(bodies, 0.5);
    const ysq::NBodyState approx = tree(0.0, ysq::positionsOf(bodies));

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        EXPECT_VEC_NEAR(approx[i], exact[i].value(), 1e-9);
    }
}

TEST(PhysicsBarnesHut, AnOpeningAngleOfZeroReproducesDirectSummation) {
    // theta = 0 forces every node open, all the way to the leaves, which is
    // direct summation with extra bookkeeping.
    const std::vector<Body> bodies = randomCluster(24);
    const double error = rmsAccelerationError(bodies, 0.0);
    const double scale = rmsAccelerationMagnitude(bodies);

    EXPECT_LT(error, scale * 1e-9);
}

TEST(PhysicsBarnesHut, ErrorFallsAsTheOpeningAngleShrinks) {
    const std::vector<Body> bodies = randomCluster(48);

    const double looseError = rmsAccelerationError(bodies, 1.0);
    const double tightError = rmsAccelerationError(bodies, 0.3);

    EXPECT_GT(looseError, 0.0)
        << "a nontrivial cluster must show some approximation error "
           "at a coarse opening angle";
    EXPECT_LT(tightError, looseError)
        << "a smaller opening angle must approximate more accurately, not less";
}

TEST(PhysicsBarnesHut, ASingleBodyFeelsNoForce) {
    const std::vector<Body> bodies{makeBody(1.0e20, Vec3{3.0, 4.0, 0.0})};
    const ysq::BarnesHutTree tree(bodies, 0.5);
    const ysq::NBodyState result = tree(0.0, ysq::positionsOf(bodies));

    EXPECT_VEC_NEAR(result[0], Vec3{}, 1e-30);
}

TEST(PhysicsBarnesHut, ADistantWellSeparatedGroupApproximatesAsItsCenterOfMass) {
    // Two bodies close together, far from a third: at the third body's
    // position, the pair's field should already be close to that of one
    // combined mass at their center of mass, even at a middling theta.
    const double m1 = 4.0e28;
    const double m2 = 6.0e28;
    const Body a = makeBody(m1, Vec3{0.0, 0.0, 0.0});
    const Body b = makeBody(m2, Vec3{1.0e8, 0.0, 0.0});
    const Body farAway = makeBody(1.0, Vec3{1.0e13, 0.0, 0.0});

    const std::vector<Body> bodies{a, b, farAway};
    const ysq::BarnesHutTree tree(bodies, 0.5);
    const ysq::NBodyState approx = tree(0.0, ysq::positionsOf(bodies));

    const Vec3 centerOfMass =
        (a.position.value() * m1 + b.position.value() * m2) / (m1 + m2);
    const Body combined = makeBody(m1 + m2, centerOfMass);
    const ysq::Acceleration3 expected =
        ysq::newtonianAcceleration(farAway.position, std::array<Body, 1>{combined});

    EXPECT_VEC_NEAR(approx[2], expected.value(), length(expected.value()) * 1e-3);
}

}  // namespace
