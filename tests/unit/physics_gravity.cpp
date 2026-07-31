#include <Math/Integrators/Symplectic.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Gravity/PostNewtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace {

using ysq::Body;
using ysq::Vec3;

Body makeBody(double mass, Vec3 position, Vec3 velocity = Vec3{}) {
    Body body{};
    body.mass = ysq::Mass{mass};
    body.position = ysq::Length3{position};
    body.momentum = ysq::Momentum3{velocity * mass};
    return body;
}

/// An oblate source at the origin, spin axis +Z (identity orientation), for
/// the J2 tests below.
Body makeOblateBody(double mass, double radius, double j2) {
    Body body = makeBody(mass, Vec3{0.0, 0.0, 0.0});
    body.radius = ysq::Length{radius};
    body.j2 = j2;
    return body;
}

// --- Newtonian -------------------------------------------------------------

TEST(PhysicsGravity, ForceMagnitudeMatchesTheInverseSquareLaw) {
    const Body a = makeBody(1.0, Vec3{0.0, 0.0, 0.0});
    const Body b = makeBody(1.0, Vec3{1.0, 0.0, 0.0});

    const ysq::Force3 force = ysq::newtonianForce(a, b);
    EXPECT_NEAR(length(force.value()), ysq::constants::G.value(),
                ysq::constants::G.value() * 1e-12);
    // Attractive: points from a toward b, +x.
    EXPECT_GT(force.value().x, 0.0);
    EXPECT_NEAR(force.value().y, 0.0, 1e-20);
    EXPECT_NEAR(force.value().z, 0.0, 1e-20);
}

TEST(PhysicsGravity, ForceOnEachBodyIsEqualAndOpposite) {
    const Body a = makeBody(3.0, Vec3{0.0, 0.0, 0.0});
    const Body b = makeBody(5.0, Vec3{2.0, -1.0, 0.5});

    const ysq::Force3 onA = ysq::newtonianForce(a, b);
    const ysq::Force3 onB = ysq::newtonianForce(b, a);
    EXPECT_QUANTITY_VEC_APPROX(onA, -onB);
}

TEST(PhysicsGravity, AccelerationMatchesForceOverMass) {
    const Body source = makeBody(4.0e10, Vec3{0.0, 0.0, 0.0});
    const Body target = makeBody(1.0, Vec3{7.0, 0.0, 0.0});

    const std::array<Body, 1> sources{source};
    const ysq::Acceleration3 acceleration =
        ysq::newtonianAcceleration(target.position, sources);
    const ysq::Force3 force = ysq::newtonianForce(target, source);

    EXPECT_QUANTITY_VEC_NEAR(acceleration, force / target.mass,
                             1e-12 * ysq::units::metrePerSecondSquared);
}

TEST(PhysicsGravity, SoftenedAccelerationMatchesTheClosedForm) {
    const Body source = makeBody(2.5e12, Vec3{0.0, 0.0, 0.0});
    const ysq::Length at{3.0};
    const ysq::Length softening{2.0};

    const std::array<Body, 1> sources{source};
    const ysq::Acceleration3 acceleration = ysq::newtonianAcceleration(
        ysq::Length3{Vec3{at.value(), 0.0, 0.0}}, sources, softening);

    const double gm = ysq::constants::G.value() * source.mass.value();
    const double r2 = at.value() * at.value() + softening.value() * softening.value();
    const double expected = -gm * at.value() / (r2 * std::sqrt(r2));

    EXPECT_NEAR(acceleration.value().x, expected, std::abs(expected) * 1e-9);
}

TEST(PhysicsGravity, DirectSummationAgreesWithPairwiseForce) {
    const std::vector<Body> bodies{makeBody(2.0, Vec3{0.0, 0.0, 0.0}),
                                   makeBody(3.0, Vec3{5.0, 0.0, 0.0}),
                                   makeBody(1.0, Vec3{0.0, 4.0, 0.0})};

    const std::vector<ysq::Acceleration3> accelerations =
        ysq::newtonianAccelerations(bodies);

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        ysq::Force3 force{};
        for (std::size_t j = 0; j < bodies.size(); ++j) {
            if (i == j) {
                continue;
            }
            force = force + ysq::newtonianForce(bodies[i], bodies[j]);
        }
        EXPECT_QUANTITY_VEC_NEAR(accelerations[i], force / bodies[i].mass,
                                 1e-9 * ysq::units::metrePerSecondSquared);
    }
}

TEST(PhysicsGravity, PotentialEnergyOfTwoBodiesMatchesMinusGMmOverR) {
    const Body a = makeBody(6.0, Vec3{0.0, 0.0, 0.0});
    const Body b = makeBody(9.0, Vec3{4.0, 3.0, 0.0});  // separation 5

    const ysq::Energy energy = ysq::newtonianPotentialEnergy(std::array<Body, 2>{a, b});
    const double expected = -ysq::constants::G.value() * 6.0 * 9.0 / 5.0;
    EXPECT_NEAR(energy.value(), expected, std::abs(expected) * 1e-12);
}

TEST(PhysicsGravity, NewtonianFieldAgreesWithTheDimensionedApi) {
    // Two independent implementations of the same law: one on Body and one
    // on the raw NBodyState the integrators actually run on. They have to
    // agree, since a discrepancy here would mean an integrated orbit and a
    // hand-checked force disagree about what the law even says.
    const std::vector<Body> bodies{
        makeBody(2.0e10, Vec3{0.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}),
        makeBody(3.0e10, Vec3{6.0, 0.0, 0.0}, Vec3{0.0, -1.0, 0.0}),
        makeBody(1.0e10, Vec3{0.0, 5.0, 2.0}, Vec3{1.0, 0.0, 0.0})};

    const std::vector<ysq::Acceleration3> dimensioned =
        ysq::newtonianAccelerations(bodies);

    const ysq::NewtonianField field(bodies);
    const ysq::NBodyState raw = field(0.0, ysq::positionsOf(bodies));

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        EXPECT_VEC_NEAR(raw[i], dimensioned[i].value(), 1e-9);
    }
}

// --- Oblateness (J2) ---------------------------------------------------

TEST(PhysicsGravity, ZeroJ2ReproducesThePlainPointMassTerm) {
    const Body oblate = makeOblateBody(4.0e10, 1.0, 0.0);
    Body point = oblate;
    point.radius = ysq::Length{};

    const std::array<Body, 1> oblateSources{oblate};
    const std::array<Body, 1> pointSources{point};
    const ysq::Length3 at{Vec3{7.0, 2.0, 1.0}};

    EXPECT_QUANTITY_VEC_APPROX(ysq::newtonianAcceleration(at, oblateSources),
                               ysq::newtonianAcceleration(at, pointSources));
}

TEST(PhysicsGravity, EquatorialJ2AccelerationMatchesTheClosedForm) {
    // s = 0 (in the source's equatorial plane): the closed form reduces to a
    // purely radial correction, -(3/2) J2 GM Req^2 / r^4, adding to the
    // point-mass attraction (Vallado's standard J2 perturbation formula).
    const double mass = 5.0e24;
    const double radius = 6.371e6;
    const double j2 = 1.08263e-3;
    const Body source = makeOblateBody(mass, radius, j2);

    const double r = 4.0e7;
    const ysq::Length3 at{Vec3{r, 0.0, 0.0}};
    const std::array<Body, 1> sources{source};
    const ysq::Acceleration3 acceleration = ysq::newtonianAcceleration(at, sources);

    const double gm = ysq::constants::G.value() * mass;
    const double pointMass = -gm / (r * r);
    const double j2Coefficient = 1.5 * j2 * gm * radius * radius;
    const double expectedX = pointMass - j2Coefficient / (r * r * r * r);

    EXPECT_NEAR(acceleration.value().x, expectedX, std::abs(expectedX) * 1e-9);
    EXPECT_NEAR(acceleration.value().y, 0.0, 1e-20);
    EXPECT_NEAR(acceleration.value().z, 0.0, 1e-20);
}

TEST(PhysicsGravity, PolarJ2AccelerationMatchesTheClosedForm) {
    // s = 1 (on the source's spin axis): the closed form reduces to
    // +2 * (3/2) J2 GM Req^2 / r^4 along that axis, a *weaker* pull than the
    // point-mass term, the bulge's mass sitting away from the poles.
    const double mass = 5.0e24;
    const double radius = 6.371e6;
    const double j2 = 1.08263e-3;
    const Body source = makeOblateBody(mass, radius, j2);

    const double r = 4.0e7;
    const ysq::Length3 at{Vec3{0.0, 0.0, r}};
    const std::array<Body, 1> sources{source};
    const ysq::Acceleration3 acceleration = ysq::newtonianAcceleration(at, sources);

    const double gm = ysq::constants::G.value() * mass;
    const double pointMass = -gm / (r * r);
    const double j2Coefficient = 1.5 * j2 * gm * radius * radius;
    const double expectedZ = pointMass + 2.0 * j2Coefficient / (r * r * r * r);

    EXPECT_NEAR(acceleration.value().z, expectedZ, std::abs(expectedZ) * 1e-9);
    EXPECT_NEAR(acceleration.value().x, 0.0, 1e-20);
    EXPECT_NEAR(acceleration.value().y, 0.0, 1e-20);
}

TEST(PhysicsGravity, ForceIsEqualAndOppositeWhenOnlyOneBodyIsOblate) {
    // The pathological case for J2, which comes from one specific body's
    // shape rather than being symmetric in the pair the way the monopole
    // term is: Newton's third law only holds if the oblate body feels an
    // explicit reaction, not merely "no J2 term because it isn't the
    // source".
    Body oblate = makeOblateBody(5.0e24, 6.371e6, 1.08263e-3);
    oblate.position = ysq::Length3{Vec3{0.0, 0.0, 0.0}};
    Body point = makeBody(7.342e22, Vec3{3.844e8, 1.0e8, 5.0e7});

    const ysq::Force3 onPoint = ysq::newtonianForce(point, oblate);
    const ysq::Force3 onOblate = ysq::newtonianForce(oblate, point);
    EXPECT_QUANTITY_VEC_APPROX(onPoint, -onOblate);
}

TEST(PhysicsGravity, MutualAccelerationsConserveMomentumWhenOnlyOneBodyIsOblate) {
    Body oblate = makeOblateBody(5.0e24, 6.371e6, 1.08263e-3);
    oblate.position = ysq::Length3{Vec3{0.0, 0.0, 0.0}};
    Body point = makeBody(7.342e22, Vec3{3.844e8, 1.0e8, 5.0e7});

    const std::vector<Body> bodies{oblate, point};
    const std::vector<ysq::Acceleration3> accelerations =
        ysq::newtonianAccelerations(bodies);

    const Vec3 momentumRateOfChange = accelerations[0].value() * oblate.mass.value() +
                                      accelerations[1].value() * point.mass.value();
    EXPECT_NEAR(length(momentumRateOfChange), 0.0,
                std::max(length(accelerations[0].value()) * oblate.mass.value(),
                         length(accelerations[1].value()) * point.mass.value()) *
                    1e-12);
}

TEST(PhysicsGravity, EnergyIsConservedForAJ2PerturbedOrbit) {
    // A satellite in a moderately close, inclined orbit around an oblate
    // primary: if oblatenessPotentialEnergy did not match the acceleration
    // oblatenessTerm actually applies, this would drift, not merely jitter,
    // over many orbits, the same failure mode the plain two-body
    // orbit_stability.cpp integration test watches for.
    Body oblate = makeOblateBody(5.0e24, 6.371e6, 1.08263e-3);
    const double gm = ysq::constants::G.value() * oblate.mass.value();

    const double radius = 6.371e6 * 3.0;
    const double speed = std::sqrt(gm / radius) * 1.05;  // mildly eccentric
    Body satellite =
        makeBody(1.0e3, Vec3{radius, 0.0, 0.0}, Vec3{0.0, speed * 0.7, speed * 0.7});
    oblate.momentum = ysq::Momentum3{-satellite.momentum.value()};

    std::vector<Body> bodies{oblate, satellite};
    const auto totalEnergy = [&]() {
        double kinetic = 0.0;
        for (const Body& body : bodies) {
            const double v = length(body.velocity().value());
            kinetic += 0.5 * body.mass.value() * v * v;
        }
        return kinetic + ysq::newtonianPotentialEnergy(bodies).value();
    };

    const double initialEnergy = totalEnergy();
    const double period =
        2.0 * ysq::kPi<double> * std::sqrt(radius * radius * radius / gm);
    const double step = period / 16000.0;

    ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
    double maxDeviation = 0.0;
    for (int i = 0; i < 16000 * 3; ++i) {
        const ysq::NewtonianField field(bodies);
        const ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                                     ysq::velocitiesOf(bodies)};
        ysq::PhaseState<ysq::NBodyState> next;
        stepper.step(field, static_cast<double>(i) * step, state, step, next);
        ysq::applyState(bodies, next.position, next.velocity);
        maxDeviation = std::max(
            maxDeviation, std::abs((totalEnergy() - initialEnergy) / initialEnergy));
    }

    EXPECT_LT(maxDeviation, 1e-5);
}

TEST(PhysicsGravity, NewtonianFieldAgreesWithTheDimensionedApiForAnOblateSource) {
    const Body oblateSource = makeOblateBody(5.0e24, 6.371e6, 1.08263e-3);
    const std::vector<Body> bodies{oblateSource,
                                   makeBody(1.0e3, Vec3{4.0e7, 1.0e7, 2.0e6})};

    const std::vector<ysq::Acceleration3> dimensioned =
        ysq::newtonianAccelerations(bodies);

    const ysq::NewtonianField field(bodies);
    const ysq::NBodyState raw = field(0.0, ysq::positionsOf(bodies));

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        EXPECT_VEC_NEAR(raw[i], dimensioned[i].value(), std::abs(length(raw[i])) * 1e-9);
    }
}

// --- Post-Newtonian ----------------------------------------------------

TEST(PhysicsGravity, PostNewtonianCorrectionAtRestMatchesTheClosedForm) {
    // v = 0: a_1PN = (4 G^2 M^2 / (c^2 r^3)) n, pointing away from the
    // source (n itself, not -n, since the term is +4GM/r inside the
    // bracket).
    const Body source = makeBody(2.0e29, Vec3{0.0, 0.0, 0.0});
    const Body testParticle = makeBody(1.0, Vec3{5.0e10, 0.0, 0.0});

    const ysq::Acceleration3 correction =
        ysq::postNewtonianCorrection(testParticle, source);

    const double gm = ysq::constants::G.value() * source.mass.value();
    const double c = ysq::constants::speedOfLight.value();
    const double r = 5.0e10;
    const double expected = 4.0 * gm * gm / (c * c * r * r * r);

    EXPECT_NEAR(correction.value().x, expected, expected * 1e-9);
    EXPECT_NEAR(correction.value().y, 0.0, expected * 1e-9 + 1e-30);
    EXPECT_NEAR(correction.value().z, 0.0, expected * 1e-9 + 1e-30);
}

TEST(PhysicsGravity, PostNewtonianCorrectionIsASmallFractionOfNewtonianInTheWeakField) {
    // A solar-mass source with a circular-orbit test particle at 1 AU: deep
    // in the weak field, where GM / (r c^2) is of order 1e-8.
    const double gm = ysq::constants::G.value() * ysq::units::solarMass.value();
    const double r = ysq::units::astronomicalUnit.value();
    const double circularSpeed = std::sqrt(gm / r);

    const Body source = makeBody(ysq::units::solarMass.value(), Vec3{0.0, 0.0, 0.0});
    const Body testParticle =
        makeBody(1.0, Vec3{r, 0.0, 0.0}, Vec3{0.0, circularSpeed, 0.0});

    const ysq::Acceleration3 newtonian =
        ysq::newtonianAcceleration(testParticle.position, std::array<Body, 1>{source});
    const ysq::Acceleration3 correction =
        ysq::postNewtonianCorrection(testParticle, source);

    const double ratio = length(correction.value()) / length(newtonian.value());
    EXPECT_GT(ratio, 0.0);
    EXPECT_LT(ratio, 1e-3) << "the 1PN term must be a correction, not a leading term, "
                              "in the weak field";
}

}  // namespace
