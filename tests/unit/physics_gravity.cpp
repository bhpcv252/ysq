#include <Math/Integrators/RK4.hpp>
#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
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

TEST(PhysicsGravity, NewtonianJerkFieldMatchesFiniteDifferenceOfAcceleration) {
    // NewtonianField's acceleration is a pure function of position; its
    // time-derivative along any trajectory is the directional derivative in
    // the direction every body's own velocity moves it -- exactly what a
    // central difference of NewtonianField itself, evaluated at every
    // body's position nudged forward and backward by its own velocity,
    // measures, independent of anything the jerk formula's own derivation
    // assumed. This is the mandatory safety net for a hand-derived formula:
    // if NewtonianJerkField's algebra were wrong, this is what would catch
    // it, not a second reading of the same derivation.
    const std::vector<Body> bodies{
        makeBody(2.0e30, Vec3{0.0, 0.0, 0.0}, Vec3{1.0e2, -5.0e1, 0.0}),
        makeBody(6.0e24, Vec3{1.5e11, 0.0, 0.0}, Vec3{0.0, 3.0e4, 0.0}),
        makeBody(5.0e20, Vec3{-2.0e11, 1.0e10, 3.0e9}, Vec3{1.0e3, -2.0e4, 5.0e2})};

    const ysq::NBodyState positions = ysq::positionsOf(bodies);
    const ysq::NBodyState velocities = ysq::velocitiesOf(bodies);

    constexpr double delta = 1.0;
    const auto nudgedPositions = [&](double dt) {
        ysq::NBodyState result(bodies.size());
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            result[i] = positions[i] + velocities[i] * dt;
        }
        return result;
    };

    const ysq::NewtonianField field(bodies);
    const ysq::NBodyState accelerationBefore = field(0.0, nudgedPositions(-delta));
    const ysq::NBodyState accelerationAfter = field(0.0, nudgedPositions(delta));

    const ysq::NewtonianJerkField jerkField(bodies);
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Vec3 finiteDifferenceJerk =
            (accelerationAfter[i] - accelerationBefore[i]) / (2.0 * delta);
        const auto [acceleration, jerk] = jerkField(i, positions, velocities);
        (void)acceleration;

        EXPECT_NEAR(distance(jerk, finiteDifferenceJerk), 0.0,
                    length(finiteDifferenceJerk) * 1e-6)
            << "body " << i;
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

TEST(PhysicsGravity, RelativisticNBodySystemWithNoPrimariesMatchesPlainNewtonianExactly) {
    // Every primaryIndex negative: no relativistic correction anywhere, so
    // this must degenerate to exactly what NewtonianField alone computes
    // (the same acceleration, wrapped as the velocity/acceleration first-
    // order system Rk4Stepper needs), not merely something close to it.
    const std::vector<Body> bodies{
        makeBody(2.0e30, Vec3{0.0, 0.0, 0.0}),
        makeBody(6.0e24, Vec3{1.5e11, 0.0, 0.0}, Vec3{0.0, 3.0e4, 0.0}),
        makeBody(5.0e20, Vec3{-2.0e11, 1.0e10, 0.0}, Vec3{1.0e3, -2.0e4, 5.0e2})};

    const ysq::NewtonianField newtonian(bodies);
    const ysq::RelativisticNBodySystem relativistic(
        bodies, std::vector<int>{-1, -1, -1});

    const ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                                 ysq::velocitiesOf(bodies)};
    const ysq::NBodyState expectedAcceleration = newtonian(0.0, state.position);
    const ysq::PhaseState<ysq::NBodyState> actual = relativistic(0.0, state);

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        // operator() returns {velocity, acceleration} as the first-order
        // system's own (position, velocity) derivative: the returned
        // *position* slot holds the input velocity echoed straight
        // through, and the returned *velocity* slot holds the actual
        // acceleration.
        EXPECT_VEC_NEAR(actual.position[i], state.velocity[i], 0.0)
            << "body " << i << ": the system's own position slot must be "
                              "state.velocity echoed straight through";
        EXPECT_VEC_NEAR(actual.velocity[i], expectedAcceleration[i],
                       length(expectedAcceleration[i]) * 1e-12)
            << "body " << i;
    }
}

TEST(PhysicsGravity, RelativisticNBodySystemCorrectsAMoonAgainstItsOwnPlanetNotTheStar) {
    // Three bodies: a star, a planet orbiting it, and a moon orbiting the
    // planet. The moon's primaryIndex names the planet, not the star; this
    // checks the correction actually uses the planet's mass and the
    // moon-planet relative state, not the star's -- the exact composition
    // Applications/SolarSystem/main.cpp relies on for real moons.
    const Body star = makeBody(2.0e30, Vec3{0.0, 0.0, 0.0});
    const Body planet =
        makeBody(6.0e24, Vec3{1.5e11, 0.0, 0.0}, Vec3{0.0, 3.0e4, 0.0});
    // The moon sits at the planet's own position offset by a small amount,
    // with a small relative velocity: close enough that its 1PN term
    // against the (much less massive, much closer) planet is not
    // negligible next to what it would be against the (far more massive,
    // far more distant) star, so the two cases are numerically
    // distinguishable, not just formally different code paths.
    const Body moon = makeBody(7.0e22, Vec3{1.5e11 + 4.0e8, 0.0, 0.0},
                               Vec3{0.0, 3.0e4 + 1.0e3, 0.0});
    const std::array<Body, 3> bodies{star, planet, moon};

    const ysq::RelativisticNBodySystem correctedAgainstPlanet(
        bodies, std::vector<int>{-1, -1, 1});
    const ysq::RelativisticNBodySystem correctedAgainstStar(bodies,
                                                            std::vector<int>{-1, -1, 0});

    const ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                                 ysq::velocitiesOf(bodies)};
    const ysq::PhaseState<ysq::NBodyState> withPlanetPrimary =
        correctedAgainstPlanet(0.0, state);
    const ysq::PhaseState<ysq::NBodyState> withStarPrimary =
        correctedAgainstStar(0.0, state);

    // operator()'s returned *velocity* slot holds the acceleration (see
    // the sibling test above for why); position holds velocity echoed
    // through and is identical in both cases regardless of primary choice.
    const double difference =
        length(withPlanetPrimary.velocity[2] - withStarPrimary.velocity[2]);
    EXPECT_GT(difference, 0.0)
        << "correcting against the planet instead of the star must change the "
           "moon's own acceleration";

    // Independently: subtract off what postNewtonianCorrection itself says
    // the moon-vs-planet term should be, and confirm what remains is
    // exactly the (primary-independent) Newtonian acceleration -- proving
    // the difference above is specifically the relativistic term, not some
    // other discrepancy.
    const ysq::Acceleration3 expectedCorrection = ysq::postNewtonianCorrection(moon, planet);
    const ysq::NewtonianField newtonian(bodies);
    const ysq::NBodyState newtonianOnly = newtonian(0.0, state.position);

    EXPECT_VEC_NEAR(withPlanetPrimary.velocity[2],
                   newtonianOnly[2] + expectedCorrection.value(),
                   length(expectedCorrection.value()) * 1e-9);
}

TEST(PhysicsGravity, RelativisticNBodySystemPerihelionPrecessionMatchesTheAnalyticRate) {
    // The Mercury-like case src/Physics/README.md's gravity ladder section
    // describes: a negligible-mass planet on an eccentric orbit around a
    // dominant star, integrated through RelativisticNBodySystem (Newtonian
    // plus this body's own 1PN correction against its primary) with
    // Rk4Stepper, and checked against the same closed-form precession rate
    // delta_phi = 6 pi GM / (c^2 a (1 - e^2)) that
    // tests/integration/lensing_deflection.cpp validates a full
    // Schwarzschild geodesic against -- the same weak-field parameters
    // (gm, 200 Schwarzschild radii, e = 0.4), so the two independent
    // routes to the same physics are directly comparable.
    const double gm = 5.0e14;
    const double c = ysq::constants::speedOfLight.value();
    const double rs = 2.0 * gm / (c * c);

    constexpr double eccentricity = 0.4;
    const double semiMajorAxis = 200.0 * rs;
    const double periapsisDistance = semiMajorAxis * (1.0 - eccentricity);
    // Vis-viva at periapsis, purely tangential velocity there: v_peri =
    // sqrt(gm (1 + e) / (a (1 - e))).
    const double periapsisSpeed =
        std::sqrt(gm * (1.0 + eccentricity) / (semiMajorAxis * (1.0 - eccentricity)));

    const Body star = makeBody(gm / ysq::constants::G.value(), Vec3{0.0, 0.0, 0.0});
    const Body planet =
        makeBody(1.0, Vec3{periapsisDistance, 0.0, 0.0}, Vec3{0.0, periapsisSpeed, 0.0});
    const std::array<Body, 2> bodies{star, planet};

    // The star (index 0) gets no relativistic correction (nothing more
    // dominant nearby it); the planet (index 1) gets one against the star.
    const ysq::RelativisticNBodySystem system(bodies, std::vector<int>{-1, 0});
    ysq::Rk4Stepper<ysq::PhaseState<ysq::NBodyState>> stepper;

    const double newtonianPeriod =
        2.0 * ysq::kPi<double> *
        std::sqrt(semiMajorAxis * semiMajorAxis * semiMajorAxis / gm);
    const double step = newtonianPeriod / 20000.0;
    constexpr int targetOrbits = 6;
    const std::size_t maxSteps =
        static_cast<std::size_t>(newtonianPeriod / step) * (targetOrbits + 2);

    ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies), ysq::velocitiesOf(bodies)};
    ysq::PhaseState<ysq::NBodyState> next = state;

    // atan2 wraps to [-pi, pi] every step; the geodesic-based version of
    // this test avoids needing to unwrap anything because Schwarzschild's
    // own phi coordinate accumulates continuously by construction. Here,
    // in Cartesian, `unwrappedAzimuth` is built to do the same: each
    // step's raw atan2 sample only ever contributes its small delta from
    // the previous *raw* sample (never more than pi in magnitude, since
    // 20000 steps per orbit moves the true angle by far less than that),
    // so the running total never actually hits a branch cut to begin
    // with. Without this, consecutive recorded periapsis azimuths would
    // differ by "precession mod 2*pi" instead of "2*pi + precession", and
    // the subtraction below would be measuring the wrong thing entirely.
    std::vector<double> periapsisAzimuths;
    double previousRadialSpeed =
        dot(state.position[1] - state.position[0], state.velocity[1] - state.velocity[0]) /
        length(state.position[1] - state.position[0]);
    double previousRawAzimuth = std::atan2(state.position[1].y, state.position[1].x);
    double previousUnwrappedAzimuth = previousRawAzimuth;
    double unwrappedAzimuth = previousRawAzimuth;

    for (std::size_t i = 0;
        i < maxSteps && static_cast<int>(periapsisAzimuths.size()) < targetOrbits; ++i) {
        stepper.step(system, static_cast<double>(i) * step, state, step, next);

        const Vec3 relativePosition = next.position[1] - next.position[0];
        const Vec3 relativeVelocity = next.velocity[1] - next.velocity[0];
        const double radialSpeed =
            dot(relativePosition, relativeVelocity) / length(relativePosition);
        const double rawAzimuth = std::atan2(relativePosition.y, relativePosition.x);

        double delta = rawAzimuth - previousRawAzimuth;
        if (delta > ysq::kPi<double>) {
            delta -= ysq::kTau<double>;
        } else if (delta < -ysq::kPi<double>) {
            delta += ysq::kTau<double>;
        }
        previousUnwrappedAzimuth = unwrappedAzimuth;
        unwrappedAzimuth += delta;

        if (previousRadialSpeed < 0.0 && radialSpeed >= 0.0) {
            const double t = -previousRadialSpeed / (radialSpeed - previousRadialSpeed);
            periapsisAzimuths.push_back(
                previousUnwrappedAzimuth + t * (unwrappedAzimuth - previousUnwrappedAzimuth));
        }

        previousRadialSpeed = radialSpeed;
        previousRawAzimuth = rawAzimuth;
        state = next;
    }

    ASSERT_GE(periapsisAzimuths.size(), 2u)
        << "not enough periapsis passages were observed";

    double meanPrecession = 0.0;
    for (std::size_t i = 1; i < periapsisAzimuths.size(); ++i) {
        meanPrecession += periapsisAzimuths[i] - periapsisAzimuths[i - 1] - ysq::kTau<double>;
    }
    meanPrecession /= static_cast<double>(periapsisAzimuths.size() - 1);

    const double expectedPrecession =
        ysq::perihelionPrecessionPerOrbit(gm, semiMajorAxis, eccentricity);

    EXPECT_NEAR(meanPrecession, expectedPrecession, expectedPrecession * 0.02)
        << "measured " << meanPrecession << " rad/orbit, expected " << expectedPrecession;
}

TEST(PhysicsGravity, PerihelionPrecessionPerOrbitMatchesTheClosedForm) {
    // Direct check of the formula itself, independent of the integration
    // test above: same (gm, a, e) as
    // RelativisticNBodySystemPerihelionPrecessionMatchesTheAnalyticRate,
    // plus a second, unrelated case, both computed by hand from
    // `6 pi GM / (c^2 a (1 - e^2))` rather than by calling the function
    // under test twice.
    const double c = ysq::constants::speedOfLight.value();

    {
        const double gm = 5.0e14;
        const double rs = 2.0 * gm / (c * c);
        const double a = 200.0 * rs;
        const double e = 0.4;
        const double expected = 6.0 * ysq::kPi<double> * gm / (c * c * a * (1.0 - e * e));
        EXPECT_NEAR(ysq::perihelionPrecessionPerOrbit(gm, a, e), expected, expected * 1e-12);
    }
    {
        const double gm = 1.32712440018e20;  // the Sun's own, real GM
        const double a = 5.7909e10;          // Mercury's real semi-major axis
        const double e = 0.2056;             // Mercury's real eccentricity
        const double expected = 6.0 * ysq::kPi<double> * gm / (c * c * a * (1.0 - e * e));
        EXPECT_NEAR(ysq::perihelionPrecessionPerOrbit(gm, a, e), expected, expected * 1e-12);
    }
}

TEST(PhysicsGravity, RelativisticNBodyJerkSystemWithNoPrimariesMatchesPlainNewtonianJerk) {
    // Same composition check RelativisticNBodySystem already has, one level
    // up: every primaryIndex negative must degenerate to exactly what
    // NewtonianJerkField alone computes for both acceleration and jerk.
    const std::vector<Body> bodies{
        makeBody(2.0e30, Vec3{0.0, 0.0, 0.0}, Vec3{1.0e2, -5.0e1, 0.0}),
        makeBody(6.0e24, Vec3{1.5e11, 0.0, 0.0}, Vec3{0.0, 3.0e4, 0.0}),
        makeBody(5.0e20, Vec3{-2.0e11, 1.0e10, 0.0}, Vec3{1.0e3, -2.0e4, 5.0e2})};

    const ysq::NewtonianJerkField newtonianJerk(bodies);
    const ysq::RelativisticNBodyJerkSystem relativisticJerk(bodies,
                                                            std::vector<int>{-1, -1, -1});

    const ysq::NBodyState positions = ysq::positionsOf(bodies);
    const ysq::NBodyState velocities = ysq::velocitiesOf(bodies);

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const auto [expectedAcceleration, expectedJerk] =
            newtonianJerk(i, positions, velocities);
        const auto [actualAcceleration, actualJerk] =
            relativisticJerk(i, positions, velocities);

        EXPECT_VEC_NEAR(actualAcceleration, expectedAcceleration,
                       length(expectedAcceleration) * 1e-12)
            << "body " << i;
        EXPECT_VEC_NEAR(actualJerk, expectedJerk, length(expectedJerk) * 1e-12)
            << "body " << i;
    }
}

TEST(PhysicsGravity, RelativisticJerkTermMatchesFiniteDifferenceOfTheAccelerationTerm) {
    // The mandatory safety net for a hand-derived closed-form time
    // derivative: relativisticJerkTerm's own algebra (product/quotient
    // rule through r, v, n, and the radial speed s = v.n) is complex
    // enough that a subtle sign or term dropped would not be obvious by
    // eye. Checked against a numerical derivative of the already-tested
    // postNewtonianCorrection (which calls the same internal acceleration
    // term this differentiates) along the two-body trajectory the jerk's
    // own derivation assumes, rather than trusted on the derivation alone.
    const double gm = 5.0e14;
    const Body source = makeBody(gm / ysq::constants::G.value(), Vec3{0.0, 0.0, 0.0});

    const Vec3 r0{3.0e10, 1.0e10, -2.0e9};
    const Vec3 v0{1.5e4, -3.0e4, 6.0e3};
    const double r0Mag = length(r0);
    // Two-body Newtonian relative acceleration -- the same scope the jerk
    // formula's own derivation uses, so the trajectory this is checked
    // along is second-order accurate, matching what a clean central
    // difference needs.
    const Vec3 a0 = r0 * (-gm / (r0Mag * r0Mag * r0Mag));

    constexpr double delta = 1.0e-3;
    const auto testParticleAt = [&](double t) {
        return makeBody(1.0, r0 + v0 * t + a0 * (0.5 * t * t), v0 + a0 * t);
    };

    const ysq::Acceleration3 accelerationBefore =
        ysq::postNewtonianCorrection(testParticleAt(-delta), source);
    const ysq::Acceleration3 accelerationAfter =
        ysq::postNewtonianCorrection(testParticleAt(delta), source);

    const Vec3 finiteDifferenceJerk =
        (accelerationAfter.value() - accelerationBefore.value()) / (2.0 * delta);
    const Vec3 analyticJerk = ysq::relativisticJerkTerm(r0, v0, gm);

    EXPECT_NEAR(distance(analyticJerk, finiteDifferenceJerk), 0.0,
                length(finiteDifferenceJerk) * 1e-3)
        << "analytic (" << analyticJerk.x << ", " << analyticJerk.y << ", " << analyticJerk.z
        << ") vs finite-difference (" << finiteDifferenceJerk.x << ", "
        << finiteDifferenceJerk.y << ", " << finiteDifferenceJerk.z << ")";
}

}  // namespace
