#include <Physics/Gravity/Kepler.hpp>

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ysq::KeplerStateVector;
using ysq::keplerMeanMotion;
using ysq::keplerOrbitalPeriod;
using ysq::OrbitalElements;
using ysq::OrbitalElementsAtEpoch;
using ysq::stateVectorAtTime;
using ysq::stateVectorFromElements;
using ysq::trueAnomalyFromMeanAnomaly;
using ysq::radians;

constexpr double kGm = 1.0;  // unit gravitational parameter; only shapes matter here

}  // namespace

TEST(PhysicsGravityKepler, ReferenceCaseAtPeriapsisWithNoRotationIsAlongThePlusXAxis) {
    // No inclination, node or argument of periapsis: the perifocal frame
    // *is* the reference frame, and at trueAnomaly = 0 (periapsis) the
    // textbook formula's own r = a(1-e), v purely tangential (+y) result
    // should come straight through unrotated.
    const OrbitalElements elements{/*a=*/2.0, /*e=*/0.5, 0.0, 0.0, 0.0, /*nu=*/0.0};
    const KeplerStateVector state = stateVectorFromElements(elements, kGm);

    const double periapsis = elements.semiMajorAxis * (1.0 - elements.eccentricity);
    EXPECT_NEAR(state.position.x, periapsis, 1e-12);
    EXPECT_NEAR(state.position.y, 0.0, 1e-12);
    EXPECT_NEAR(state.position.z, 0.0, 1e-12);

    EXPECT_NEAR(state.velocity.x, 0.0, 1e-12);
    EXPECT_GT(state.velocity.y, 0.0) << "prograde motion is +y in the perifocal frame";
    EXPECT_NEAR(state.velocity.z, 0.0, 1e-12);
}

TEST(PhysicsGravityKepler, ZeroInclinationKeepsTheOrbitExactlyInTheReferencePlane) {
    for (double nu = 0.0; nu < ysq::kTau<double>; nu += 0.3) {
        for (double node = 0.0; node < ysq::kTau<double>; node += 0.9) {
            for (double periapsisArg = 0.0; periapsisArg < ysq::kTau<double>;
                periapsisArg += 0.9) {
                const OrbitalElements elements{1.0, 0.3, 0.0, node, periapsisArg, nu};
                const KeplerStateVector state = stateVectorFromElements(elements, kGm);
                EXPECT_NEAR(state.position.z, 0.0, 1e-10);
                EXPECT_NEAR(state.velocity.z, 0.0, 1e-10);
            }
        }
    }
}

TEST(PhysicsGravityKepler, PeriapsisAndApoapsisDistancesMatchTheEccentricFormula) {
    const double a = 3.0;
    const double e = 0.4;

    const OrbitalElements atPeriapsis{a, e, radians(20.0), radians(50.0), radians(10.0), 0.0};
    const KeplerStateVector periapsisState = stateVectorFromElements(atPeriapsis, kGm);
    EXPECT_NEAR(length(periapsisState.position), a * (1.0 - e), 1e-9);

    OrbitalElements atApoapsis = atPeriapsis;
    atApoapsis.trueAnomaly = ysq::kPi<double>;
    const KeplerStateVector apoapsisState = stateVectorFromElements(atApoapsis, kGm);
    EXPECT_NEAR(length(apoapsisState.position), a * (1.0 + e), 1e-9);
}

TEST(PhysicsGravityKepler, VisVivaHoldsAtEveryPointOnTheOrbit) {
    // v^2 = gm (2/r - 1/a): a resolution- and orientation-independent
    // energy check that has to hold regardless of the rotation angles,
    // since only the shape (a, e) and the true anomaly determine speed.
    const double gm = 4.0;
    const double a = 2.5;
    const double e = 0.6;

    for (double nu = 0.0; nu < ysq::kTau<double>; nu += 0.4) {
        const OrbitalElements elements{a, e, radians(15.0), radians(70.0), radians(200.0), nu};
        const KeplerStateVector state = stateVectorFromElements(elements, gm);

        const double r = length(state.position);
        const double expectedSpeedSquared = gm * (2.0 / r - 1.0 / a);
        EXPECT_NEAR(lengthSquared(state.velocity), expectedSpeedSquared, 1e-9)
            << "at true anomaly " << nu;
    }
}

TEST(PhysicsGravityKepler, SpecificAngularMomentumMagnitudeIsConstantAroundTheOrbit) {
    // |r x v| = sqrt(gm a (1 - e^2)), independent of true anomaly and of
    // every rotation angle: the orbit's own shape sets it, nothing else.
    const double gm = 2.0;
    const double a = 1.6;
    const double e = 0.3;
    const double expected = std::sqrt(gm * a * (1.0 - e * e));

    for (double nu = 0.0; nu < ysq::kTau<double>; nu += 0.5) {
        const OrbitalElements elements{a, e, radians(33.0), radians(80.0), radians(120.0), nu};
        const KeplerStateVector state = stateVectorFromElements(elements, gm);
        const double h = length(cross(state.position, state.velocity));
        EXPECT_NEAR(h, expected, 1e-9) << "at true anomaly " << nu;
    }
}

TEST(PhysicsGravityKepler, InclinationIsExactlyTheAngleBetweenAngularMomentumAndThePole) {
    // The angle between the orbital angular momentum vector (r x v, the
    // orbit's own pole) and +z is `inclination` by definition, regardless
    // of the node or argument of periapsis -- the one rotation-angle check
    // that does not depend on trusting the perifocal formula's own sign
    // convention for anything else.
    const double a = 1.0;
    const double e = 0.2;

    for (double inclinationDeg : {0.0, 15.0, 45.0, 90.0, 135.0, 179.0}) {
        const OrbitalElements elements{a, e, radians(inclinationDeg), radians(40.0),
                                       radians(65.0), radians(25.0)};
        const KeplerStateVector state = stateVectorFromElements(elements, kGm);
        const ysq::Vec3 h = cross(state.position, state.velocity);
        const double cosAngle = h.z / length(h);
        EXPECT_NEAR(cosAngle, std::cos(radians(inclinationDeg)), 1e-9)
            << "at inclination " << inclinationDeg << " degrees";
    }
}

TEST(PhysicsGravityKepler, MeanAnomalyZeroAndPiMapToThemselvesAtAnyEccentricity) {
    // Periapsis and apoapsis are the two points where mean, eccentric and
    // true anomaly all coincide exactly, for any eccentricity: a
    // closed-form check independent of the solver's own iteration.
    for (double e : {0.0, 0.3, 0.7, 0.95}) {
        EXPECT_NEAR(trueAnomalyFromMeanAnomaly(0.0, e), 0.0, 1e-12) << "at e=" << e;
        EXPECT_NEAR(trueAnomalyFromMeanAnomaly(ysq::kPi<double>, e), ysq::kPi<double>, 1e-10)
            << "at e=" << e;
    }
}

TEST(PhysicsGravityKepler, MeanAnomalyEqualsTrueAnomalyExactlyForACircularOrbit) {
    // e = 0 makes Kepler's equation M = E trivially, and true anomaly = E
    // for a circular orbit too, so mean and true anomaly must agree exactly
    // at every angle, not just periapsis/apoapsis.
    for (double m = -3.0; m <= 3.0; m += 0.5) {
        EXPECT_NEAR(trueAnomalyFromMeanAnomaly(m, 0.0), m, 1e-12) << "at M=" << m;
    }
}

TEST(PhysicsGravityKepler, TrueAnomalyRoundTripsThroughMeanAnomaly) {
    // Independent of trueAnomalyFromMeanAnomaly itself: convert a chosen
    // true anomaly to eccentric anomaly by the inverse half-angle relation,
    // then to mean anomaly by Kepler's equation directly (M = E - e sin E),
    // and check the solver recovers the true anomaly it started from.
    for (double e : {0.0, 0.2, 0.5, 0.8, 0.99}) {
        for (double nu0 = -3.0; nu0 <= 3.0; nu0 += 0.4) {
            const double eccentricAnomaly =
                2.0 * std::atan2(std::sqrt(1.0 - e) * std::sin(nu0 / 2.0),
                                 std::sqrt(1.0 + e) * std::cos(nu0 / 2.0));
            const double meanAnomaly = eccentricAnomaly - e * std::sin(eccentricAnomaly);

            EXPECT_NEAR(trueAnomalyFromMeanAnomaly(meanAnomaly, e), nu0, 1e-9)
                << "at e=" << e << ", true anomaly=" << nu0;
        }
    }
}

TEST(PhysicsGravityKepler, MeanMotionMatchesTheTextbookFormula) {
    const double gm = 4.0;
    const double a = 3.0;
    EXPECT_NEAR(keplerMeanMotion(gm, a), std::sqrt(gm / (a * a * a)), 1e-12);
}

TEST(PhysicsGravityKepler, OrbitalPeriodIsTauOverMeanMotion) {
    const double gm = 4.0;
    const double a = 3.0;
    EXPECT_NEAR(keplerOrbitalPeriod(gm, a), ysq::kTau<double> / keplerMeanMotion(gm, a), 1e-12);
}

TEST(PhysicsGravityKepler, OrbitalPeriodMatchesEarthsRealYearAroundTheSun) {
    // A real, independently known number, not just internal self-consistency
    // with keplerMeanMotion: GM_sun and 1 AU give back a period within a
    // fraction of a percent of the real length of a year in seconds.
    const double gmSun = 1.32712440018e20;  // m^3/s^2
    const double oneAu = 1.495978707e11;    // m
    const double secondsPerJulianYear = 365.25 * 86400.0;

    EXPECT_NEAR(keplerOrbitalPeriod(gmSun, oneAu), secondsPerJulianYear,
               secondsPerJulianYear * 1e-3);
}

TEST(PhysicsGravityKepler, StateVectorAtTimeZeroElapsedMatchesStateVectorFromElements) {
    // elapsedSeconds = 0 must reproduce stateVectorFromElements exactly
    // (mean motion times zero is zero, precession times zero is zero): the
    // one point where both functions describe the same instant.
    const double gm = 3.5;
    const OrbitalElementsAtEpoch elements{/*a=*/2.0,
                                          /*e=*/0.4,
                                          radians(20.0),
                                          radians(50.0),
                                          radians(10.0),
                                          /*meanAnomalyAtEpoch=*/radians(75.0),
                                          /*precessionRatePerSecond=*/0.02};
    const KeplerStateVector atZero = stateVectorAtTime(elements, gm, 0.0);

    OrbitalElements atEpoch{};
    atEpoch.semiMajorAxis = elements.semiMajorAxis;
    atEpoch.eccentricity = elements.eccentricity;
    atEpoch.inclination = elements.inclination;
    atEpoch.longitudeOfAscendingNode = elements.longitudeOfAscendingNode;
    atEpoch.argumentOfPeriapsis = elements.argumentOfPeriapsis;
    atEpoch.trueAnomaly =
        trueAnomalyFromMeanAnomaly(elements.meanAnomalyAtEpoch, elements.eccentricity);
    const KeplerStateVector expected = stateVectorFromElements(atEpoch, gm);

    EXPECT_NEAR(atZero.position.x, expected.position.x, 1e-9);
    EXPECT_NEAR(atZero.position.y, expected.position.y, 1e-9);
    EXPECT_NEAR(atZero.position.z, expected.position.z, 1e-9);
    EXPECT_NEAR(atZero.velocity.x, expected.velocity.x, 1e-9);
    EXPECT_NEAR(atZero.velocity.y, expected.velocity.y, 1e-9);
    EXPECT_NEAR(atZero.velocity.z, expected.velocity.z, 1e-9);
}

TEST(PhysicsGravityKepler, StateVectorAtTimeReturnsToItsStartAfterOneFullPeriod) {
    // No precession: a full period's own elapsed time must land exactly
    // back where it started, regardless of how large that elapsed time is
    // in absolute terms -- a direct check that propagation itself, not just
    // the shape conversion, is correct.
    const double gm = 4.0;
    const double a = 3.0;
    const double period = keplerOrbitalPeriod(gm, a);

    const OrbitalElementsAtEpoch elements{a, 0.5, radians(15.0), radians(80.0),
                                          radians(200.0), radians(30.0), 0.0};
    const KeplerStateVector atStart = stateVectorAtTime(elements, gm, 0.0);
    const KeplerStateVector afterOnePeriod = stateVectorAtTime(elements, gm, period);
    const KeplerStateVector afterManyPeriods = stateVectorAtTime(elements, gm, period * 137.0);

    EXPECT_NEAR(length(afterOnePeriod.position - atStart.position), 0.0, a * 1e-9);
    EXPECT_NEAR(length(afterManyPeriods.position - atStart.position), 0.0, a * 1e-6)
        << "137 periods forward, still needs to land back at the start";
}

TEST(PhysicsGravityKepler, StateVectorAtTimeAppliesPrecessionToTheArgumentOfPeriapsis) {
    // A nonzero precession rate must rotate the ellipse itself (where
    // periapsis sits), not just move the body along a fixed one: after
    // exactly one period the body is not back at its starting position,
    // but at the *new* periapsis-relative point the rotated
    // argumentOfPeriapsis implies.
    const double gm = 4.0;
    const double a = 3.0;
    const double e = 0.5;
    const double period = keplerOrbitalPeriod(gm, a);
    const double precessionRatePerSecond = 0.01;  // rad/s, deliberately large for a clean signal

    const OrbitalElementsAtEpoch elements{a,
                                          e,
                                          0.0,
                                          0.0,
                                          /*argumentOfPeriapsis=*/0.0,
                                          /*meanAnomalyAtEpoch=*/0.0,
                                          precessionRatePerSecond};
    const KeplerStateVector afterOnePeriod = stateVectorAtTime(elements, gm, period);

    OrbitalElements expectedShape{};
    expectedShape.semiMajorAxis = a;
    expectedShape.eccentricity = e;
    expectedShape.argumentOfPeriapsis = precessionRatePerSecond * period;
    expectedShape.trueAnomaly = 0.0;  // one full period forward: back to periapsis itself
    const KeplerStateVector expected = stateVectorFromElements(expectedShape, gm);

    EXPECT_NEAR(afterOnePeriod.position.x, expected.position.x, a * 1e-6);
    EXPECT_NEAR(afterOnePeriod.position.y, expected.position.y, a * 1e-6);
}
