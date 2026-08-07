#include <Applications/KeplerSolarSystem/Scenario.hpp>

#include <Core/Timer.hpp>

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <Physics/Gravity/Kepler.hpp>
#include <Physics/Gravity/PostNewtonian.hpp>

#include <Units/Constants.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

/// KeplerSolarSystem's own scenario, propagated headless, checked against
/// the invariants a correct closed-form Kepler propagator must have: every
/// body stays within its own real perihelion/aphelion bounds forever (not
/// just near t=0), Mercury's own perihelion really precesses at the
/// closed-form GR rate over many real orbits, and -- the direct regression
/// test for why this app exists at all -- an enormous simulated-time jump
/// still produces a finite result, instantly, rather than the runaway
/// catch-up cost a real n-body integrator (Applications::SolarSystem's own)
/// would have at the same requested speed.

namespace {

using namespace ysq::kepler_solar_system;

const ysq::applications::KeplerCatalogBody& find(const Scenario& scenario,
                                                  const std::string& name) {
    for (const auto& body : scenario.bodies) {
        if (body.name == name) {
            return body;
        }
    }
    ADD_FAILURE() << "no body named '" << name << "'";
    return scenario.bodies.front();
}

}  // namespace

TEST(KeplerSolarSystemE2E, EveryRealBodyStaysWithinItsOwnPerihelionAphelionBounds) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    // Planets, dwarf planets, and a couple of real moons: every kind of
    // parent-child relationship this catalog has.
    const std::vector<std::string> sample{"Mercury", "Earth",  "Jupiter", "Neptune",
                                          "Pluto",   "Eris",   "Moon",    "Io"};

    // Simulated times spanning from "now" out to roughly 50,000 real years
    // forward and backward -- the kind of jump 1 year/sec sustained for
    // hours would actually reach, and exactly what an n-body integrator
    // cannot do cheaply.
    const std::vector<double> times{0.0, 1.0e9, -1.0e9, 1.0e12, -1.0e12, 1.5e15};

    for (const std::string& name : sample) {
        const ysq::applications::KeplerCatalogBody& body = find(*scenario, name);
        ASSERT_TRUE(body.elements.has_value()) << name;
        const double a = body.elements->semiMajorAxis;
        const double e = body.elements->eccentricity;
        const double lo = a * (1.0 - e) * 0.999;
        const double hi = a * (1.0 + e) * 1.001;

        for (double t : times) {
            const ysq::KeplerStateVector local =
                ysq::stateVectorAtTime(*body.elements, body.parentGm, t);
            const double distance = length(local.position);
            ASSERT_TRUE(std::isfinite(distance)) << name << " at t=" << t;
            EXPECT_GE(distance, lo) << name << " at t=" << t;
            EXPECT_LE(distance, hi) << name << " at t=" << t;
        }
    }
}

TEST(KeplerSolarSystemE2E, AMoonWithNoPrecessionReturnsExactlyToItsStartAfterOnePeriod) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    const ysq::applications::KeplerCatalogBody& moon = find(*scenario, "Moon");
    ASSERT_TRUE(moon.elements.has_value());
    ASSERT_DOUBLE_EQ(moon.elements->precessionRatePerSecond, 0.0)
        << "this test needs a body whose ellipse does not itself rotate, to isolate "
           "period round-tripping from precession";

    const double a = moon.elements->semiMajorAxis;
    const double period = ysq::keplerOrbitalPeriod(moon.parentGm, a);

    const ysq::KeplerStateVector start =
        ysq::stateVectorAtTime(*moon.elements, moon.parentGm, 0.0);
    const ysq::KeplerStateVector afterOnePeriod =
        ysq::stateVectorAtTime(*moon.elements, moon.parentGm, period);
    const ysq::KeplerStateVector after50Periods =
        ysq::stateVectorAtTime(*moon.elements, moon.parentGm, period * 50.0);

    EXPECT_NEAR(length(afterOnePeriod.position - start.position), 0.0, a * 1e-6);
    EXPECT_NEAR(length(after50Periods.position - start.position), 0.0, a * 1e-4);
}

TEST(KeplerSolarSystemE2E, MercuryPerihelionPrecessesByTheRealAnalyticRateOverManyOrbits) {
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    const ysq::applications::KeplerCatalogBody& mercury = find(*scenario, "Mercury");
    ASSERT_TRUE(mercury.elements.has_value());
    const double a = mercury.elements->semiMajorAxis;
    const double e = mercury.elements->eccentricity;
    const double gmSun = mercury.parentGm;
    const double meanMotion = ysq::keplerMeanMotion(gmSun, a);
    const double period = ysq::keplerOrbitalPeriod(gmSun, a);

    // Time of the first periapsis passage after t=0 (mean anomaly wraps to
    // exactly 0 there), so both samples below are genuinely at periapsis,
    // not at whatever true anomaly Mercury happens to start the scenario
    // at.
    double meanAnomalyAtEpoch = std::fmod(mercury.elements->meanAnomalyAtEpoch, ysq::kTau<double>);
    if (meanAnomalyAtEpoch < 0.0) {
        meanAnomalyAtEpoch += ysq::kTau<double>;
    }
    const double firstPeriapsisTime = (ysq::kTau<double> - meanAnomalyAtEpoch) / meanMotion;

    constexpr int kOrbits = 50;
    const double laterPeriapsisTime = firstPeriapsisTime + static_cast<double>(kOrbits) * period;

    const ysq::KeplerStateVector firstPeriapsis =
        ysq::stateVectorAtTime(*mercury.elements, gmSun, firstPeriapsisTime);
    const ysq::KeplerStateVector laterPeriapsis =
        ysq::stateVectorAtTime(*mercury.elements, gmSun, laterPeriapsisTime);

    // Precession rotates the periapsis direction strictly within the
    // orbital plane; the angle between two periapsis-direction vectors is
    // exactly that rotation regardless of the orbit's own 3D tilt
    // (inclination and node), so this holds for Mercury's real,
    // non-coplanar elements without needing to project into its orbital
    // plane by hand first.
    const double cosAngle =
        dot(firstPeriapsis.position, laterPeriapsis.position) /
        (length(firstPeriapsis.position) * length(laterPeriapsis.position));
    const double measuredAngle = std::acos(std::clamp(cosAngle, -1.0, 1.0));

    const double expectedAngle =
        static_cast<double>(kOrbits) * ysq::perihelionPrecessionPerOrbit(gmSun, a, e);

    EXPECT_NEAR(measuredAngle, expectedAngle, expectedAngle * 0.01)
        << "measured " << measuredAngle << " rad over " << kOrbits
        << " orbits, expected " << expectedAngle;
}

TEST(KeplerSolarSystemE2E, AnEnormousSimulatedTimeJumpStillProducesAFiniteResultInstantly) {
    // The actual regression test for why this app exists: Applications::
    // SolarSystem's individual-timestep n-body scheduler would spend real
    // wall-clock time catching up to a jump this size, proportional to how
    // large it is. A closed-form Kepler evaluation does not: every body's
    // position at any one instant costs the same handful of Newton-Raphson
    // iterations regardless of how far past the last frame that instant is.
    const std::optional<Scenario> scenario = makeScenario();
    ASSERT_TRUE(scenario.has_value());

    // Roughly what 1 year/sec sustained for six real hours reaches.
    constexpr double kHugeSimulatedTime = 6.0 * 3600.0 * 365.25 * 24.0 * 3600.0;

    ysq::Timer timer;
    timer.lap();
    for (const auto& body : scenario->bodies) {
        if (!body.elements.has_value()) {
            continue;
        }
        const ysq::KeplerStateVector state = ysq::stateVectorAtTime(
            *body.elements, body.parentGm, kHugeSimulatedTime);
        ASSERT_TRUE(std::isfinite(length(state.position))) << body.name;
        ASSERT_TRUE(std::isfinite(length(state.velocity))) << body.name;
    }
    const double elapsedSeconds = timer.lap().count();

    // Generous (a real machine does every body here in well under a
    // millisecond): this is a regression guard against the jump itself
    // costing something proportional to its size, not a tight performance
    // benchmark.
    EXPECT_LT(elapsedSeconds, 0.25)
        << "evaluating every body at a single huge simulated time took " << elapsedSeconds
        << "s -- should cost the same as evaluating at t=0";
}
