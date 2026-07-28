#include <Math/Integrators/RK4.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Optics/Lensing.hpp>
#include <Physics/Spacetime/Geodesic.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Units/Constants.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

/// Schwarzschild + Geodesic + Optics, validated against the two closed-form
/// results docs/physics.md and the module layout both name: the weak-field
/// deflection angle, and the perihelion precession that is Gravity's 1PN
/// rung's own prediction (physics_gravity.cpp), now reproduced from an
/// exact null and an exact timelike geodesic respectively rather than a
/// weak-field approximation to the force law.

namespace {

using ysq::PhaseState;
using ysq::Vec4;

using Phase = PhaseState<Vec4>;

constexpr double kPi = ysq::kPi<double>;

TEST(LensingDeflection, WeakFieldDeflectionMatchesTheAnalyticFormula) {
    const ysq::GravitationalParameter gm{5.0e14};
    const ysq::Schwarzschild schwarzschild{gm};
    const double rs = schwarzschild.schwarzschildRadius();

    const double impactParameter = 500.0 * rs;  // deep in the weak field
    const double startRadius = 20.0 * impactParameter;
    const double step = startRadius * 1.0e-3;
    constexpr std::size_t maxSteps = 5000;

    const Phase ray = ysq::schwarzschildRayFromImpactParameter(
        schwarzschild, impactParameter, startRadius);
    const double measured = ysq::deflectionAngle(schwarzschild, ray, impactParameter,
                                                 startRadius, step, maxSteps);

    ASSERT_FALSE(std::isnan(measured)) << "the ray must actually return to startRadius";

    const double expected = ysq::weakFieldDeflectionAngle(rs, impactParameter);
    EXPECT_NEAR(measured, expected, expected * 0.02)
        << "measured=" << measured << " expected=" << expected;
}

/// The exact conserved energy and angular momentum (per unit mass) of a
/// Schwarzschild equatorial timelike orbit with the given turning points,
/// found by requiring dr/dtau = 0 at both: see docs/physics.md.
struct TurningPointOrbit {
    double energy;
    double angularMomentum;
};

[[nodiscard]] TurningPointOrbit exactTurningPointOrbit(double schwarzschildRadius,
                                                       double speedOfLight,
                                                       double periapsis,
                                                       double apoapsis) {
    const double factorPeriapsis = 1.0 - schwarzschildRadius / periapsis;
    const double factorApoapsis = 1.0 - schwarzschildRadius / apoapsis;
    const double cSquared = speedOfLight * speedOfLight;

    const double angularMomentumSquared = cSquared * (factorApoapsis - factorPeriapsis) /
                                          (factorPeriapsis / (periapsis * periapsis) -
                                           factorApoapsis / (apoapsis * apoapsis));
    const double energySquared =
        factorPeriapsis * (cSquared + angularMomentumSquared / (periapsis * periapsis));

    return {std::sqrt(energySquared), std::sqrt(angularMomentumSquared)};
}

TEST(LensingDeflection, PerihelionPrecessionMatchesTheAnalyticRate) {
    const ysq::GravitationalParameter gm{5.0e14};
    const ysq::Schwarzschild schwarzschild{gm};
    const double rs = schwarzschild.schwarzschildRadius();
    const double c = ysq::constants::speedOfLight.value();

    constexpr double eccentricity = 0.4;
    const double semiMajorAxis = 200.0 * rs;  // weak field: r_s / a = 0.005
    const double periapsis = semiMajorAxis * (1.0 - eccentricity);
    const double apoapsis = semiMajorAxis * (1.0 + eccentricity);

    const TurningPointOrbit orbit = exactTurningPointOrbit(rs, c, periapsis, apoapsis);
    const double factorApoapsis = 1.0 - rs / apoapsis;

    const Phase start{Vec4{0.0, apoapsis, kPi / 2.0, 0.0},
                      Vec4{orbit.energy / factorApoapsis, 0.0, 0.0,
                           orbit.angularMomentum / (apoapsis * apoapsis)}};

    ASSERT_NEAR(
        ysq::metricProduct(schwarzschild, start.position, start.velocity, start.velocity),
        -(c * c), (c * c) * 1e-6)
        << "the constructed tangent must be timelike, normalized to proper time";

    const auto system = ysq::geodesicSystem(schwarzschild);
    ysq::Rk4Stepper<Phase> stepper;

    const double newtonianPeriod =
        2.0 * kPi * std::sqrt(semiMajorAxis * semiMajorAxis * semiMajorAxis / gm.value());
    const double step = newtonianPeriod / 20000.0;
    constexpr int targetOrbits = 6;
    const std::size_t maxSteps =
        static_cast<std::size_t>(newtonianPeriod / step) * (targetOrbits + 2);

    std::vector<double> periapsisAzimuths;
    Phase state = start;
    Phase next = start;
    double previousUr = state.velocity.y;
    double previousAzimuth = state.position.w;

    for (std::size_t i = 0;
         i < maxSteps && static_cast<int>(periapsisAzimuths.size()) < targetOrbits; ++i) {
        stepper.step(system, static_cast<double>(i) * step, state, step, next);
        const double ur = next.velocity.y;

        if (previousUr < 0.0 && ur >= 0.0) {
            const double t = -previousUr / (ur - previousUr);
            periapsisAzimuths.push_back(previousAzimuth +
                                        t * (next.position.w - previousAzimuth));
        }

        previousUr = ur;
        previousAzimuth = next.position.w;
        state = next;
    }

    ASSERT_GE(periapsisAzimuths.size(), 2u)
        << "not enough periapsis passages were observed";

    double meanPrecession = 0.0;
    for (std::size_t i = 1; i < periapsisAzimuths.size(); ++i) {
        meanPrecession += periapsisAzimuths[i] - periapsisAzimuths[i - 1] - 2.0 * kPi;
    }
    meanPrecession /= static_cast<double>(periapsisAzimuths.size() - 1);

    const double analyticPrecession =
        6.0 * kPi * gm.value() /
        (c * c * semiMajorAxis * (1.0 - eccentricity * eccentricity));

    EXPECT_GT(meanPrecession, 0.0);
    EXPECT_NEAR(meanPrecession, analyticPrecession, analyticPrecession * 0.05)
        << "measured=" << meanPrecession << " analytic=" << analyticPrecession;
}

}  // namespace
