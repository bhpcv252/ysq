#include <Math/Integrators/RK4.hpp>
#include <Math/ODE.hpp>
#include <Math/Statistics.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Spacetime/Geodesic.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Minkowski.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Units/Constants.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

using ysq::PhaseState;
using ysq::Vec4;

using Phase = PhaseState<Vec4>;

constexpr double kPi = ysq::kPi<double>;
constexpr double kTau = ysq::kTau<double>;

TEST(SpacetimeGeodesic, MinkowskiGeodesicsAreStraightLinesAtConstantVelocity) {
    // Gamma is exactly zero everywhere in flat spacetime, so the geodesic
    // equation reduces to d^2x/dlambda^2 = 0: RK4 integrates that exactly,
    // regardless of step size, since every term past the first in its
    // truncation error involves a derivative of the (identically zero)
    // acceleration.
    const ysq::Minkowski flat;
    const auto system = ysq::geodesicSystem(flat);

    const double c = ysq::constants::speedOfLight.value();
    const Phase start{Vec4{0.0, 0.0, 0.0, 0.0}, Vec4{c, 0.3 * c, -0.2 * c, 0.1 * c}};

    ysq::Rk4Stepper<Phase> stepper;
    constexpr double lambda = 100.0;
    constexpr std::size_t steps = 40;
    const double h = lambda / static_cast<double>(steps);

    Phase state = start;
    Phase next = start;
    for (std::size_t i = 0; i < steps; ++i) {
        stepper.step(system, static_cast<double>(i) * h, state, h, next);
        state = next;
    }

    const Vec4 expectedPosition = start.position + start.velocity * lambda;
    EXPECT_VEC_NEAR(state.position, expectedPosition, 1e-6);
    EXPECT_VEC_NEAR(state.velocity, start.velocity, 1e-9);
}

TEST(SpacetimeGeodesic, SchwarzschildPhotonSphereOrbitStaysCircular) {
    // The photon sphere, r = 1.5 r_s: the unique radius in the equatorial
    // plane where a null geodesic with no radial velocity has no radial
    // acceleration either, so it orbits at constant r. It is an unstable
    // equilibrium, which is why this checks one revolution rather than
    // many.
    const ysq::GravitationalParameter gm{1.0e15};
    const ysq::Schwarzschild schwarzschild{gm};
    const double rs = schwarzschild.schwarzschildRadius();
    const double rPhoton = 1.5 * rs;

    // Null normalization at constant r, polar = pi/2, with only T and
    // azimuth components: -(1 - r_s/r) uT^2 + r^2 uPhi^2 = 0.
    const double factor = 1.0 - rs / rPhoton;
    const double uT = 1.0;
    const double uPhi = std::sqrt(factor) / rPhoton;

    const Phase start{Vec4{0.0, rPhoton, kPi / 2.0, 0.0}, Vec4{uT, 0.0, 0.0, uPhi}};
    ASSERT_NEAR(
        ysq::metricProduct(schwarzschild, start.position, start.velocity, start.velocity),
        0.0, 1e-9)
        << "the constructed tangent must actually be null";

    const auto system = ysq::geodesicSystem(schwarzschild);
    ysq::Rk4Stepper<Phase> stepper;

    const double totalLambda = kTau / uPhi;  // one full revolution in azimuth
    constexpr std::size_t steps = 20000;
    const double h = totalLambda / static_cast<double>(steps);

    ysq::RunningStatistics<double> radius;
    Phase state = start;
    Phase next = start;
    radius.add(state.position.y);
    for (std::size_t i = 0; i < steps; ++i) {
        stepper.step(system, static_cast<double>(i) * h, state, h, next);
        state = next;
        radius.add(state.position.y);
    }

    EXPECT_NEAR(radius.mean(), rPhoton, rPhoton * 1e-3);
    EXPECT_LT(radius.range() / rPhoton, 1e-3)
        << "an unstable orbit is still expected to hold for one revolution";

    // Azimuth should have advanced by very close to a full turn.
    EXPECT_NEAR(state.position.w, kTau, kTau * 1e-3);
}

}  // namespace
