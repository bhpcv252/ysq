#include <Applications/LunarEclipse/Scenario.hpp>

#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
#include <Math/Vector3.hpp>

#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Physics/Mechanics/RigidBody.hpp>
#include <Physics/Optics/Illumination.hpp>
#include <Physics/Optics/RefractiveMedium.hpp>

#include <Units/Length.hpp>
#include <Units/Mass.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

/// Two invariants tests/unit already can't reach because they need the
/// actual shipped scenario and a real integrated run: that Scenario.cpp's
/// real masses, radii and orbital elements, propagated by the real n-body
/// stepper with Earth's J2 and rotation active, conserve momentum
/// (tests/unit/physics_gravity.cpp already proves the law does in the
/// abstract; this asks whether the specific data here does), and that
/// Optics/Illumination.hpp, pointed at a deliberately constructed
/// near-syzygy configuration, actually comes out dimmer and reddened, an
/// invariant of what totality *is*, not a fixed expected color.

namespace {

using namespace ysq::lunar_eclipse;

ysq::Vec3 totalMomentum(const std::vector<ysq::Body>& bodies) {
    ysq::Vec3 total = ysq::Vec3::zero();
    for (const ysq::Body& body : bodies) {
        total += body.momentum.value();
    }
    return total;
}

}  // namespace

TEST(LunarEclipseIntegration, RealScenarioConservesMomentumOverSeveralMoonOrbits) {
    const Scenario scenario = makeScenario();
    std::vector<ysq::Body> bodies = scenario.allBodies();
    constexpr std::size_t kEarth = 1;
    constexpr std::size_t kMoon = 2;

    const double gmEarth = ysq::constants::G.value() * bodies[kEarth].mass.value();
    const double moonRadius =
        length(bodies[kMoon].position.value() - bodies[kEarth].position.value());
    const double moonPeriod = 2.0 * ysq::kPi<double> *
                              std::sqrt(moonRadius * moonRadius * moonRadius / gmEarth);

    const double step = moonPeriod / 4000.0;
    const int totalSteps = static_cast<int>(20.0 * moonPeriod / step);
    const ysq::Length softening{1.0e6};

    const ysq::Vec3 initialMomentum = totalMomentum(bodies);
    const double momentumScale = length(bodies[kMoon].momentum.value());

    ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
    double maxMomentumDeviation = 0.0;
    double minEarthMoonDistance = moonRadius;
    double maxEarthMoonDistance = moonRadius;

    for (int i = 0; i < totalSteps; ++i) {
        const ysq::NewtonianField field(bodies, softening);
        const ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                                     ysq::velocitiesOf(bodies)};
        ysq::PhaseState<ysq::NBodyState> next;
        stepper.step(field, static_cast<double>(i) * step, state, step, next);
        ysq::applyState(bodies, next.position, next.velocity);

        std::vector<ysq::Body> perturbersOfEarth;
        for (std::size_t b = 0; b < bodies.size(); ++b) {
            if (b != kEarth) {
                perturbersOfEarth.push_back(bodies[b]);
            }
        }
        ysq::stepRigidBody(bodies[kEarth], perturbersOfEarth, step);

        const ysq::Vec3 momentum = totalMomentum(bodies);
        maxMomentumDeviation =
            std::max(maxMomentumDeviation, length(momentum - initialMomentum));

        const double distance =
            length(bodies[kMoon].position.value() - bodies[kEarth].position.value());
        minEarthMoonDistance = std::min(minEarthMoonDistance, distance);
        maxEarthMoonDistance = std::max(maxEarthMoonDistance, distance);
    }

    EXPECT_LT(maxMomentumDeviation, momentumScale * 1e-6)
        << "total momentum is conserved structurally, including J2's own "
           "Newton's-third-law reaction";

    // A bounded-orbit sanity check standing in for full energy
    // conservation: the Moon's real orbit (a = 384,399 km, e = 0.0549) has
    // perigee/apogee at roughly 363,000 / 405,500 km; a correct integration
    // over 20 orbits should stay in that neighborhood, not spiral in or
    // fly apart.
    EXPECT_GT(minEarthMoonDistance, 3.5e8);
    EXPECT_LT(maxEarthMoonDistance, 4.2e8);
}

TEST(LunarEclipseIntegration, TotalityIsDimmerAndReddenedThanFullSunlight) {
    // A deliberately scripted near-syzygy configuration, not the real
    // scenario's own slow-to-arrive geometry: real Sun/Earth/Moon radii and
    // Earth's real atmosphere, Moon placed directly opposite the Sun,
    // deep in Earth's geometric shadow.
    const double earthRadius = 6.371e6;
    const double scaleHeight = 8434.5;

    ysq::RefractingOccluder earth;
    earth.center = ysq::Vec3{0.0, 0.0, 0.0};
    earth.opaqueRadius = earthRadius;
    earth.medium = ysq::RefractiveMedium{earthRadius, 2.9e-4, scaleHeight};
    earth.surfaceNumberDensity = 2.6868e25;
    earth.scatteringScaleHeight = scaleHeight;

    const ysq::Vec3 sourceCenter{1.496e11, 0.0, 0.0};
    const double sourceRadius = 6.957e8;
    const ysq::Vec3 target{-3.844e8, 0.0,
                           0.0};  // real Earth-Moon distance, opposite the Sun

    const std::array<double, 3> wavelengths{630.0e-9, 532.0e-9, 465.0e-9};
    const ysq::IlluminationResult result = ysq::illuminate(
        sourceCenter, sourceRadius, {}, &earth, target, wavelengths, 4, 800);

    const double totalTransmission =
        result.transmission.x + result.transmission.y + result.transmission.z;

    EXPECT_NEAR(result.geometricVisibility, 0.0, 1e-9)
        << "the Moon is fully within the geometric shadow here, no direct line of "
           "sight to any part of the Sun's disk";
    EXPECT_GT(totalTransmission, 0.0) << "some light still reaches it, bent through "
                                         "Earth's atmosphere";
    EXPECT_LT(result.transmission.x + result.transmission.y + result.transmission.z, 3.0)
        << "and it is far dimmer than full sunlight (transmission (1,1,1))";
    EXPECT_GT(result.transmission.x, result.transmission.z)
        << "and reddened: red survives the grazing path far better than blue";
}
