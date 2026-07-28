#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Mechanics/Kinematics.hpp>
#include <Physics/Optics/FrequencyShift.hpp>
#include <Physics/Optics/Propagation.hpp>
#include <Physics/Spacetime/FLRW.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Minkowski.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Units/Constants.hpp>
#include <Units/Velocity.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ysq::Vec3;
using ysq::Vec4;

constexpr double kPi = ysq::kPi<double>;

TEST(OpticsFrequencyShift, LongitudinalDopplerMatchesTheRelativisticFormula) {
    const ysq::Minkowski flat;
    const double c = ysq::constants::speedOfLight.value();
    constexpr double beta = 0.4;

    const Vec4 emissionEvent{};
    const Vec4 k = ysq::nullTangent(flat, emissionEvent, Vec3{1.0, 0.0, 0.0});
    const Vec4 emitterVelocity{c, 0.0, 0.0, 0.0};

    // An observer receding along the photon's direction of travel. Flat
    // spacetime is homogeneous and the observer's four-velocity is constant
    // along an inertial worldline, so where exactly this event sits does
    // not matter for this check, only that k is unchanged (Minkowski
    // geodesics do not bend).
    const Vec4 observationEvent{10.0, 10.0, 0.0, 0.0};
    const ysq::Velocity3 observerSpeed{Vec3{beta * c, 0.0, 0.0}};
    const Vec4 observerVelocity = ysq::fourVelocity(observerSpeed).value();

    const double measured = ysq::frequencyShift(flat, emissionEvent, k, emitterVelocity,
                                                observationEvent, k, observerVelocity);
    const double expected = std::sqrt((1.0 - beta) / (1.0 + beta));

    EXPECT_NEAR(measured, expected, 1e-9);
}

TEST(OpticsFrequencyShift, RadialGravitationalRedshiftMatchesTheClosedForm) {
    const ysq::GravitationalParameter gm{5.0e14};
    const ysq::Schwarzschild schwarzschild{gm};
    const double rs = schwarzschild.schwarzschildRadius();

    const double r1 = 5.0 * rs;
    const Vec4 emissionEvent{0.0, r1, kPi / 2.0, 0.0};
    const Vec4 k = ysq::nullTangent(schwarzschild, emissionEvent, Vec3{1.0, 0.0, 0.0});
    const ysq::PhaseState<Vec4> start{emissionEvent, k};

    const ysq::PhaseState<Vec4> end =
        ysq::propagate(schwarzschild, start, 50.0 * rs, 5000);
    const double r2 = end.position.y;
    ASSERT_GT(r2, r1) << "the photon must actually have climbed outward";

    const Vec4 emitterVelocity =
        ysq::staticObserverFourVelocity(schwarzschild, emissionEvent);
    const Vec4 observerVelocity =
        ysq::staticObserverFourVelocity(schwarzschild, end.position);

    const double measured =
        ysq::frequencyShift(schwarzschild, emissionEvent, k, emitterVelocity,
                            end.position, end.velocity, observerVelocity);

    const double factor1 = 1.0 - rs / r1;
    const double factor2 = 1.0 - rs / r2;
    const double expected = std::sqrt(factor1 / factor2);

    EXPECT_NEAR(measured, expected, expected * 1e-4);
}

TEST(OpticsFrequencyShift, CosmologicalRedshiftMatchesTheScaleFactorRatio) {
    const double referenceTime = 1.0e17;
    const ysq::MatterDominatedFLRW cosmology{referenceTime, 0.0};

    const Vec4 emissionEvent{referenceTime, 1.0, kPi / 2.0, 0.0};
    const Vec4 k = ysq::nullTangent(cosmology, emissionEvent, Vec3{1.0, 0.0, 0.0});
    const ysq::PhaseState<Vec4> start{emissionEvent, k};

    const ysq::PhaseState<Vec4> end =
        ysq::propagate(cosmology, start, 2.0 * referenceTime, 5000);
    const double laterTime = end.position.x;
    ASSERT_GT(laterTime, referenceTime);

    const Vec4 emitterVelocity =
        ysq::staticObserverFourVelocity(cosmology, emissionEvent);
    const Vec4 observerVelocity =
        ysq::staticObserverFourVelocity(cosmology, end.position);

    const double measured =
        ysq::frequencyShift(cosmology, emissionEvent, k, emitterVelocity, end.position,
                            end.velocity, observerVelocity);

    // a(referenceTime) = 1 by construction, so the closed form 1 + z =
    // a(observed) / a(emitted) reduces to nu_obs / nu_emit = 1 / a(observed).
    const double scaleFactorAtObservation =
        std::pow(laterTime / referenceTime, 2.0 / 3.0);
    const double expected = 1.0 / scaleFactorAtObservation;

    EXPECT_NEAR(measured, expected, expected * 1e-3);
}

}  // namespace
