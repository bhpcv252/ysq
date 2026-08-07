#include <Physics/Mechanics/Hermite.hpp>

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <utility>

/// The test that decides whether the Hermite predictor-corrector is
/// actually implemented correctly, the same reasoning
/// tests/unit/math_integrators.cpp already documents for RK4/Verlet: a
/// wrong scheme does not crash, it just converges more slowly than
/// advertised, and only measuring the observed order catches that.

namespace {

using ysq::Vec3;

/// Two-body Kepler system: a fixed, dominant mass `gm` at the origin, the
/// evolving body's own acceleration and jerk as a function of its own
/// (position, velocity) -- the same closed-form-comparable problem
/// math_integrators.cpp's own order tests use, applied here to the
/// predictor-corrector cycle standalone, before any multi-body scheduling
/// is involved.
std::pair<Vec3, Vec3> keplerAccelerationAndJerk(const Vec3& position, const Vec3& velocity,
                                                double gm) {
    const double r2 = lengthSquared(position);
    const double r = std::sqrt(r2);
    const double r3 = r2 * r;
    const double r5 = r3 * r2;
    const double radialTerm = dot(position, velocity);

    const Vec3 acceleration = position * (-gm / r3);
    const Vec3 jerk = velocity * (-gm / r3) + position * (3.0 * gm * radialTerm / r5);
    return {acceleration, jerk};
}

/// Integrates a circular orbit for exactly one period at `steps` Hermite
/// steps, and returns the position error against the known closed form: a
/// circular orbit is back exactly where it started after one period.
double circularOrbitErrorAtStepCount(double gm, double r0, std::size_t steps) {
    const double omega = std::sqrt(gm / (r0 * r0 * r0));
    const double period = ysq::kTau<double> / omega;
    const double dt = period / static_cast<double>(steps);

    Vec3 position{r0, 0.0, 0.0};
    Vec3 velocity{0.0, r0 * omega, 0.0};
    auto [acceleration, jerk] = keplerAccelerationAndJerk(position, velocity, gm);

    for (std::size_t i = 0; i < steps; ++i) {
        const auto [predictedPosition, predictedVelocity] =
            ysq::hermitePredict(position, velocity, acceleration, jerk, dt);
        const auto [newAcceleration, newJerk] =
            keplerAccelerationAndJerk(predictedPosition, predictedVelocity, gm);
        const auto [correctedPosition, correctedVelocity] = ysq::hermiteCorrect(
            acceleration, jerk, newAcceleration, newJerk, dt, predictedPosition,
            predictedVelocity);
        position = correctedPosition;
        velocity = correctedVelocity;
        acceleration = newAcceleration;
        jerk = newJerk;
    }

    return distance(position, Vec3{r0, 0.0, 0.0});
}

}  // namespace

TEST(PhysicsMechanicsHermite, PredictorCorrectorHitsFourthOrderOnACircularOrbit) {
    constexpr double gm = 4.0;
    constexpr double r0 = 1.0;

    const double errorCoarse = circularOrbitErrorAtStepCount(gm, r0, 400);
    const double errorFine = circularOrbitErrorAtStepCount(gm, r0, 800);

    ASSERT_GT(errorCoarse, 0.0);
    ASSERT_GT(errorFine, 0.0);

    // 4th order: doubling the step count should cut the error by roughly
    // 2^4 = 16. Some slack for this being measured over a whole period
    // rather than a single infinitesimal step.
    const double observedOrder = std::log2(errorCoarse / errorFine);
    EXPECT_GT(observedOrder, 3.5);
    EXPECT_LT(observedOrder, 4.5);
}

TEST(PhysicsMechanicsHermite, TimestepShrinksWhereJerkIsLargeRelativeToAcceleration) {
    constexpr double eta = 0.01;
    constexpr double baseInterval = 100.0;

    // Same acceleration magnitude, very different jerk: the criterion must
    // respond to jerk, not to acceleration alone.
    const double calmDt =
        ysq::hermiteTimestep(Vec3{1.0, 0.0, 0.0}, Vec3{0.01, 0.0, 0.0}, eta, baseInterval);
    const double sharpDt =
        ysq::hermiteTimestep(Vec3{1.0, 0.0, 0.0}, Vec3{10.0, 0.0, 0.0}, eta, baseInterval);

    EXPECT_LT(sharpDt, calmDt);
    EXPECT_LE(calmDt, baseInterval);
    EXPECT_LE(sharpDt, baseInterval);
}

TEST(PhysicsMechanicsHermite, TimestepIsAPowerOfTwoFractionOfTheBaseInterval) {
    constexpr double baseInterval = 64.0;
    const double dt = ysq::hermiteTimestep(Vec3{5.0, 0.0, 0.0}, Vec3{3.0, 0.0, 0.0}, 0.01,
                                           baseInterval);

    ASSERT_GT(dt, 0.0);
    const double ratio = baseInterval / dt;
    const double log2Ratio = std::log2(ratio);
    EXPECT_NEAR(log2Ratio, std::round(log2Ratio), 1e-9);
}

TEST(PhysicsMechanicsHermite, TimestepFallsBackToBaseIntervalWhenJerkIsZero) {
    const double dt = ysq::hermiteTimestep(Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 0.0, 0.0}, 0.01,
                                           50.0);
    EXPECT_DOUBLE_EQ(dt, 50.0);
}

TEST(PhysicsMechanicsHermite, CurrentTimeAdvancesEvenWhenNoBodyIsDueYet) {
    // Regression test: currentTime() must track targetTime even when
    // advanceTo() does zero updates because nothing is due yet -- not stay
    // pinned at the last real update. A caller driving many small
    // advanceTo() calls (say, once per rendered frame) with a request far
    // smaller than any body's own step must still accumulate toward that
    // body's own due time, not spin forever at t=0. (Found the hard way:
    // a caller that clamps its own clock to currentTime() after every
    // call -- exactly the fix a real over-budget scenario needed -- froze
    // solid at any speed slow enough that nothing was ever due within a
    // single call.)
    const Vec3 acceleration{1.0, 0.0, 0.0};
    const Vec3 jerk{1.0e-3, 0.0, 0.0};  // tiny jerk -> a large step
    constexpr double eta = 0.01;
    constexpr double baseInterval = 1000.0;
    const double dt = ysq::hermiteTimestep(acceleration, jerk, eta, baseInterval);
    ASSERT_GT(dt, 1.0) << "the test needs a step meaningfully larger than the "
                          "per-call target increment below";

    ysq::NBodyState positions{Vec3::zero()};
    ysq::NBodyState velocities{Vec3::zero()};
    ysq::NBodyState accelerations{acceleration};
    ysq::NBodyState jerks{jerk};

    ysq::IndividualTimestepScheduler scheduler(positions, velocities, accelerations, jerks,
                                               0.0, eta, baseInterval);

    const auto constantJerkField = [&](std::size_t, const ysq::NBodyState&,
                                       const ysq::NBodyState&) {
        return std::pair<Vec3, Vec3>{acceleration, jerk};
    };

    double targetTime = 0.0;
    constexpr double perCallIncrement = 0.01;  // far smaller than dt
    for (int call = 0; call < 50; ++call) {
        targetTime += perCallIncrement;
        scheduler.advanceTo(constantJerkField, targetTime, 1000);
        EXPECT_NEAR(scheduler.currentTime(), targetTime, 1e-9)
            << "call " << call
            << ": currentTime() must track targetTime even though nothing is due yet";
    }
}
