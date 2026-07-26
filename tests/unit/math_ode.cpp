#include <Math/ODE.hpp>

#include <Math/Integrators/Adaptive.hpp>
#include <Math/Integrators/Euler.hpp>
#include <Math/Integrators/RK4.hpp>
#include <Math/Integrators/Symplectic.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using ysq::PhaseState;
using ysq::StateVector;
using ysq::Vec3;

using ScalarPhase = PhaseState<double>;
using VectorPhase = PhaseState<Vec3>;

/// dy/dt = -y, whose solution is exp(-t).
constexpr auto decay = [](double, double y) { return -y; };

/// q'' = -q, the unit harmonic oscillator.
constexpr auto spring = [](double, double q) { return -q; };

// --- The state concept ------------------------------------------------------

TEST(MathOde, TheScalarOfAStateIsFoundThroughAnyNesting) {
    // The recursion is what makes a nested state work: a PhaseState of Vector3
    // is built out of doubles, and it is doubles that a step size multiplies.
    static_assert(std::is_same_v<ysq::StateScalarT<double>, double>);
    static_assert(std::is_same_v<ysq::StateScalarT<Vec3>, double>);
    static_assert(std::is_same_v<ysq::StateScalarT<ScalarPhase>, double>);
    static_assert(std::is_same_v<ysq::StateScalarT<VectorPhase>, double>);
    static_assert(std::is_same_v<ysq::StateScalarT<StateVector<double>>, double>);
    static_assert(std::is_same_v<ysq::StateScalarT<ysq::Vec3f>, float>);

    static_assert(ysq::OdeState<double>);
    static_assert(ysq::OdeState<Vec3>);
    static_assert(ysq::OdeState<ScalarPhase>);
    static_assert(ysq::OdeState<VectorPhase>);
    static_assert(ysq::OdeState<StateVector<double>>);
    static_assert(ysq::OdeState<PhaseState<StateVector<double>>>);

    static_assert(ysq::OdeSystem<decltype(decay), double>);
    static_assert(ysq::AccelerationField<decltype(spring), double>);
}

TEST(MathOde, PhaseStateIsAVectorSpace) {
    const VectorPhase a{Vec3{1.0, 2.0, 3.0}, Vec3{4.0, 5.0, 6.0}};
    const VectorPhase b{Vec3{-1.0, 0.5, 2.0}, Vec3{3.0, -2.0, 1.0}};

    EXPECT_EQ(a + b, b + a);
    EXPECT_EQ(a - a, VectorPhase{});
    EXPECT_EQ(-(-a), a);
    EXPECT_EQ(a * 1.0, a);
    EXPECT_EQ(2.0 * a, a * 2.0);
    EXPECT_EQ((a + b) * 3.0, a * 3.0 + b * 3.0);

    VectorPhase mutated = a;
    mutated += b;
    EXPECT_EQ(mutated, a + b);
    mutated = a;
    mutated *= 2.0;
    EXPECT_EQ(mutated, a * 2.0);

    mutated = a;
    mutated /= 2.0;
    EXPECT_EQ(mutated, a / 2.0);
    EXPECT_EQ(a / 2.0, a * 0.5);

    // Index 0 is the position, which is what the error norm walks into.
    EXPECT_EQ(a[0], a.position);
    EXPECT_EQ(a[1], a.velocity);
    EXPECT_EQ(VectorPhase::size(), 2u);
}

TEST(MathOde, StateVectorIsAVectorSpaceSizedAtRunTime) {
    const StateVector<double> a{1.0, 2.0, 3.0};
    const StateVector<double> b{4.0, 5.0, 6.0};

    EXPECT_EQ(a.size(), 3u);
    EXPECT_EQ(a[1], 2.0);
    EXPECT_FALSE(a.empty());
    EXPECT_TRUE(StateVector<double>{}.empty());
    EXPECT_EQ(StateVector<double>(4, 1.5).size(), 4u);
    EXPECT_EQ(StateVector<double>(4, 1.5)[3], 1.5);

    EXPECT_EQ(a + b, b + a);
    EXPECT_EQ(a - a, StateVector<double>(3, 0.0));
    EXPECT_EQ(a * 2.0, (StateVector<double>{2.0, 4.0, 6.0}));
    EXPECT_EQ(2.0 * a, a * 2.0);
    EXPECT_EQ(a / 2.0, (StateVector<double>{0.5, 1.0, 1.5}));
    EXPECT_EQ(-a, (StateVector<double>{-1.0, -2.0, -3.0}));

    StateVector<double> mutated = a;
    mutated += b;
    EXPECT_EQ(mutated, (StateVector<double>{5.0, 7.0, 9.0}));
}

TEST(MathOde, AsPhaseSystemPairsVelocityWithAcceleration) {
    const auto system = ysq::asPhaseSystem(spring);
    const ScalarPhase at{2.0, 3.0};
    const ScalarPhase derivative = system(0.0, at);

    EXPECT_APPROX(derivative.position, 3.0) << "dq/dt is the velocity";
    EXPECT_APPROX(derivative.velocity, -2.0) << "dv/dt is the acceleration";
}

// --- The fixed-step driver --------------------------------------------------

TEST(MathOde, TheDriverLandsExactlyOnTheEndTime) {
    // A loop that adds h until it passes the end accumulates rounding error in
    // the final time and takes one short step at the end. Both of those
    // corrupt an order measurement, which is most of what this driver is for.
    ysq::Rk4Stepper<double> stepper;

    std::vector<double> times;
    const auto record = [&](double t, double) { times.push_back(t); };

    // 1.0 / 0.3 is not an integer, so the step has to be adjusted down.
    ysq::integrate(stepper, decay, 1.0, 0.0, 1.0, 0.3, record);

    ASSERT_FALSE(times.empty());
    EXPECT_EQ(times.front(), 0.0);
    EXPECT_EQ(times.back(), 1.0) << "exactly, not approximately";
    EXPECT_EQ(times.size(), 5u) << "four steps of 0.25, not three of 0.3 and a stub";

    // Every step is the same size.
    for (std::size_t i = 1; i < times.size(); ++i) {
        EXPECT_NEAR(times[i] - times[i - 1], 0.25, 1e-15);
    }
}

TEST(MathOde, StepCountAgreesWithWhatTheDriverDoes) {
    ysq::Rk4Stepper<double> stepper;

    for (const double step : {0.3, 0.25, 1.0, 2.0, 0.1}) {
        std::size_t observed = 0;
        ysq::integrate(stepper, decay, 1.0, 0.0, 1.0, step,
                       [&](double, double) { ++observed; });
        // The observer fires once before the first step as well.
        EXPECT_EQ(observed, ysq::stepCount(0.0, 1.0, step) + 1)
            << "at step " << step;
    }

    EXPECT_EQ(ysq::stepCount(0.0, 1.0, 0.0), 0u);
    EXPECT_EQ(ysq::stepCount(0.0, 0.0, 0.1), 0u);
    EXPECT_EQ(ysq::stepCount(0.0, 1.0, 2.0), 1u) << "at least one step";
}

TEST(MathOde, AZeroSpanOrANonPositiveStepDoesNothing) {
    ysq::Rk4Stepper<double> stepper;

    EXPECT_EQ(ysq::integrate(stepper, decay, 5.0, 1.0, 1.0, 0.1), 5.0);
    EXPECT_EQ(ysq::integrate(stepper, decay, 5.0, 0.0, 1.0, 0.0), 5.0);
    EXPECT_EQ(ysq::integrate(stepper, decay, 5.0, 0.0, 1.0, -0.1), 5.0);
    EXPECT_EQ(stepper.evaluations(), 0u) << "and evaluates nothing";
}

TEST(MathOde, IntegratingBackwardsInTimeUndoesTheForwardRun) {
    ysq::Rk4Stepper<double> stepper;

    const double forward = ysq::integrate(stepper, decay, 1.0, 0.0, 1.0, 0.01);
    const double back = ysq::integrate(stepper, decay, forward, 1.0, 0.0, 0.01);

    EXPECT_NEAR(forward, std::exp(-1.0), 1e-10);
    EXPECT_NEAR(back, 1.0, 1e-9) << "a reversed run returns to where it started";
}

TEST(MathOde, TheObserverSeesTheStateAtTheTimeItIsGiven) {
    ysq::Rk4Stepper<double> stepper;

    std::vector<double> times;
    std::vector<double> values;
    ysq::integrate(stepper, decay, 1.0, 0.0, 1.0, 0.05, [&](double t, double y) {
        times.push_back(t);
        values.push_back(y);
    });

    ASSERT_EQ(times.size(), values.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        // RK4 at this step is worth about eight digits, and the error
        // accumulates along the run rather than staying at the first sample.
        EXPECT_NEAR(values[i], std::exp(-times[i]), 1e-7)
            << "sample " << i << " at t = " << times[i];
    }
}

TEST(MathOde, ANonFiniteBoundOrStepDoesNotRunAway) {
    // The fixed driver computes its step count from the span. A NaN span must
    // not reach the conversion to an integer count, which would be undefined.
    ysq::Rk4Stepper<double> stepper;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_EQ(ysq::stepCount(0.0, 1.0, nan), 0u) << "a NaN step is no step";
    EXPECT_EQ(ysq::integrate(stepper, decay, 5.0, 0.0, 1.0, nan), 5.0);
    EXPECT_EQ(stepper.evaluations(), 0u);

    // A NaN end time is a bounded run whose answer is a NaN, which propagates,
    // rather than an unbounded one.
    EXPECT_LE(ysq::stepCount(0.0, nan, 0.1), 1u);
    EXPECT_TRUE(std::isnan(ysq::integrate(stepper, decay, 1.0, 0.0, nan, 0.1)));
}

TEST(MathOde, ASystemThatGoesNonFiniteIsReportedRatherThanRetriedForever) {
    // The controller rejects a step it cannot measure, so a system that blows
    // up has to terminate through the rejection limit instead of shrinking
    // toward zero indefinitely.
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.maximumRejections = 8;

    const auto blowsUp = [](double t, double y) {
        return (t > 0.25) ? std::numeric_limits<double>::quiet_NaN() : -y;
    };

    const auto result =
        ysq::integrateAdaptive(stepper, blowsUp, 1.0, 0.0, 1.0, 0.1, settings);

    EXPECT_FALSE(result.succeeded);
    EXPECT_LT(result.time, 1.0);
}

// --- Stepper state ----------------------------------------------------------

TEST(MathOde, ASteppersScratchDoesNotLeakBetweenRuns) {
    // Steppers hold scratch across calls so they do not allocate per step. The
    // price of that is that a leak between runs would be a silent correctness
    // bug rather than a compile error, so it is worth asserting there is none.
    ysq::Rk4Stepper<double> reused;

    const double first = ysq::integrate(reused, decay, 1.0, 0.0, 1.0, 0.05);
    const double second = ysq::integrate(reused, decay, 1.0, 0.0, 1.0, 0.05);
    EXPECT_EQ(first, second) << "bit for bit, not approximately";

    ysq::Rk4Stepper<double> fresh;
    EXPECT_EQ(ysq::integrate(fresh, decay, 1.0, 0.0, 1.0, 0.05), first);

    // Interleaving a different problem must not change the answer either.
    ysq::integrate(reused, [](double, double y) { return 3.0 * y; }, 2.0, 0.0, 0.5,
                   0.05);
    EXPECT_EQ(ysq::integrate(reused, decay, 1.0, 0.0, 1.0, 0.05), first);
}

TEST(MathOde, AStepperMayWriteItsOutputOverItsInput) {
    // `stepper.step(f, t, y, h, y)` is the obvious way to advance in place and
    // avoid a temporary, so it has to give the same answer as stepping into a
    // separate object. Every stepper here either finishes reading the input
    // before it writes, or copies first; this pins that down, because the next
    // one written could easily not.
    const auto acceleration = [](double, const Vec3& q) { return q * -1.0; };
    const VectorPhase start{Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}};

    const auto agree = [&](auto& aliased, auto& separate, const auto& system,
                           auto state) {
        auto inPlace = state;
        std::remove_cvref_t<decltype(state)> intoFresh{};
        aliased.step(system, 0.0, inPlace, 0.1, inPlace);
        separate.step(system, 0.0, state, 0.1, intoFresh);
        return inPlace == intoFresh;
    };

    ysq::Rk4Stepper<double> rk4A, rk4B;
    EXPECT_TRUE(agree(rk4A, rk4B, decay, 1.0)) << "RK4";

    ysq::DormandPrince54Stepper<double> dpA, dpB;
    EXPECT_TRUE(agree(dpA, dpB, decay, 1.0)) << "Dormand-Prince";

    ysq::ExplicitEulerStepper<double> eulerA, eulerB;
    EXPECT_TRUE(agree(eulerA, eulerB, decay, 1.0)) << "explicit Euler";

    ysq::SemiImplicitEulerStepper<Vec3> siA, siB;
    EXPECT_TRUE(agree(siA, siB, acceleration, start)) << "semi-implicit Euler";

    ysq::VelocityVerletStepper<Vec3> verletA, verletB;
    EXPECT_TRUE(agree(verletA, verletB, acceleration, start)) << "velocity Verlet";

    ysq::ForestRuthStepper<Vec3> frA, frB;
    EXPECT_TRUE(agree(frA, frB, acceleration, start)) << "Forest-Ruth";

    ysq::PefrlStepper<Vec3> pefrlA, pefrlB;
    EXPECT_TRUE(agree(pefrlA, pefrlB, acceleration, start)) << "PEFRL";
}

TEST(MathOde, SteppersCountTheirOwnEvaluations) {
    ysq::ExplicitEulerStepper<double> euler;
    ysq::Rk4Stepper<double> rk4;
    ysq::VelocityVerletStepper<double> verlet;
    ysq::ForestRuthStepper<double> forestRuth;
    ysq::PefrlStepper<double> pefrl;

    constexpr std::size_t steps = 10;
    constexpr double step = 0.1;

    ysq::integrate(euler, decay, 1.0, 0.0, 1.0, step);
    ysq::integrate(rk4, decay, 1.0, 0.0, 1.0, step);
    ysq::integrate(verlet, spring, ScalarPhase{1.0, 0.0}, 0.0, 1.0, step);
    ysq::integrate(forestRuth, spring, ScalarPhase{1.0, 0.0}, 0.0, 1.0, step);
    ysq::integrate(pefrl, spring, ScalarPhase{1.0, 0.0}, 0.0, 1.0, step);

    EXPECT_EQ(euler.evaluations(), steps * 1);
    EXPECT_EQ(rk4.evaluations(), steps * 4);
    EXPECT_EQ(verlet.evaluations(), steps * 2);
    EXPECT_EQ(forestRuth.evaluations(), steps * 3);
    EXPECT_EQ(pefrl.evaluations(), steps * 4);
}

// --- The error norm ---------------------------------------------------------

TEST(MathOde, TheErrorNormWeighsEachComponentAgainstItsOwnTolerance) {
    // A single norm of the whole state would let the largest component set the
    // step for all of them, which is wrong the moment a position in metres
    // shares a state with a velocity in metres per second.
    EXPECT_APPROX(ysq::errorNorm(0.0, 1.0, 1.0, 1e-6, 1e-6), 0.0);

    // An error exactly at tolerance reads as one, which is the accept
    // threshold.
    EXPECT_NEAR(ysq::errorNorm(2e-6, 1.0, 1.0, 1e-6, 1e-6), 1.0, 1e-12);

    // The relative term is what keeps a large component from dominating.
    const Vec3 error{1e-6, 1e-6, 1e-6};
    const Vec3 small{1.0, 1.0, 1.0};
    const Vec3 large{1.0, 1.0, 1e6};
    EXPECT_LT(ysq::errorNorm(error, large, large, 1e-9, 1e-9),
              ysq::errorNorm(error, small, small, 1e-9, 1e-9))
        << "the same absolute error is less significant on a larger component";

    // It walks into a nested state rather than stopping at the top level.
    const VectorPhase phaseError{Vec3{1e-6, 0.0, 0.0}, Vec3{0.0, 0.0, 0.0}};
    const VectorPhase reference{Vec3{1.0, 1.0, 1.0}, Vec3{1.0, 1.0, 1.0}};
    // One component of six carries the error, so the RMS is that over sqrt(6).
    EXPECT_NEAR(ysq::errorNorm(phaseError, reference, reference, 1e-6, 0.0),
                1.0 / std::sqrt(6.0), 1e-12);
}

// --- The adaptive driver ----------------------------------------------------

TEST(MathOde, TheAdaptiveDriverLandsExactlyOnTheEndTime) {
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.absoluteTolerance = 1e-10;
    settings.relativeTolerance = 1e-10;

    const auto result =
        ysq::integrateAdaptive(stepper, decay, 1.0, 0.0, 1.0, 0.1, settings);

    EXPECT_TRUE(result.succeeded);
    EXPECT_EQ(result.time, 1.0) << "exactly, by clipping the last step";
    EXPECT_NEAR(result.state, std::exp(-1.0), 1e-9);
    EXPECT_GT(result.acceptedSteps, 0u);
}

TEST(MathOde, TheAdaptiveDriverRespectsItsStepBounds) {
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.maximumStep = 0.05;

    std::vector<double> times;
    const auto result = ysq::integrateAdaptive(
        stepper, decay, 1.0, 0.0, 1.0, 0.5, settings,
        [&](double t, double) { times.push_back(t); });

    ASSERT_TRUE(result.succeeded);
    for (std::size_t i = 1; i < times.size(); ++i) {
        EXPECT_LE(times[i] - times[i - 1], 0.05 + 1e-12) << "step " << i;
    }
    EXPECT_GE(result.acceptedSteps, 20u);
}

TEST(MathOde, TheAdaptiveDriverReportsFailureRatherThanSpinning) {
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    // A tolerance no step can meet, with a floor under the step size.
    settings.absoluteTolerance = 1e-30;
    settings.relativeTolerance = 1e-30;
    settings.minimumStep = 0.01;
    settings.maximumRejections = 5;

    const auto result =
        ysq::integrateAdaptive(stepper, decay, 1.0, 0.0, 1.0, 0.5, settings);

    EXPECT_FALSE(result.succeeded);
    EXPECT_LT(result.time, 1.0);
    EXPECT_GT(result.rejectedSteps, 0u);
}

TEST(MathOde, TheAdaptiveDriverIntegratesBackwardsToo) {
    // The fixed-step driver has always run in either direction. This one used
    // to compare its time against the end with a bare `<`, so a backward run
    // fell straight out of the loop, changed nothing, and reported success:
    // the worst kind of failure, an answer that looks like one.
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.absoluteTolerance = 1e-11;
    settings.relativeTolerance = 1e-11;

    const auto forward =
        ysq::integrateAdaptive(stepper, decay, 1.0, 0.0, 1.0, 0.1, settings);
    ASSERT_TRUE(forward.succeeded);

    stepper.reset();
    const auto back = ysq::integrateAdaptive(stepper, decay, forward.state, 1.0,
                                             0.0, 0.1, settings);

    EXPECT_TRUE(back.succeeded);
    EXPECT_EQ(back.time, 0.0) << "it has to reach the end it was given";
    EXPECT_GT(back.acceptedSteps, 0u) << "and get there by stepping";
    EXPECT_NEAR(back.state, 1.0, 1e-8) << "a reversed run returns to the start";

    // The step bounds still read the same way when the direction is negative.
    ysq::AdaptiveSettings<double> bounded;
    bounded.maximumStep = 0.05;
    std::vector<double> times;
    const auto clamped = ysq::integrateAdaptive(
        stepper, decay, 1.0, 1.0, 0.0, 0.5, bounded,
        [&](double t, double) { times.push_back(t); });
    ASSERT_TRUE(clamped.succeeded);
    for (std::size_t i = 1; i < times.size(); ++i) {
        EXPECT_LE(times[i - 1] - times[i], 0.05 + 1e-12) << "step " << i;
    }
}

TEST(MathOde, TheAdaptiveDriverSurvivesAnUnusableInitialStep) {
    // A zero initial step used to be taken literally: every attempt advanced
    // the time by nothing, every attempt was accepted because its error was
    // zero, and the controller scaled zero by five forever.
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.absoluteTolerance = 1e-9;
    settings.relativeTolerance = 1e-9;

    for (const double guess : {0.0, -0.1,
                               std::numeric_limits<double>::quiet_NaN()}) {
        stepper.reset();
        const auto result =
            ysq::integrateAdaptive(stepper, decay, 1.0, 0.0, 1.0, guess, settings);

        EXPECT_TRUE(result.succeeded) << "at guess " << guess;
        EXPECT_EQ(result.time, 1.0);
        EXPECT_NEAR(result.state, std::exp(-1.0), 1e-7);
    }
}

TEST(MathOde, AStepFloorAboveTheIntervalIsReportedRatherThanIgnored) {
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.minimumStep = 10.0;  // larger than the whole interval

    const auto result =
        ysq::integrateAdaptive(stepper, decay, 1.0, 0.0, 1.0, 0.1, settings);

    EXPECT_FALSE(result.succeeded);
    EXPECT_LT(result.time, 1.0);
}

TEST(MathOde, TheAdaptiveDriverStopsAtItsStepBudget) {
    // maximumRejections bounds how long the controller may struggle at one
    // point. It says nothing about a run that is simply very long, which for
    // an unattended integration is the case that matters.
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.maximumStep = 1e-3;  // forces many steps over the interval
    settings.maximumSteps = 50;

    const auto result =
        ysq::integrateAdaptive(stepper, decay, 1.0, 0.0, 1.0, 1e-3, settings);

    EXPECT_FALSE(result.succeeded);
    EXPECT_EQ(result.acceptedSteps, 50u);
    EXPECT_LT(result.time, 1.0);

    // And a budget that is not in the way does not change the answer.
    ysq::DormandPrince54Stepper<double> unbounded;
    ysq::AdaptiveSettings<double> generous;
    generous.maximumStep = 1e-3;
    const auto full =
        ysq::integrateAdaptive(unbounded, decay, 1.0, 0.0, 1.0, 1e-3, generous);
    EXPECT_TRUE(full.succeeded);
    EXPECT_GT(full.acceptedSteps, 50u);
}

TEST(MathOde, AZeroSpanAdaptiveRunSucceedsWithoutStepping) {
    ysq::DormandPrince54Stepper<double> stepper;
    const ysq::AdaptiveSettings<double> settings;

    const auto result =
        ysq::integrateAdaptive(stepper, decay, 3.0, 1.0, 1.0, 0.1, settings);

    EXPECT_TRUE(result.succeeded);
    EXPECT_EQ(result.state, 3.0);
    EXPECT_EQ(result.acceptedSteps, 0u);
    EXPECT_EQ(result.evaluations, 0u);
}

TEST(MathOde, TheAdaptiveDriverRunsAVectorValuedSystemToo) {
    ysq::DormandPrince54Stepper<Vec3> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.absoluteTolerance = 1e-10;
    settings.relativeTolerance = 1e-10;

    const auto system = [](double, const Vec3& y) { return y * -1.0; };
    const auto result = ysq::integrateAdaptive(
        stepper, system, Vec3{1.0, 2.0, 3.0}, 0.0, 1.0, 0.1, settings);

    ASSERT_TRUE(result.succeeded);
    EXPECT_VEC_NEAR(result.state, (Vec3{1.0, 2.0, 3.0} * std::exp(-1.0)), 1e-9);
}

}  // namespace
