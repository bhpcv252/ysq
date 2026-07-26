#include <Core/Clock.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

// Every step size here is a power of two, so the accumulator arithmetic is exact
// and the assertions can be too. The one test that deliberately uses a
// non-representable step asserts with a tolerance instead.
ysq::Clock::Settings settings(double fixedStep, double timeScale, int maxSteps) {
    ysq::Clock::Settings s;
    s.fixedStep = fixedStep;
    s.timeScale = timeScale;
    s.maxStepsPerAdvance = maxSteps;
    return s;
}

/// One frame of the loop a host would write: advance() decides, consumeStep()
/// executes. Returns the steps actually taken.
int runFrame(ysq::Clock& clock, double realDeltaSeconds) {
    const int due = clock.advance(realDeltaSeconds);
    int taken = 0;
    while (clock.consumeStep()) {
        ++taken;
    }
    EXPECT_EQ(taken, due) << "consumeStep did not honour the advertised count";
    return taken;
}

TEST(CoreClock, SimulationTimeRunsAtTimeScaleAgainstRealTime) {
    ysq::Clock clock(settings(0.25, 2.0, 100));

    EXPECT_EQ(runFrame(clock, 1.0), 8);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 2.0);
    EXPECT_DOUBLE_EQ(clock.realTime(), 1.0);
    EXPECT_EQ(clock.stepCount(), 8u);
}

TEST(CoreClock, SimulationTimeAdvancesOneStepAtATime) {
    ysq::Clock clock(settings(0.25, 1.0, 100));

    ASSERT_EQ(clock.advance(1.0), 4);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 0.0) << "advance() must not run the steps";

    std::vector<double> times;
    while (clock.consumeStep()) {
        times.push_back(clock.simulationTime());
    }

    EXPECT_EQ(times, (std::vector<double>{0.25, 0.5, 0.75, 1.0}));
    EXPECT_EQ(clock.stepsDue(), 0);
    EXPECT_FALSE(clock.consumeStep());
}

TEST(CoreClock, PausedAdvancesRealTimeOnly) {
    ysq::Clock clock(settings(0.25, 1.0, 100));
    runFrame(clock, 0.5);
    const double simulated = clock.simulationTime();

    clock.pause();
    EXPECT_EQ(runFrame(clock, 4.0), 0);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), simulated);
    EXPECT_DOUBLE_EQ(clock.realTime(), 4.5);

    clock.resume();
    EXPECT_EQ(runFrame(clock, 0.25), 1);
}

TEST(CoreClock, AccumulatorCarriesRemaindersAcrossCalls) {
    ysq::Clock clock(settings(0.25, 1.0, 100));

    int steps = 0;
    for (int frame = 0; frame < 8; ++frame) {
        steps += runFrame(clock, 0.125);  // half a step per frame
    }

    EXPECT_EQ(steps, 4);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 1.0);
    EXPECT_DOUBLE_EQ(clock.realTime(), 1.0);
}

TEST(CoreClock, ClampBoundsStepsAndDropsTheBacklog) {
    ysq::Clock clock(settings(0.25, 1.0, 4));

    EXPECT_EQ(runFrame(clock, 5.0), 4) << "a stall must not return an unbounded count";
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 1.0);
    EXPECT_DOUBLE_EQ(clock.realTime(), 5.0);
    EXPECT_DOUBLE_EQ(clock.alpha(), 0.0);

    // The backlog is gone rather than waiting to make the next frame worse.
    EXPECT_EQ(runFrame(clock, 0.0), 0);
}

// The accumulator is spent by advance(), so a caller that ignores the steps it
// asked for loses them. Documented, and asserted so it stays deliberate.
TEST(CoreClock, StepsNotConsumedBeforeTheNextAdvanceAreDropped) {
    ysq::Clock clock(settings(0.25, 1.0, 100));

    EXPECT_EQ(clock.advance(1.0), 4);
    EXPECT_EQ(clock.advance(0.25), 1);
    EXPECT_EQ(clock.stepsDue(), 1);

    EXPECT_TRUE(clock.consumeStep());
    EXPECT_FALSE(clock.consumeStep());
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 0.25);
}

// A delta the clock refuses is refused whole: it must not take the pending steps
// with it, or a frame of simulation vanishes on a bad timing sample.
TEST(CoreClock, RefusedDeltasDoNotDropPendingSteps) {
    ysq::Clock clock(settings(0.25, 1.0, 100));
    ASSERT_EQ(clock.advance(1.0), 4);

    EXPECT_EQ(clock.advance(-1.0), 4) << "a negative delta discarded pending steps";
    EXPECT_EQ(clock.advance(std::numeric_limits<double>::quiet_NaN()), 4);
    EXPECT_EQ(clock.advance(std::numeric_limits<double>::infinity()), 4);
    EXPECT_DOUBLE_EQ(clock.realTime(), 1.0) << "a refused delta accrued real time";

    // Pausing does not retroactively cancel steps the last advance() decided on.
    clock.pause();
    EXPECT_EQ(clock.advance(0.5), 4) << "pausing discarded pending steps";

    int taken = 0;
    while (clock.consumeStep()) {
        ++taken;
    }
    EXPECT_EQ(taken, 4);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 1.0);
}

TEST(CoreClock, SettersTakeEffectOnTheNextFrame) {
    ysq::Clock clock(settings(0.25, 1.0, 100));

    clock.setFixedStep(0.5);
    EXPECT_DOUBLE_EQ(clock.fixedStep(), 0.5);
    EXPECT_EQ(runFrame(clock, 1.0), 2);

    clock.setTimeScale(4.0);
    EXPECT_DOUBLE_EQ(clock.timeScale(), 4.0);
    EXPECT_EQ(runFrame(clock, 1.0), 8);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 5.0);

    clock.setTimeScale(0.0);
    EXPECT_EQ(runFrame(clock, 10.0), 0) << "a zero scale must freeze simulation time";
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 5.0);
    EXPECT_DOUBLE_EQ(clock.realTime(), 12.0);
}

TEST(CoreClock, StepOnceAdvancesExactlyOneStepWhilePaused) {
    ysq::Clock clock(settings(0.25, 1.0, 100));
    clock.pause();

    clock.stepOnce();
    clock.stepOnce();

    EXPECT_EQ(clock.stepCount(), 2u);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 0.5);
    EXPECT_DOUBLE_EQ(clock.realTime(), 0.0);
}

TEST(CoreClock, AlphaIsTheUnconsumedFractionOfAStep) {
    ysq::Clock clock(settings(0.25, 1.0, 100));

    EXPECT_EQ(runFrame(clock, 0.125), 0);
    EXPECT_DOUBLE_EQ(clock.alpha(), 0.5);

    EXPECT_EQ(runFrame(clock, 0.125), 1);
    EXPECT_DOUBLE_EQ(clock.alpha(), 0.0);
}

TEST(CoreClock, IgnoresNegativeAndNonFiniteDeltas) {
    ysq::Clock clock(settings(0.25, 1.0, 100));

    EXPECT_EQ(runFrame(clock, -1.0), 0);
    EXPECT_EQ(runFrame(clock, std::numeric_limits<double>::quiet_NaN()), 0);
    EXPECT_EQ(runFrame(clock, std::numeric_limits<double>::infinity()), 0);

    EXPECT_DOUBLE_EQ(clock.realTime(), 0.0);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 0.0);
}

TEST(CoreClock, RejectsUnusableSettings) {
    const ysq::Clock defaults;
    ysq::Clock clock(settings(0.0, -1.0, 0));

    EXPECT_DOUBLE_EQ(clock.fixedStep(), defaults.fixedStep());
    EXPECT_DOUBLE_EQ(clock.timeScale(), defaults.timeScale());

    clock.setFixedStep(-1.0);
    clock.setTimeScale(std::numeric_limits<double>::quiet_NaN());
    EXPECT_DOUBLE_EQ(clock.fixedStep(), defaults.fixedStep());
    EXPECT_DOUBLE_EQ(clock.timeScale(), defaults.timeScale());

    // maxStepsPerAdvance was normalised to 1, not left at 0.
    EXPECT_EQ(runFrame(clock, 10.0), 1);
}

TEST(CoreClock, ResetZeroesTheRunButKeepsSettings) {
    ysq::Clock clock(settings(0.25, 3.0, 100));
    runFrame(clock, 1.0);

    clock.reset();

    EXPECT_EQ(clock.stepCount(), 0u);
    EXPECT_EQ(clock.stepsDue(), 0);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 0.0);
    EXPECT_DOUBLE_EQ(clock.realTime(), 0.0);
    EXPECT_DOUBLE_EQ(clock.alpha(), 0.0);
    EXPECT_DOUBLE_EQ(clock.timeScale(), 3.0);
    EXPECT_DOUBLE_EQ(clock.fixedStep(), 0.25);
}

// The point of feeding deltas in rather than reading a wall clock: a run is a
// pure function of its inputs, so it replays exactly.
TEST(CoreClock, ReplayingTheSameDeltasIsDeterministic) {
    std::vector<double> deltas;
    deltas.reserve(200);
    for (int i = 0; i < 200; ++i) {
        deltas.push_back(0.001 * static_cast<double>((i * 37) % 23 + 1));
    }

    const auto play = [&deltas](ysq::Clock& clock) {
        std::uint64_t steps = 0;
        for (const double delta : deltas) {
            steps += static_cast<std::uint64_t>(runFrame(clock, delta));
        }
        return steps;
    };

    ysq::Clock first(settings(1.0 / 60.0, 1.0, 8));
    ysq::Clock second(settings(1.0 / 60.0, 1.0, 8));

    EXPECT_EQ(play(first), play(second));
    EXPECT_EQ(first.stepCount(), second.stepCount());
    EXPECT_DOUBLE_EQ(first.simulationTime(), second.simulationTime());
    EXPECT_DOUBLE_EQ(first.alpha(), second.alpha());

    // A non-representable step still tracks real time to within one step.
    EXPECT_NEAR(first.simulationTime(), first.realTime(), first.fixedStep());
}

}  // namespace
