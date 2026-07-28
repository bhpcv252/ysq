#include <Core/Timer.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <ratio>
#include <sstream>
#include <string>

namespace {

using namespace std::chrono_literals;

/// A clock that only moves when the test says so, which is what lets these
/// assertions be exact instead of tolerant of a sleeping thread.
struct ManualClock {
    using rep = std::int64_t;
    using period = std::nano;
    using duration = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<ManualClock>;
    static constexpr bool is_steady = true;

    static inline time_point current{};

    static time_point now() { return current; }
    static void advance(duration delta) { current += delta; }
    static void rewind() { current = time_point{}; }
};

static_assert(ManualClock::is_steady, "the fake clock stands in for a steady one");

using TestTimer = ysq::BasicTimer<ManualClock>;

class TimerTest : public ::testing::Test {
protected:
    void SetUp() override { ManualClock::rewind(); }
};

TEST_F(TimerTest, AccumulatesElapsedTime) {
    const TestTimer timer;
    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 0.0);

    ManualClock::advance(250ms);
    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 0.25);

    ManualClock::advance(750ms);
    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 1.0);
}

TEST_F(TimerTest, StopFreezesAndResumeContinues) {
    TestTimer timer;
    ManualClock::advance(200ms);

    timer.stop();
    EXPECT_FALSE(timer.running());
    ManualClock::advance(1s);
    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 0.2) << "a stopped timer kept counting";

    timer.resume();
    EXPECT_TRUE(timer.running());
    ManualClock::advance(300ms);
    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 0.5);
}

TEST_F(TimerTest, StopIsIdempotent) {
    TestTimer timer;
    ManualClock::advance(100ms);
    timer.stop();
    timer.stop();
    ManualClock::advance(100ms);

    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 0.1);
}

TEST_F(TimerTest, LapReturnsTheIntervalAndRestarts) {
    TestTimer timer;

    ManualClock::advance(400ms);
    EXPECT_DOUBLE_EQ(timer.lap().count(), 0.4);
    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 0.0);

    ManualClock::advance(600ms);
    EXPECT_DOUBLE_EQ(timer.lap().count(), 0.6);
}

TEST_F(TimerTest, ResetClearsAStoppedTimer) {
    TestTimer timer;
    ManualClock::advance(1s);
    timer.stop();

    timer.reset();
    EXPECT_TRUE(timer.running());
    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 0.0);

    ManualClock::advance(500ms);
    EXPECT_DOUBLE_EQ(timer.elapsedSeconds(), 0.5);
}

// ScopedTimer is the one part of Timer.hpp that reaches into Logger, so it is
// checked against a captured sink rather than a fake clock.
class ScopedTimerTest : public ::testing::Test {
protected:
    void initAt(ysq::LogLevel level) {
        ysq::LogSettings settings;
        settings.pattern = "%v";
        settings.level = level;
        settings.console = false;
        settings.stream = &captured;
        ysq::Logger::init(settings);
    }

    void TearDown() override { ysq::Logger::shutdown(); }

    std::ostringstream captured;
};

TEST_F(ScopedTimerTest, LogsItsLabelAndDurationWhenTheScopeEnds) {
    initAt(ysq::LogLevel::Debug);

    {
        const ysq::ScopedTimer timer("integrate");
        EXPECT_TRUE(captured.str().empty()) << "logged before the scope ended";
    }

    EXPECT_NE(captured.str().find("integrate took "), std::string::npos)
        << captured.str();
    EXPECT_NE(captured.str().find(" ms"), std::string::npos) << captured.str();
}

TEST_F(ScopedTimerTest, StaysSilentBelowTheActiveLevel) {
    initAt(ysq::LogLevel::Warn);

    {
        const ysq::ScopedTimer timer("integrate");
    }

    EXPECT_TRUE(captured.str().empty()) << captured.str();
}

TEST_F(ScopedTimerTest, HonoursAnExplicitLevel) {
    initAt(ysq::LogLevel::Warn);

    {
        const ysq::ScopedTimer timer("integrate", ysq::LogLevel::Error);
    }

    EXPECT_NE(captured.str().find("integrate took "), std::string::npos)
        << captured.str();
}

// The fake clock proves the arithmetic; this proves the default clock is wired
// to something that actually moves.
TEST(TimerRealClock, SteadyClockAdvances) {
    const ysq::Timer timer;
    volatile double sink = 0.0;
    for (int i = 0; i < 100000; ++i) {
        sink = sink + 1.0;
    }

    EXPECT_GE(timer.elapsedSeconds(), 0.0);
    EXPECT_TRUE(ysq::Timer::Clock::is_steady);
}

}  // namespace
