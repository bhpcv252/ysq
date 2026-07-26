#include <Core/Clock.hpp>
#include <Core/Config.hpp>
#include <Core/Event.hpp>
#include <Core/Logger.hpp>
#include <Core/Timer.hpp>
#include <Core/UUID.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// Stage 1's done-when, as one run: configure, log, time, identify, dispatch.
// This is the wiring an application will do in main(); it lives here because
// Core has no application to host it yet.

namespace {

struct StepCompleted {
    std::uint64_t index = 0;
    double simulationTime = 0.0;
};

struct RunCompleted {
    ysq::UUID run;
    std::uint64_t steps = 0;
};

constexpr std::string_view kConfigText = R"(# stage 1 runtime
[log]
level = info

[sim]
fixedStep    = 0.25
timeScale    = 2.0
maxSteps     = 64
frames       = 8
frameSeconds = 0.5
)";

ysq::LogLevel toLevel(std::string_view name) {
    if (name == "trace") return ysq::LogLevel::Trace;
    if (name == "debug") return ysq::LogLevel::Debug;
    if (name == "warn") return ysq::LogLevel::Warn;
    if (name == "error") return ysq::LogLevel::Error;
    if (name == "critical") return ysq::LogLevel::Critical;
    if (name == "off") return ysq::LogLevel::Off;
    return ysq::LogLevel::Info;
}

std::size_t occurrences(std::string_view haystack, std::string_view needle) {
    std::size_t count = 0;
    for (std::size_t pos = haystack.find(needle); pos != std::string_view::npos;
         pos = haystack.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

TEST(CoreRuntime, ConfigDrivesALoggedTimedDeterministicRun) {
    ysq::ConfigError error;
    const std::optional<ysq::Config> config = ysq::Config::parse(kConfigText, &error);
    ASSERT_TRUE(config.has_value()) << "line " << error.line << ": " << error.message;

    std::ostringstream captured;
    ysq::LogSettings logSettings;
    logSettings.name = "runtime";
    logSettings.pattern = "%l %v";
    logSettings.level = toLevel(config->get<std::string>("log.level", "info"));
    logSettings.console = false;
    logSettings.stream = &captured;
    ysq::Logger::init(logSettings);

    ysq::Clock::Settings clockSettings;
    clockSettings.fixedStep = config->get<double>("sim.fixedStep", 1.0 / 60.0);
    clockSettings.timeScale = config->get<double>("sim.timeScale", 1.0);
    clockSettings.maxStepsPerAdvance = config->get<int>("sim.maxSteps", 8);
    ysq::Clock clock(clockSettings);

    const int frames = config->get<int>("sim.frames", 0);
    const double frameSeconds = config->get<double>("sim.frameSeconds", 0.0);
    ASSERT_GT(frames, 0);
    ASSERT_GT(frameSeconds, 0.0);

    const ysq::UUID runId = ysq::UUID::generate();
    ysq::EventBus bus;

    std::vector<StepCompleted> steps;
    const ysq::Subscription onStep =
        bus.subscribe<StepCompleted>([&steps](const StepCompleted& e) {
            steps.push_back(e);
        });

    std::optional<RunCompleted> completion;
    const ysq::Subscription onCompletion =
        bus.subscribe<RunCompleted>([&completion](const RunCompleted& e) {
            completion = e;
        });

    ysq::log::info("run {} starting", runId.toString());

    const ysq::Timer wallClock;
    for (int frame = 0; frame < frames; ++frame) {
        const int due = clock.advance(frameSeconds);
        while (clock.consumeStep()) {
            bus.publish(StepCompleted{clock.stepCount(), clock.simulationTime()});
        }
        ysq::log::info("frame {} ran {} steps, t={}", frame, due, clock.simulationTime());
    }
    const double wallSeconds = wallClock.elapsedSeconds();

    bus.enqueue(RunCompleted{runId, clock.stepCount()});
    EXPECT_FALSE(completion.has_value()) << "a queued event was delivered early";
    bus.dispatchQueued();

    ysq::log::info("run {} finished: {} steps of simulation in {:.6f}s of wall clock",
                   runId.toString(), clock.stepCount(), wallSeconds);
    ysq::Logger::shutdown();

    // 0.5 real seconds per frame at 2x is 1.0 simulated second, which is four
    // 0.25s steps. Every value here is a power of two, so this is exact.
    EXPECT_EQ(clock.stepCount(), 32u);
    EXPECT_DOUBLE_EQ(clock.simulationTime(), 8.0);
    EXPECT_DOUBLE_EQ(clock.realTime(), 4.0);
    EXPECT_DOUBLE_EQ(clock.alpha(), 0.0);
    EXPECT_GE(wallSeconds, 0.0);

    ASSERT_EQ(steps.size(), 32u);
    EXPECT_EQ(steps.front().index, 1u);
    EXPECT_DOUBLE_EQ(steps.front().simulationTime, 0.25);
    EXPECT_EQ(steps.back().index, 32u);
    EXPECT_DOUBLE_EQ(steps.back().simulationTime, 8.0);

    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->run, runId);
    EXPECT_EQ(completion->steps, 32u);

    const std::string log = captured.str();
    EXPECT_EQ(occurrences(log, "info frame "), static_cast<std::size_t>(frames));
    EXPECT_NE(log.find("run " + runId.toString() + " starting"), std::string::npos);
    EXPECT_NE(log.find("run " + runId.toString() + " finished: 32 steps"),
              std::string::npos);
    EXPECT_EQ(occurrences(log, "frame 0 ran 4 steps, t=1"), 1u);
}

}  // namespace
