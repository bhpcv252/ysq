#include <Core/Logger.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>

namespace {

// The logger is a process-wide singleton, so every test re-initialises it
// against its own stream. The pattern is level and message only: a timestamp
// would make the assertions depend on the clock.
//
// The streams are fixture members, not locals. A sink holds a reference to the
// stream it writes to, and shutdown() flushes; a local would already have been
// destroyed by the time TearDown ran.
class LoggerTest : public ::testing::Test {
protected:
    void initTo(std::ostream& stream, ysq::LogLevel level) {
        ysq::LogSettings settings;
        settings.name = "test";
        settings.pattern = "%l %v";
        settings.level = level;
        settings.console = false;
        settings.stream = &stream;
        ysq::Logger::init(settings);
    }

    void TearDown() override { ysq::Logger::shutdown(); }

    std::ostringstream captured;
    std::ostringstream alternate;
};

TEST_F(LoggerTest, FormatsThroughTheConfiguredSink) {
    initTo(captured, ysq::LogLevel::Info);

    ysq::log::info("{} bodies at t={}", 3, 1.5);

    EXPECT_TRUE(captured.str().starts_with("info 3 bodies at t=1.5")) << captured.str();
}

TEST_F(LoggerTest, SuppressesBelowTheActiveLevel) {
    initTo(captured, ysq::LogLevel::Warn);

    ysq::log::trace("trace line");
    ysq::log::debug("debug line");
    ysq::log::info("info line");
    ysq::log::warn("warn line");
    ysq::log::error("error line");
    ysq::log::critical("critical line");

    const std::string text = captured.str();
    EXPECT_EQ(text.find("trace line"), std::string::npos);
    EXPECT_EQ(text.find("debug line"), std::string::npos);
    EXPECT_EQ(text.find("info line"), std::string::npos);
    EXPECT_NE(text.find("warn line"), std::string::npos);
    EXPECT_NE(text.find("error line"), std::string::npos);
    EXPECT_NE(text.find("critical line"), std::string::npos);
}

TEST_F(LoggerTest, LevelChangesTakeEffectImmediately) {
    initTo(captured, ysq::LogLevel::Error);

    ysq::log::info("before");
    EXPECT_EQ(ysq::Logger::level(), ysq::LogLevel::Error);
    EXPECT_FALSE(ysq::Logger::enabled(ysq::LogLevel::Info));

    ysq::Logger::setLevel(ysq::LogLevel::Debug);
    EXPECT_EQ(ysq::Logger::level(), ysq::LogLevel::Debug);
    EXPECT_TRUE(ysq::Logger::enabled(ysq::LogLevel::Info));
    ysq::log::info("after");

    const std::string text = captured.str();
    EXPECT_EQ(text.find("before"), std::string::npos);
    EXPECT_NE(text.find("after"), std::string::npos);
}

TEST_F(LoggerTest, OffSuppressesEveryLevel) {
    initTo(captured, ysq::LogLevel::Off);

    ysq::log::critical("critical line");

    EXPECT_FALSE(ysq::Logger::enabled(ysq::LogLevel::Critical));
    EXPECT_TRUE(captured.str().empty()) << captured.str();
}

TEST_F(LoggerTest, ReinitialisingSwapsTheSink) {
    initTo(captured, ysq::LogLevel::Info);
    ysq::log::info("to the first sink");

    initTo(alternate, ysq::LogLevel::Info);
    ysq::log::info("to the second sink");

    EXPECT_NE(captured.str().find("to the first sink"), std::string::npos);
    EXPECT_EQ(captured.str().find("to the second sink"), std::string::npos);
    EXPECT_NE(alternate.str().find("to the second sink"), std::string::npos);
}

// Formatting happens on our side, so what reaches spdlog is a finished string.
// If it were ever routed through spdlog's own formatting overload instead, a
// value containing braces would be reinterpreted as a format string.
TEST_F(LoggerTest, DoesNotReinterpretBracesInAFormattedValue) {
    initTo(captured, ysq::LogLevel::Info);

    ysq::log::info("{}", "literal {braces} and {0} and {:>9}");

    EXPECT_NE(captured.str().find("literal {braces} and {0} and {:>9}"),
              std::string::npos)
        << captured.str();
}

TEST_F(LoggerTest, WritesToAFileSink) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ysq_core_logger_test.log";
    std::error_code ec;
    std::filesystem::remove(path, ec);  // the file sink appends

    ysq::LogSettings settings;
    settings.pattern = "%v";
    settings.level = ysq::LogLevel::Info;
    settings.console = false;
    settings.file = path;
    ysq::Logger::init(settings);
    ysq::log::info("to the file sink");
    ysq::Logger::shutdown();

    {
        std::ifstream file(path);
        ASSERT_TRUE(file.is_open());
        const std::string text{std::istreambuf_iterator<char>{file},
                               std::istreambuf_iterator<char>{}};
        EXPECT_NE(text.find("to the file sink"), std::string::npos) << text;
    }

    // shutdown() has to drop the sink, not just flush it. Windows refuses to
    // delete a file that is still open, so this is the assertion that catches a
    // logger being kept alive past shutdown; on POSIX the removal would succeed
    // either way.
    EXPECT_TRUE(std::filesystem::remove(path, ec))
        << "log file still held open after shutdown: " << ec.message();
}

// Exercises the lazy path in detail::write. There is no sink to capture yet by
// definition, so this prints one line to stdout; that it does not crash, and
// that it does not discard a level set beforehand, is the whole assertion.
TEST_F(LoggerTest, LoggingBeforeInitUsesDefaultsAndKeepsTheLevel) {
    ysq::Logger::shutdown();
    ysq::Logger::setLevel(ysq::LogLevel::Critical);

    ysq::log::info("suppressed: below the level set before the first log call");
    ysq::log::critical("expected on stdout: logging before init does not crash");

    EXPECT_EQ(ysq::Logger::level(), ysq::LogLevel::Critical);
}

}  // namespace
