#include <gtest/gtest.h>

#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>

#include <memory>
#include <sstream>
#include <string>

// spdlog is built with SPDLOG_USE_STD_FORMAT so the project has exactly one
// formatting implementation and no bundled fmt. That option is easy to lose
// silently in a dependency bump, so assert it instead of trusting it.

TEST(SpdlogSmoke, BuiltAgainstStdFormat) {
#ifdef SPDLOG_USE_STD_FORMAT
    SUCCEED();
#else
    FAIL() << "spdlog was built without SPDLOG_USE_STD_FORMAT; it is using "
              "bundled fmt instead of std::format";
#endif
}

TEST(SpdlogSmoke, FormatsThroughASink) {
    std::ostringstream captured;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(captured);

    spdlog::logger logger("smoke", sink);
    logger.set_pattern("%v");  // message only, no timestamp or level
    logger.info("{} bodies at t={}", 3, 1.5);

    // Compared with starts_with because spdlog's line ending is platform-dependent.
    EXPECT_TRUE(std::string{captured.str()}.starts_with("3 bodies at t=1.5"))
        << "actual: " << captured.str();
}
