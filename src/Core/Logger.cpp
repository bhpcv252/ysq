#include <Core/Logger.hpp>

#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <memory>
#include <mutex>
#include <vector>

namespace ysq {

namespace {

spdlog::level::level_enum toSpdlog(LogLevel level) {
    switch (level) {
        case LogLevel::Trace:    return spdlog::level::trace;
        case LogLevel::Debug:    return spdlog::level::debug;
        case LogLevel::Info:     return spdlog::level::info;
        case LogLevel::Warn:     return spdlog::level::warn;
        case LogLevel::Error:    return spdlog::level::err;
        case LogLevel::Critical: return spdlog::level::critical;
        case LogLevel::Off:      return spdlog::level::off;
    }
    return spdlog::level::info;
}

// Function-local statics, so destruction order is defined and a log call during
// static teardown re-creates the logger rather than touching freed memory.
//
// The lock on every emitted line is deliberate. Keeping the logger in a
// shared_ptr under a mutex is what makes shutdown() release its sinks, and a
// file sink that outlives shutdown() holds the log file open, which on Windows
// means it cannot be rotated or deleted. A lock-free read needs the pointer to
// stay valid for a writer that loaded it just before a swap, which without
// std::atomic<std::shared_ptr> (unimplemented in libc++) means retaining every
// logger forever and giving up that guarantee. spdlog's sink already takes a
// mutex per line, so this second uncontended acquire is the cheaper trade.
struct State {
    std::mutex mutex;
    std::shared_ptr<spdlog::logger> logger;
};

State& state() {
    static State s;
    return s;
}

// Callers already hold state().mutex. init() and the lazy path in write() share
// this; going through init() from write() would deadlock on the same mutex.
void initLocked(State& s, const LogSettings& settings) {
    std::vector<spdlog::sink_ptr> sinks;
    if (settings.console) {
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    }
    if (settings.file) {
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            settings.file->string(), /*truncate=*/false));
    }
    if (settings.stream != nullptr) {
        sinks.push_back(
            std::make_shared<spdlog::sinks::ostream_sink_mt>(*settings.stream));
    }

    auto logger =
        std::make_shared<spdlog::logger>(settings.name, sinks.begin(), sinks.end());
    logger->set_pattern(settings.pattern);
    logger->set_level(toSpdlog(settings.level));
    logger->flush_on(spdlog::level::warn);

    s.logger = std::move(logger);
    detail::g_logLevel.store(static_cast<int>(settings.level), std::memory_order_relaxed);
}

}  // namespace

void Logger::init(const LogSettings& settings) {
    State& s = state();
    const std::lock_guard lock(s.mutex);
    initLocked(s, settings);
}

void Logger::shutdown() {
    State& s = state();
    const std::lock_guard lock(s.mutex);
    if (s.logger) {
        s.logger->flush();
    }
    // Dropping the logger drops its sinks, which is what closes the log file.
    s.logger.reset();
}

void Logger::setLevel(LogLevel level) {
    State& s = state();
    const std::lock_guard lock(s.mutex);
    if (s.logger) {
        s.logger->set_level(toSpdlog(level));
    }
    detail::g_logLevel.store(static_cast<int>(level), std::memory_order_relaxed);
}

LogLevel Logger::level() noexcept {
    return static_cast<LogLevel>(detail::g_logLevel.load(std::memory_order_relaxed));
}

void detail::write(LogLevel level, std::string_view message) {
    if (level == LogLevel::Off) {
        return;
    }

    std::shared_ptr<spdlog::logger> logger;
    {
        State& s = state();
        const std::lock_guard lock(s.mutex);
        if (!s.logger) {
            // A setLevel() before the first log call still holds: take the level
            // from the mirror rather than resetting it to the default.
            LogSettings defaults;
            defaults.level = Logger::level();
            initLocked(s, defaults);
        }
        logger = s.logger;
    }
    // Outside the lock, and holding a copy: a concurrent shutdown() can drop the
    // logger without pulling it out from under this write.
    logger->log(toSpdlog(level), message);
}

}  // namespace ysq
