#pragma once

#include <atomic>
#include <filesystem>
#include <format>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>

namespace ysq {

enum class LogLevel { Trace, Debug, Info, Warn, Error, Critical, Off };

/// Sinks are independent; with all three off the logger silently drops output.
struct LogSettings {
    std::string name = "ysq";
    LogLevel level = LogLevel::Info;
    /// Time, logger name, level, message.
    std::string pattern = "[%T.%e] [%n] [%l] %v";
    bool console = true;
    std::optional<std::filesystem::path> file{};
    /// Not owned, and must outlive the logger. Tests capture output with it.
    std::ostream* stream = nullptr;
};

namespace detail {

/// Mirrors the active level so a filtered call never leaves this header and
/// never formats. Relaxed: a stale read costs one log line, not correctness.
inline std::atomic<int> g_logLevel{static_cast<int>(LogLevel::Info)};

void write(LogLevel level, std::string_view message);

}  // namespace detail

/// spdlog lives behind this. Nothing here names it, so consumers of Core neither
/// compile spdlog headers nor depend on the backend staying spdlog.
class Logger {
public:
    /// Optional. The first log call initialises defaults if this was not called.
    static void init(const LogSettings& settings = {});
    static void shutdown();

    static void setLevel(LogLevel level);
    [[nodiscard]] static LogLevel level() noexcept;

    [[nodiscard]] static bool enabled(LogLevel level) noexcept {
        const int active = detail::g_logLevel.load(std::memory_order_relaxed);
        return active != static_cast<int>(LogLevel::Off) &&
               static_cast<int>(level) >= active;
    }
};

namespace log {

/// Args are const lvalue references rather than forwarding references because
/// P2905 made std::make_format_args bind lvalues only; forwarding rvalues here
/// fails to compile on current libc++ and libstdc++.
template <LogLevel Level, class... Args>
void emit(std::format_string<Args...> fmt, const Args&... args) {
    if (!Logger::enabled(Level)) {
        return;
    }
    detail::write(Level, std::vformat(fmt.get(), std::make_format_args(args...)));
}

template <class... Args>
void trace(std::format_string<Args...> fmt, const Args&... args) {
    emit<LogLevel::Trace>(fmt, args...);
}

template <class... Args>
void debug(std::format_string<Args...> fmt, const Args&... args) {
    emit<LogLevel::Debug>(fmt, args...);
}

template <class... Args>
void info(std::format_string<Args...> fmt, const Args&... args) {
    emit<LogLevel::Info>(fmt, args...);
}

template <class... Args>
void warn(std::format_string<Args...> fmt, const Args&... args) {
    emit<LogLevel::Warn>(fmt, args...);
}

template <class... Args>
void error(std::format_string<Args...> fmt, const Args&... args) {
    emit<LogLevel::Error>(fmt, args...);
}

template <class... Args>
void critical(std::format_string<Args...> fmt, const Args&... args) {
    emit<LogLevel::Critical>(fmt, args...);
}

}  // namespace log

}  // namespace ysq
