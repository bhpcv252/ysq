#pragma once

#include <Core/Logger.hpp>

#include <chrono>
#include <format>
#include <string>
#include <utility>

namespace ysq {

/// Wall-clock stopwatch. Templated on the clock so tests can drive it with a
/// manual one and assert exact values instead of sleeping.
template <class ClockT = std::chrono::steady_clock>
class BasicTimer {
public:
    using Clock = ClockT;
    using Seconds = std::chrono::duration<double>;

    BasicTimer() : m_start(Clock::now()) {}

    void reset() {
        m_start = Clock::now();
        m_accumulated = Seconds::zero();
        m_running = true;
    }

    void stop() {
        if (m_running) {
            m_accumulated += Clock::now() - m_start;
            m_running = false;
        }
    }

    void resume() {
        if (!m_running) {
            m_start = Clock::now();
            m_running = true;
        }
    }

    [[nodiscard]] bool running() const noexcept { return m_running; }

    [[nodiscard]] Seconds elapsed() const {
        if (!m_running) {
            return m_accumulated;
        }
        // Cast rather than a conditional expression: the clock's own duration
        // and Seconds convert to each other, which makes a ternary ambiguous.
        return m_accumulated +
               std::chrono::duration_cast<Seconds>(Clock::now() - m_start);
    }

    [[nodiscard]] double elapsedSeconds() const { return elapsed().count(); }

    /// Elapsed since the last lap, then restart. The frame-loop idiom.
    Seconds lap() {
        const Seconds e = elapsed();
        reset();
        return e;
    }

private:
    typename Clock::time_point m_start;
    Seconds m_accumulated = Seconds::zero();
    bool m_running = true;
};

using Timer = BasicTimer<std::chrono::steady_clock>;

/// Logs how long its scope took, on destruction.
class ScopedTimer {
public:
    explicit ScopedTimer(std::string label, LogLevel level = LogLevel::Debug)
        : m_label(std::move(label)),
          m_level(level) {}

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    ~ScopedTimer() {
        if (!Logger::enabled(m_level)) {
            return;
        }
        detail::write(m_level,
                      std::format("{} took {:.3f} ms", m_label,
                                  m_timer.elapsedSeconds() * 1000.0));
    }

private:
    std::string m_label;
    LogLevel m_level;
    Timer m_timer;
};

}  // namespace ysq
