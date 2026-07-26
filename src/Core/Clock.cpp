#include <Core/Clock.hpp>

#include <cmath>

namespace ysq {

namespace {

bool isUsableStep(double seconds) {
    return std::isfinite(seconds) && seconds > 0.0;
}

bool isUsableScale(double scale) {
    return std::isfinite(scale) && scale >= 0.0;
}

}  // namespace

Clock::Clock() : Clock(Settings{}) {}

Clock::Clock(Settings settings) : m_settings(settings) {
    // Normalise once here so advance() can trust its own settings.
    const Settings defaults;
    if (!isUsableStep(m_settings.fixedStep)) {
        m_settings.fixedStep = defaults.fixedStep;
    }
    if (!isUsableScale(m_settings.timeScale)) {
        m_settings.timeScale = defaults.timeScale;
    }
    if (m_settings.maxStepsPerAdvance < 1) {
        m_settings.maxStepsPerAdvance = 1;
    }
}

int Clock::advance(double realDeltaSeconds) {
    // Both guards return before touching m_due, so an ignored delta is genuinely
    // ignored: steps a previous advance() made due survive it.
    if (!std::isfinite(realDeltaSeconds) || realDeltaSeconds < 0.0) {
        return m_due;
    }

    m_realTime += realDeltaSeconds;
    if (m_settings.paused) {
        return m_due;
    }

    m_due = 0;
    m_accumulator += realDeltaSeconds * m_settings.timeScale;

    // Subtracting per step rather than dividing keeps the remainder exact for
    // the next call, which is what makes a long run of small deltas add up.
    while (m_accumulator >= m_settings.fixedStep &&
           m_due < m_settings.maxStepsPerAdvance) {
        m_accumulator -= m_settings.fixedStep;
        ++m_due;
    }

    if (m_accumulator >= m_settings.fixedStep) {
        m_accumulator = 0.0;  // clamp tripped; drop the backlog
    }
    return m_due;
}

bool Clock::consumeStep() {
    if (m_due <= 0) {
        return false;
    }
    --m_due;
    m_simulationTime += m_settings.fixedStep;
    ++m_stepCount;
    return true;
}

void Clock::stepOnce() {
    m_simulationTime += m_settings.fixedStep;
    ++m_stepCount;
}

void Clock::pause() noexcept {
    m_settings.paused = true;
}

void Clock::resume() noexcept {
    m_settings.paused = false;
}

void Clock::setTimeScale(double scale) noexcept {
    if (isUsableScale(scale)) {
        m_settings.timeScale = scale;
    }
}

void Clock::setFixedStep(double seconds) noexcept {
    if (isUsableStep(seconds)) {
        m_settings.fixedStep = seconds;
    }
}

void Clock::reset() noexcept {
    m_due = 0;
    m_accumulator = 0.0;
    m_simulationTime = 0.0;
    m_realTime = 0.0;
    m_stepCount = 0;
}

double Clock::alpha() const noexcept {
    return m_accumulator / m_settings.fixedStep;
}

}  // namespace ysq
