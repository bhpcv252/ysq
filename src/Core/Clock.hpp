#pragma once

#include <cstdint>

namespace ysq {

/// Simulation time, in seconds.
///
/// The clock never reads a wall clock. The host loop feeds it real elapsed time
/// and it hands back whole fixed steps to run, so integration is deterministic:
/// the same sequence of deltas always produces the same trajectory, and a test
/// can replay a run exactly without sleeping.
///
///     clock.advance(frame.lap().count());
///     while (clock.consumeStep()) {
///         world.integrate(clock.fixedStep());
///         // simulationTime() is now the time at the end of this step
///     }
///     renderer.draw(clock.alpha());
///
/// advance() decides how many steps are due; consumeStep() takes them one at a
/// time. Splitting the two is what lets the loop body see the simulation time of
/// the step it is running rather than only of the frame.
class Clock {
public:
    struct Settings {
        /// Simulation seconds advanced per step.
        double fixedStep = 1.0 / 60.0;
        /// Simulation seconds per real second.
        double timeScale = 1.0;
        /// Upper bound on steps returned by one advance(). Without it a long
        /// stall asks for more steps than the next frame can run, which makes
        /// the next stall longer still.
        int maxStepsPerAdvance = 8;
        bool paused = false;
    };

    // Two constructors rather than a defaulted argument: a nested class's member
    // initializers are not usable from a default argument of the enclosing class.
    Clock();
    explicit Clock(Settings settings);

    /// Feed one frame of real elapsed time. Returns the number of fixed steps
    /// now due, which is how many times consumeStep() will return true.
    ///
    /// Negative and non-finite deltas are ignored: they neither advance real
    /// time nor disturb steps already due. While paused, real time still
    /// accrues and the due count is left alone. Otherwise the count is recomputed
    /// from scratch, so steps not consumed before the next advance() are lost.
    /// When the step clamp trips the leftover is discarded, so simulation time
    /// falls behind real time rather than compounding.
    int advance(double realDeltaSeconds);

    /// Take one due step, advancing simulation time by fixedStep(). False once
    /// the steps from the last advance() are spent.
    bool consumeStep();

    [[nodiscard]] int stepsDue() const noexcept { return m_due; }

    /// Advance one step regardless of pause, the accumulator and the steps due.
    /// This is the single-step debugging control.
    void stepOnce();

    void pause() noexcept;
    void resume() noexcept;
    [[nodiscard]] bool paused() const noexcept { return m_settings.paused; }

    /// Non-finite and negative scales are ignored.
    void setTimeScale(double scale) noexcept;
    [[nodiscard]] double timeScale() const noexcept { return m_settings.timeScale; }

    /// Non-finite and non-positive steps are ignored.
    void setFixedStep(double seconds) noexcept;
    [[nodiscard]] double fixedStep() const noexcept { return m_settings.fixedStep; }

    /// Zeroes the times, the step count and the accumulator. Settings are kept.
    void reset() noexcept;

    [[nodiscard]] double simulationTime() const noexcept { return m_simulationTime; }
    /// Real time fed in since the last reset, including while paused.
    [[nodiscard]] double realTime() const noexcept { return m_realTime; }
    [[nodiscard]] std::uint64_t stepCount() const noexcept { return m_stepCount; }

    /// Unconsumed fraction of a step, in [0, 1). The blend factor for
    /// interpolating a render between the last two simulation states.
    [[nodiscard]] double alpha() const noexcept;

private:
    Settings m_settings;
    int m_due = 0;
    double m_accumulator = 0.0;
    double m_simulationTime = 0.0;
    double m_realTime = 0.0;
    std::uint64_t m_stepCount = 0;
};

}  // namespace ysq
