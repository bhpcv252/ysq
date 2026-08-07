#pragma once

#include <Physics/Mechanics/Dynamics.hpp>

#include <Math/Vector3.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace ysq {

/// A 4th-order predictor-corrector integration scheme (Makino & Aarseth
/// 1992) and the per-body scheduler built on it that lets every body in an
/// N-body system advance at its own step size rather than all sharing one.
///
/// A single shared step forces every body through whichever one of them
/// moves fastest: correct for that body, wasteful for every slower one,
/// and there is no shared value of the step that is both cheap and right
/// for bodies whose dynamical timescales differ by orders of magnitude.
/// Individual timesteps only work because Hermite's predictor uses each
/// body's own last known acceleration *and* jerk (its rate of change) to
/// extrapolate a physically reasonable position for a body that has not
/// been updated in a while -- which is also exactly what lets a body's own
/// step size be chosen from how fast its acceleration is itself changing
/// (the Aarseth criterion below), rather than from any notion of "whose
/// orbit this is."
///
/// This lives in Mechanics, not Gravity: nothing here knows what jerk
/// *is*, physically. It is force-law-agnostic in the same sense
/// Dynamics.hpp's NBodyState already is -- Gravity/Newtonian.hpp and
/// Gravity/PostNewtonian.hpp supply the concrete gravity (and 1PN) jerk
/// this is evaluated against.
///
/// This is 4th-order accurate but **not symplectic** -- the same category
/// RK4 is in. Its per-step error is small, and each body is resolved at
/// its own appropriate scale rather than everyone paying for the fastest
/// one, but there is no guarantee against slow energy drift over an
/// arbitrarily long run the way VelocityVerletStepper's bounded error is.

/// Predicts one body's position and velocity forward from its own last
/// known state to `dt` later, by a third-order Taylor expansion in
/// acceleration and jerk. This is what lets another body, evaluating its
/// own force, use a physically reasonable estimate of a not-yet-updated
/// body's position rather than its stale, last-computed one.
[[nodiscard]] std::pair<Vec3, Vec3> hermitePredict(const Vec3& position,
                                                   const Vec3& velocity,
                                                   const Vec3& acceleration,
                                                   const Vec3& jerk, double dt);

/// Corrects a predicted (position, velocity) once a fresh acceleration and
/// jerk are known at the predicted state: fits the acceleration's own 2nd
/// and 3rd derivatives (snap and crackle) from the endpoints' (old, new)
/// acceleration/jerk pairs, then folds those into the prediction to reach
/// full 4th order. `dt` is the same interval `hermitePredict` was called
/// with; `newAcceleration`/`newJerk` are evaluated at
/// `predictedPosition`/`predictedVelocity`.
[[nodiscard]] std::pair<Vec3, Vec3> hermiteCorrect(
    const Vec3& oldAcceleration, const Vec3& oldJerk, const Vec3& newAcceleration,
    const Vec3& newJerk, double dt, const Vec3& predictedPosition,
    const Vec3& predictedVelocity);

/// The Aarseth (1985) criterion: a body's own next step shrinks where its
/// acceleration is large *and* rapidly changing (a close encounter, any
/// sharp perturbation), and grows where it is calm and slowly varying --
/// automatically, from the body's own current dynamics, with nothing
/// naming an orbit, a parent, or a period anywhere in the formula.
///
///     dt = sqrt(eta * |a| / |jerk|)
///
/// Rounded down to the nearest power-of-two fraction of `baseInterval`, so
/// bodies with similar dynamical timescales land on shared update times
/// (standard "block timestep" practice) instead of every body drifting to
/// an arbitrary time of its own that nothing else ever lines up with.
/// `baseInterval` itself (and its own reciprocal, the largest step any
/// body may take) is the caller's choice, not derived here. Falls back to
/// `baseInterval` wherever the jerk is zero -- nothing about this body's
/// current dynamics asks for a smaller step.
[[nodiscard]] double hermiteTimestep(const Vec3& acceleration, const Vec3& jerk, double eta,
                                     double baseInterval);

/// What the scheduler asks its caller for every time a body's turn comes
/// up: that one body's own (acceleration, jerk), given every body's
/// position and velocity already predicted to the same instant.
/// Physics/Gravity supplies the concrete gravity (+ 1PN) implementation;
/// this concept only names the shape.
template <class F>
concept IndividualJerkField =
    requires(const F& f, std::size_t bodyIndex, const NBodyState& predictedPositions,
            const NBodyState& predictedVelocities) {
        { f(bodyIndex, predictedPositions, predictedVelocities) }
            -> std::convertible_to<std::pair<Vec3, Vec3>>;
    };

/// Owns every body's own (last-update time, step size, position, velocity,
/// acceleration, jerk) and advances them individually rather than in
/// lockstep.
class IndividualTimestepScheduler {
public:
    /// `accelerations`/`jerks` are each body's own value at `initialTime`,
    /// already evaluated by the caller against whichever IndividualJerkField
    /// it is about to drive advanceTo() with -- the scheduler does not
    /// evaluate a jerk field itself until advanceTo() is first called.
    IndividualTimestepScheduler(NBodyState positions, NBodyState velocities,
                                NBodyState accelerations, NBodyState jerks,
                                double initialTime, double eta, double baseInterval);

    /// Advances the scheduler's own global time toward `targetTime`, one
    /// body at a time: whichever body's own (last update time + its own
    /// step) is soonest goes next, every other body is predicted to that
    /// instant via hermitePredict(), `jerkField` is asked for the mover's
    /// own (acceleration, jerk) against those predictions, hermiteCorrect()
    /// refines the mover's own position and velocity, and its own next step
    /// is re-chosen from hermiteTimestep(). Stops at `targetTime` or after
    /// `maxUpdates` single-body updates, whichever comes first -- the
    /// individual-update analogue of a uniform stepper's own
    /// maxStepsPerAdvance, guarding the same runaway-catch-up failure mode.
    template <IndividualJerkField JerkField>
    void advanceTo(const JerkField& jerkField, double targetTime, int maxUpdates);

    /// How many single-body updates the most recent advanceTo() call
    /// actually performed, out of the maxUpdates it was allowed: read
    /// this after calling it to tell "reached targetTime comfortably" (well
    /// under maxUpdates) apart from "hit the cap and fell behind" (exactly
    /// maxUpdates) -- e.g. for a diagnostic readout while tuning
    /// maxUpdates itself.
    [[nodiscard]] int lastAdvanceUpdateCount() const noexcept {
        return m_lastAdvanceUpdateCount;
    }

    /// Every body's position and velocity predicted to `atTime`, regardless
    /// of whether its own next scheduled update lands exactly there.
    /// Rendering, trails, and any energy/momentum diagnostic should all
    /// read through this rather than a body's own last true update: total
    /// system energy is only meaningful when every body is read at the
    /// same instant, and bodies advancing at their own individual rate are
    /// essentially never all exactly synchronized except incidentally.
    ///
    /// **`atTime` must not be allowed to drift arbitrarily far ahead of
    /// `currentTime()`.** The prediction is a cubic (3rd-order) Taylor
    /// extrapolation from a body's own last update, valid only for a gap
    /// comparable to that body's own step -- if a caller keeps requesting
    /// `atTime` values that outpace what `advanceTo()` actually reaches
    /// (say, because `maxUpdates` isn't enough to keep up), the gap grows
    /// every call and the extrapolation diverges. Callers should clamp
    /// their own notion of "now" to `currentTime()` after every
    /// `advanceTo()`, the same way `Core::Clock` lets simulation time fall
    /// behind real time rather than let an unconsumed backlog compound.
    [[nodiscard]] std::pair<Vec3, Vec3> predictedState(std::size_t bodyIndex,
                                                       double atTime) const;

    [[nodiscard]] double currentTime() const noexcept { return m_currentTime; }
    [[nodiscard]] std::size_t bodyCount() const noexcept { return m_position.size(); }

private:
    /// The body whose (last update time + its own step) is smallest, and
    /// that value -- shared by advanceTo()'s loop condition and its choice
    /// of who moves next.
    [[nodiscard]] std::pair<std::size_t, double> nextMover() const;

    NBodyState m_position;
    NBodyState m_velocity;
    NBodyState m_acceleration;
    NBodyState m_jerk;
    std::vector<double> m_lastUpdateTime;
    std::vector<double> m_timestep;
    double m_currentTime;
    double m_eta;
    double m_baseInterval;
    int m_lastAdvanceUpdateCount = 0;
};

template <IndividualJerkField JerkField>
void IndividualTimestepScheduler::advanceTo(const JerkField& jerkField, double targetTime,
                                            int maxUpdates) {
    m_lastAdvanceUpdateCount = 0;
    for (int update = 0; update < maxUpdates; ++update) {
        const auto [mover, moverNextTime] = nextMover();
        if (moverNextTime > targetTime) {
            // Genuinely caught up, not merely out of budget: every body's
            // own next update is still further out than targetTime, so
            // reaching targetTime cost no work at all. currentTime() has
            // to advance to targetTime here regardless -- if it stayed at
            // whatever the last *actual* update happened to be, a caller
            // driving a slow-moving clock (e.g. requesting a small amount
            // of new simulated time each frame, far less than any body's
            // own step) could never accumulate past that stale value, no
            // matter how many times it called this. Safe to do: every
            // body's own gap (targetTime - its own lastUpdateTime) is, by
            // this branch's own condition, still under that body's own
            // step, so predictedState()'s small-gap assumption still holds.
            m_currentTime = std::max(m_currentTime, targetTime);
            break;
        }
        ++m_lastAdvanceUpdateCount;

        // Every body -- including the mover -- predicted to the instant the
        // mover is about to be updated at, each from its own last known
        // state: this is what stands in for a not-yet-updated body's
        // current position when the mover's own force is evaluated below.
        const std::size_t n = bodyCount();
        NBodyState predictedPositions(n);
        NBodyState predictedVelocities(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto [p, v] =
                hermitePredict(m_position[i], m_velocity[i], m_acceleration[i], m_jerk[i],
                               moverNextTime - m_lastUpdateTime[i]);
            predictedPositions[i] = p;
            predictedVelocities[i] = v;
        }

        const auto [newAcceleration, newJerk] =
            jerkField(mover, predictedPositions, predictedVelocities);

        const double dt = moverNextTime - m_lastUpdateTime[mover];
        const auto [correctedPosition, correctedVelocity] =
            hermiteCorrect(m_acceleration[mover], m_jerk[mover], newAcceleration, newJerk,
                          dt, predictedPositions[mover], predictedVelocities[mover]);

        m_position[mover] = correctedPosition;
        m_velocity[mover] = correctedVelocity;
        m_acceleration[mover] = newAcceleration;
        m_jerk[mover] = newJerk;
        m_lastUpdateTime[mover] = moverNextTime;
        m_timestep[mover] = hermiteTimestep(newAcceleration, newJerk, m_eta, m_baseInterval);
        m_currentTime = moverNextTime;
    }
}

}  // namespace ysq
