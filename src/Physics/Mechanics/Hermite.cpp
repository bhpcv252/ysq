#include <Physics/Mechanics/Hermite.hpp>

#include <Compute/ComputeBackend.hpp>

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace ysq {

namespace {

// nextMover() runs once per single-body update, not once per timestep, so
// dispatch overhead is paid far more often here than in a per-step field
// like Newtonian.cpp's -- reflected in the measured value being an order
// of magnitude higher. Measured on the development machine (Apple
// Silicon, Metal backend) by benchmarks/compute_thresholds.cpp
// (`minIndex`, the same simple O(n) reduction scan both this file's
// below-threshold path and Compute::CpuBackend's own reference use, so no
// adjustment for a mismatched CPU shape is needed here the way
// Gravity/Newtonian.cpp's and Fluids/SPH.cpp's own thresholds needed):
// the GPU path never won up to 16384 bodies, the largest count tried, so
// this is a measured floor (double the largest size tried), not an
// observed crossover. Re-run the benchmark and update this if the
// reference machine or backend ever changes.
constexpr std::size_t kGpuDispatchThreshold = 32768;

}  // namespace

std::pair<Vec3, Vec3> hermitePredict(const Vec3& position, const Vec3& velocity,
                                     const Vec3& acceleration, const Vec3& jerk,
                                     double dt) {
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const Vec3 predictedPosition =
        position + velocity * dt + acceleration * (dt2 / 2.0) + jerk * (dt3 / 6.0);
    const Vec3 predictedVelocity = velocity + acceleration * dt + jerk * (dt2 / 2.0);
    return {predictedPosition, predictedVelocity};
}

std::pair<Vec3, Vec3> hermiteCorrect(const Vec3& oldAcceleration, const Vec3& oldJerk,
                                     const Vec3& newAcceleration, const Vec3& newJerk,
                                     double dt, const Vec3& predictedPosition,
                                     const Vec3& predictedVelocity) {
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;
    const double dt5 = dt4 * dt;

    // The acceleration's own 2nd derivative ("snap") and 3rd derivative
    // ("crackle") across the step, fit from the (acceleration, jerk) known
    // at both endpoints (Makino & Aarseth 1992).
    const Vec3 accelerationDelta = oldAcceleration - newAcceleration;
    const Vec3 snap =
        (accelerationDelta * -6.0 - (oldJerk * 4.0 + newJerk * 2.0) * dt) / dt2;
    const Vec3 crackle =
        (accelerationDelta * 12.0 + (oldJerk + newJerk) * (6.0 * dt)) / dt3;

    const Vec3 correctedPosition =
        predictedPosition + snap * (dt4 / 24.0) + crackle * (dt5 / 120.0);
    const Vec3 correctedVelocity =
        predictedVelocity + snap * (dt3 / 6.0) + crackle * (dt4 / 24.0);
    return {correctedPosition, correctedVelocity};
}

double hermiteTimestep(const Vec3& acceleration, const Vec3& jerk, double eta,
                       double baseInterval) {
    const double a = length(acceleration);
    const double j = length(jerk);
    if (a <= 0.0 || j <= 0.0) {
        return baseInterval;
    }

    const double raw = std::sqrt(eta * a / j);
    if (raw >= baseInterval) {
        return baseInterval;
    }

    // Largest power-of-two fraction of baseInterval that is still <= raw:
    // halving is bounded (baseInterval and raw are both finite and
    // positive here) rather than open-ended, but capped anyway against a
    // pathologically tiny raw relative to baseInterval.
    double dt = baseInterval;
    constexpr int kMaxHalvings = 60;
    for (int i = 0; i < kMaxHalvings && dt > raw; ++i) {
        dt *= 0.5;
    }
    return dt;
}

IndividualTimestepScheduler::IndividualTimestepScheduler(
    NBodyState positions, NBodyState velocities, NBodyState accelerations,
    NBodyState jerks, double initialTime, double eta, double baseInterval)
    : m_position(std::move(positions)),
      m_velocity(std::move(velocities)),
      m_acceleration(std::move(accelerations)),
      m_jerk(std::move(jerks)),
      m_currentTime(initialTime),
      m_eta(eta),
      m_baseInterval(baseInterval) {
    assert(m_velocity.size() == m_position.size());
    assert(m_acceleration.size() == m_position.size());
    assert(m_jerk.size() == m_position.size());

    const std::size_t n = m_position.size();
    m_lastUpdateTime.assign(n, initialTime);
    m_timestep.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        m_timestep[i] =
            hermiteTimestep(m_acceleration[i], m_jerk[i], m_eta, m_baseInterval);
    }
}

std::pair<std::size_t, double> IndividualTimestepScheduler::nextMover() const {
    const std::size_t n = bodyCount();

    if (n >= kGpuDispatchThreshold) {
        // Cast to float relative to m_currentTime, not the raw absolute
        // time: over a long run m_lastUpdateTime grows arbitrarily large
        // while every body's own step stays bounded by baseInterval, and
        // float32 has only about 7 decimal digits, so casting the raw sum
        // directly could easily scramble which body is genuinely soonest
        // once accumulated time is large enough. The offset keeps every
        // value small (a few times baseInterval at most, by construction:
        // nothing here lets a body's own next time fall far behind
        // m_currentTime), so the ordering minIndex finds is trustworthy;
        // the actual returned time is still read back from the double
        // arrays at the winning index, never reconstructed from the float
        // reduction.
        std::vector<float> relativeNextTimes(n);
        for (std::size_t i = 0; i < n; ++i) {
            relativeNextTimes[i] =
                static_cast<float>(m_lastUpdateTime[i] + m_timestep[i] - m_currentTime);
        }
        const std::size_t mover = defaultBackend().minIndex(relativeNextTimes);
        return {mover, m_lastUpdateTime[mover] + m_timestep[mover]};
    }

    std::size_t mover = 0;
    double moverNextTime = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double nextTime = m_lastUpdateTime[i] + m_timestep[i];
        if (nextTime < moverNextTime) {
            moverNextTime = nextTime;
            mover = i;
        }
    }
    return {mover, moverNextTime};
}

std::pair<Vec3, Vec3> IndividualTimestepScheduler::predictedState(std::size_t bodyIndex,
                                                                  double atTime) const {
    return hermitePredict(m_position[bodyIndex], m_velocity[bodyIndex],
                          m_acceleration[bodyIndex], m_jerk[bodyIndex],
                          atTime - m_lastUpdateTime[bodyIndex]);
}

}  // namespace ysq
