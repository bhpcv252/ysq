#include <Physics/Mechanics/Hermite.hpp>

#include <cassert>
#include <cmath>
#include <limits>

namespace ysq {

std::pair<Vec3, Vec3> hermitePredict(const Vec3& position, const Vec3& velocity,
                                     const Vec3& acceleration, const Vec3& jerk, double dt) {
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
    const Vec3 crackle = (accelerationDelta * 12.0 + (oldJerk + newJerk) * (6.0 * dt)) / dt3;

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

IndividualTimestepScheduler::IndividualTimestepScheduler(NBodyState positions,
                                                         NBodyState velocities,
                                                         NBodyState accelerations,
                                                         NBodyState jerks,
                                                         double initialTime, double eta,
                                                         double baseInterval)
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
        m_timestep[i] = hermiteTimestep(m_acceleration[i], m_jerk[i], m_eta, m_baseInterval);
    }
}

std::pair<std::size_t, double> IndividualTimestepScheduler::nextMover() const {
    std::size_t mover = 0;
    double moverNextTime = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < bodyCount(); ++i) {
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
