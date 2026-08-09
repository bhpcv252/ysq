#include <Physics/Acoustics/Acoustic.hpp>

namespace ysq {

AcousticField1D::AcousticField1D(std::size_t cellCount, double spacing,
                                 double mediumDensity, double soundSpeed)
    : m_pressure(cellCount, spacing, 1),
      m_velocity(cellCount, spacing, 1),
      m_density(mediumDensity),
      m_soundSpeed(soundSpeed) {}

std::size_t AcousticField1D::cellCount() const noexcept {
    return m_pressure.cellCount();
}

double AcousticField1D::spacing() const noexcept {
    return m_pressure.spacing();
}

double AcousticField1D::mediumDensity() const noexcept {
    return m_density;
}

double AcousticField1D::soundSpeed() const noexcept {
    return m_soundSpeed;
}

double AcousticField1D::pressure(std::size_t cell) const {
    return m_pressure[static_cast<std::ptrdiff_t>(cell)];
}

void AcousticField1D::setPressure(std::size_t cell, double value) {
    m_pressure[static_cast<std::ptrdiff_t>(cell)] = value;
}

double AcousticField1D::velocity(std::size_t cell) const {
    return m_velocity[static_cast<std::ptrdiff_t>(cell)];
}

void AcousticField1D::setVelocity(std::size_t cell, double value) {
    m_velocity[static_cast<std::ptrdiff_t>(cell)] = value;
}

void AcousticField1D::step(double dt) {
    const double dx = m_pressure.spacing();
    const auto n = static_cast<std::ptrdiff_t>(m_pressure.cellCount());

    // Velocity half-step: needs pressure's ghost cells, since cell n-1's
    // update reads pressure[n], the wrapped-around copy of pressure[0].
    m_pressure.applyPeriodicBoundary();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        m_velocity[i] -= (dt / dx) / m_density * (m_pressure[i + 1] - m_pressure[i]);
    }

    // Pressure full step, using the just-updated velocity: needs velocity's
    // ghost cells, since cell 0's update reads velocity[-1], the
    // wrapped-around copy of velocity[n-1].
    m_velocity.applyPeriodicBoundary();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        m_pressure[i] -= m_density * m_soundSpeed * m_soundSpeed * (dt / dx) *
                         (m_velocity[i] - m_velocity[i - 1]);
    }
}

double AcousticField1D::totalEnergy() const {
    const auto n = static_cast<std::ptrdiff_t>(m_pressure.cellCount());

    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const double velocityPrev = m_velocity[(i == 0) ? (n - 1) : (i - 1)];
        const double velocityAtPressure = 0.5 * (m_velocity[i] + velocityPrev);
        const double pressureValue = m_pressure[i];
        total += 0.5 * (pressureValue * pressureValue /
                            (m_density * m_soundSpeed * m_soundSpeed) +
                        m_density * velocityAtPressure * velocityAtPressure);
    }
    return total * m_pressure.spacing();
}

double magicAcousticTimeStep(double spacing, double soundSpeed) {
    return spacing / soundSpeed;
}

}  // namespace ysq
