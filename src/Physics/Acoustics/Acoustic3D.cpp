#include <Physics/Acoustics/Acoustic3D.hpp>

#include <cmath>
#include <cstddef>

namespace ysq {

AcousticField3D::AcousticField3D(std::size_t cellCountX, std::size_t cellCountY,
                                 std::size_t cellCountZ, double spacing,
                                 double mediumDensity, double soundSpeed)
    : m_pressure(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_velocityX(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_velocityY(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_velocityZ(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_density(mediumDensity),
      m_soundSpeed(soundSpeed) {}

std::size_t AcousticField3D::cellCountX() const noexcept {
    return m_pressure.cellCountX();
}

std::size_t AcousticField3D::cellCountY() const noexcept {
    return m_pressure.cellCountY();
}

std::size_t AcousticField3D::cellCountZ() const noexcept {
    return m_pressure.cellCountZ();
}

double AcousticField3D::spacing() const noexcept {
    return m_pressure.spacing();
}

double AcousticField3D::mediumDensity() const noexcept {
    return m_density;
}

double AcousticField3D::soundSpeed() const noexcept {
    return m_soundSpeed;
}

double AcousticField3D::pressure(std::size_t i, std::size_t j, std::size_t k) const {
    return m_pressure(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                      static_cast<std::ptrdiff_t>(k));
}

void AcousticField3D::setPressure(std::size_t i, std::size_t j, std::size_t k,
                                  double value) {
    m_pressure(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
               static_cast<std::ptrdiff_t>(k)) = value;
}

double AcousticField3D::velocityX(std::size_t i, std::size_t j, std::size_t k) const {
    return m_velocityX(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                       static_cast<std::ptrdiff_t>(k));
}

void AcousticField3D::setVelocityX(std::size_t i, std::size_t j, std::size_t k,
                                   double value) {
    m_velocityX(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                static_cast<std::ptrdiff_t>(k)) = value;
}

double AcousticField3D::velocityY(std::size_t i, std::size_t j, std::size_t k) const {
    return m_velocityY(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                       static_cast<std::ptrdiff_t>(k));
}

void AcousticField3D::setVelocityY(std::size_t i, std::size_t j, std::size_t k,
                                   double value) {
    m_velocityY(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                static_cast<std::ptrdiff_t>(k)) = value;
}

double AcousticField3D::velocityZ(std::size_t i, std::size_t j, std::size_t k) const {
    return m_velocityZ(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                       static_cast<std::ptrdiff_t>(k));
}

void AcousticField3D::setVelocityZ(std::size_t i, std::size_t j, std::size_t k,
                                   double value) {
    m_velocityZ(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                static_cast<std::ptrdiff_t>(k)) = value;
}

double AcousticField3D::stableTimeStep(double courantFactor) const {
    return courantFactor * m_pressure.spacing() / (m_soundSpeed * std::sqrt(3.0));
}

void AcousticField3D::step(double dt) {
    const auto nx = static_cast<std::ptrdiff_t>(m_pressure.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_pressure.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_pressure.cellCountZ());
    const double dx = m_pressure.spacing();
    const double velocityFactor = (dt / dx) / m_density;

    m_pressure.applyPeriodicBoundary();
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                const double p = m_pressure(i, j, k);
                m_velocityX(i, j, k) -= velocityFactor * (m_pressure(i + 1, j, k) - p);
                m_velocityY(i, j, k) -= velocityFactor * (m_pressure(i, j + 1, k) - p);
                m_velocityZ(i, j, k) -= velocityFactor * (m_pressure(i, j, k + 1) - p);
            }
        }
    }

    m_velocityX.applyPeriodicBoundary();
    m_velocityY.applyPeriodicBoundary();
    m_velocityZ.applyPeriodicBoundary();
    const double pressureFactor = m_density * m_soundSpeed * m_soundSpeed * (dt / dx);
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                const double divergence =
                    (m_velocityX(i, j, k) - m_velocityX(i - 1, j, k)) +
                    (m_velocityY(i, j, k) - m_velocityY(i, j - 1, k)) +
                    (m_velocityZ(i, j, k) - m_velocityZ(i, j, k - 1));
                m_pressure(i, j, k) -= pressureFactor * divergence;
            }
        }
    }
}

double AcousticField3D::totalEnergy() const {
    const auto nx = static_cast<std::ptrdiff_t>(m_pressure.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_pressure.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_pressure.cellCountZ());
    const double h = m_pressure.spacing();
    const double compressibility = 1.0 / (m_density * m_soundSpeed * m_soundSpeed);

    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                const double p = m_pressure(i, j, k);
                const double uxAt =
                    0.5 * (m_velocityX(i, j, k) + m_velocityX(i - 1, j, k));
                const double uyAt =
                    0.5 * (m_velocityY(i, j, k) + m_velocityY(i, j - 1, k));
                const double uzAt =
                    0.5 * (m_velocityZ(i, j, k) + m_velocityZ(i, j, k - 1));
                const double kinetic =
                    m_density * (uxAt * uxAt + uyAt * uyAt + uzAt * uzAt);
                total += 0.5 * (p * p * compressibility + kinetic);
            }
        }
    }
    return total * h * h * h;
}

}  // namespace ysq
