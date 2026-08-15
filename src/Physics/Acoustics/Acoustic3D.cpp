#include <Physics/Acoustics/Acoustic3D.hpp>

#include <Compute/ComputeBackend.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ysq {

namespace {

// Measured on the development machine (Apple Silicon, Metal backend) by
// benchmarks/compute_thresholds.cpp: the GPU path never won up to 128^3
// (~2.1M cells), the largest grid tried, matching
// HeatEquation3D.cpp's own measured floor and for the same reason (O(n)
// per cell, no O(n^2) inner loop, and this kernel's own two-pass velocity-
// then-pressure structure adds proportionally more per-cell CPU work too,
// keeping it CPU-competitive at least as long). Also keeps every existing
// grid size in tests/unit/acoustics_acoustic3d.cpp on the exact
// double-precision CPU path. Re-run the benchmark and update this if the
// reference machine or backend ever changes.
constexpr std::size_t kGpuDispatchThreshold = 4194304;

}  // namespace

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
    const std::size_t nxCount = m_pressure.cellCountX();
    const std::size_t nyCount = m_pressure.cellCountY();
    const std::size_t nzCount = m_pressure.cellCountZ();
    const double dx = m_pressure.spacing();
    const double velocityFactor = (dt / dx) / m_density;
    const double pressureFactorGpu = m_density * m_soundSpeed * m_soundSpeed * (dt / dx);

    if (nxCount * nyCount * nzCount >= kGpuDispatchThreshold) {
        const std::size_t total = nxCount * nyCount * nzCount;
        std::vector<float> pressure(total);
        std::vector<float> velX(total);
        std::vector<float> velY(total);
        std::vector<float> velZ(total);
        for (std::size_t i = 0; i < nxCount; ++i) {
            for (std::size_t j = 0; j < nyCount; ++j) {
                for (std::size_t k = 0; k < nzCount; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    const std::size_t flat = (i * nyCount + j) * nzCount + k;
                    pressure[flat] = static_cast<float>(m_pressure(pi, pj, pk));
                    velX[flat] = static_cast<float>(m_velocityX(pi, pj, pk));
                    velY[flat] = static_cast<float>(m_velocityY(pi, pj, pk));
                    velZ[flat] = static_cast<float>(m_velocityZ(pi, pj, pk));
                }
            }
        }

        std::vector<float> nextPressure(total);
        std::vector<float> nextVelX(total);
        std::vector<float> nextVelY(total);
        std::vector<float> nextVelZ(total);
        defaultBackend().acoustic3DStep(pressure, velX, velY, velZ, nxCount, nyCount,
                                        nzCount, static_cast<float>(velocityFactor),
                                        static_cast<float>(pressureFactorGpu),
                                        nextPressure, nextVelX, nextVelY, nextVelZ);

        for (std::size_t i = 0; i < nxCount; ++i) {
            for (std::size_t j = 0; j < nyCount; ++j) {
                for (std::size_t k = 0; k < nzCount; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    const std::size_t flat = (i * nyCount + j) * nzCount + k;
                    m_pressure(pi, pj, pk) = static_cast<double>(nextPressure[flat]);
                    m_velocityX(pi, pj, pk) = static_cast<double>(nextVelX[flat]);
                    m_velocityY(pi, pj, pk) = static_cast<double>(nextVelY[flat]);
                    m_velocityZ(pi, pj, pk) = static_cast<double>(nextVelZ[flat]);
                }
            }
        }
        return;
    }

    const auto nx = static_cast<std::ptrdiff_t>(nxCount);
    const auto ny = static_cast<std::ptrdiff_t>(nyCount);
    const auto nz = static_cast<std::ptrdiff_t>(nzCount);

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
