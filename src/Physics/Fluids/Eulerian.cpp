#include <Physics/Fluids/Eulerian.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ysq {

namespace {

struct Conserved {
    double density;
    double momentum;
    double energy;
};

struct Flux {
    double density;
    double momentum;
    double energy;
};

[[nodiscard]] double pressureFrom(const Conserved& state, double gamma) {
    const double velocity = state.momentum / state.density;
    const double kinetic = 0.5 * state.density * velocity * velocity;
    return (gamma - 1.0) * (state.energy - kinetic);
}

[[nodiscard]] double soundSpeed(double density, double pressure, double gamma) {
    return std::sqrt(gamma * pressure / density);
}

[[nodiscard]] Flux fluxOf(const Conserved& state, double gamma) {
    const double velocity = state.momentum / state.density;
    const double pressure = pressureFrom(state, gamma);
    return {state.momentum, state.momentum * velocity + pressure,
            velocity * (state.energy + pressure)};
}

/// The Rusanov (local Lax-Friedrichs) flux: the average of the two states'
/// physical fluxes, stabilized by a diffusive term scaled to the fastest
/// signal speed either state can carry.
[[nodiscard]] Flux rusanovFlux(const Conserved& left, const Conserved& right,
                               double gamma) {
    const Flux fluxLeft = fluxOf(left, gamma);
    const Flux fluxRight = fluxOf(right, gamma);

    const double velocityLeft = left.momentum / left.density;
    const double velocityRight = right.momentum / right.density;
    const double soundLeft = soundSpeed(left.density, pressureFrom(left, gamma), gamma);
    const double soundRight =
        soundSpeed(right.density, pressureFrom(right, gamma), gamma);
    const double maxSpeed = std::max(std::abs(velocityLeft) + soundLeft,
                                     std::abs(velocityRight) + soundRight);

    return {0.5 * (fluxLeft.density + fluxRight.density) -
                0.5 * maxSpeed * (right.density - left.density),
            0.5 * (fluxLeft.momentum + fluxRight.momentum) -
                0.5 * maxSpeed * (right.momentum - left.momentum),
            0.5 * (fluxLeft.energy + fluxRight.energy) -
                0.5 * maxSpeed * (right.energy - left.energy)};
}

}  // namespace

EulerianFluid1D::EulerianFluid1D(std::size_t cellCount, double spacing,
                                 double adiabaticIndex)
    : m_gamma(adiabaticIndex),
      m_density(cellCount, spacing, 1),
      m_momentum(cellCount, spacing, 1),
      m_energy(cellCount, spacing, 1) {}

std::size_t EulerianFluid1D::cellCount() const noexcept {
    return m_density.cellCount();
}

double EulerianFluid1D::spacing() const noexcept {
    return m_density.spacing();
}

double EulerianFluid1D::adiabaticIndex() const noexcept {
    return m_gamma;
}

void EulerianFluid1D::setState(std::size_t cell, double density, double velocity,
                               double pressure) {
    const auto i = static_cast<std::ptrdiff_t>(cell);
    const double kinetic = 0.5 * density * velocity * velocity;
    const double internal = pressure / (m_gamma - 1.0);

    m_density[i] = density;
    m_momentum[i] = density * velocity;
    m_energy[i] = internal + kinetic;
}

double EulerianFluid1D::density(std::size_t cell) const {
    return m_density[static_cast<std::ptrdiff_t>(cell)];
}

double EulerianFluid1D::velocity(std::size_t cell) const {
    const auto i = static_cast<std::ptrdiff_t>(cell);
    return m_momentum[i] / m_density[i];
}

double EulerianFluid1D::pressure(std::size_t cell) const {
    const auto i = static_cast<std::ptrdiff_t>(cell);
    const Conserved state{m_density[i], m_momentum[i], m_energy[i]};
    return pressureFrom(state, m_gamma);
}

double EulerianFluid1D::stableTimeStep(double courantNumber) const {
    const auto n = static_cast<std::ptrdiff_t>(m_density.cellCount());
    double maxSpeed = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const Conserved state{m_density[i], m_momentum[i], m_energy[i]};
        const double velocity = state.momentum / state.density;
        const double sound =
            soundSpeed(state.density, pressureFrom(state, m_gamma), m_gamma);
        maxSpeed = std::max(maxSpeed, std::abs(velocity) + sound);
    }
    return courantNumber * m_density.spacing() / maxSpeed;
}

void EulerianFluid1D::step(double dt) {
    m_density.applyPeriodicBoundary();
    m_momentum.applyPeriodicBoundary();
    m_energy.applyPeriodicBoundary();

    const auto n = static_cast<std::ptrdiff_t>(m_density.cellCount());
    std::vector<Flux> fluxes(static_cast<std::size_t>(n) + 1);
    for (std::ptrdiff_t i = 0; i <= n; ++i) {
        const Conserved left{m_density[i - 1], m_momentum[i - 1], m_energy[i - 1]};
        const Conserved right{m_density[i], m_momentum[i], m_energy[i]};
        fluxes[static_cast<std::size_t>(i)] = rusanovFlux(left, right, m_gamma);
    }

    const double dx = m_density.spacing();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const Flux& fluxLeft = fluxes[static_cast<std::size_t>(i)];
        const Flux& fluxRight = fluxes[static_cast<std::size_t>(i + 1)];
        m_density[i] -= (dt / dx) * (fluxRight.density - fluxLeft.density);
        m_momentum[i] -= (dt / dx) * (fluxRight.momentum - fluxLeft.momentum);
        m_energy[i] -= (dt / dx) * (fluxRight.energy - fluxLeft.energy);
    }
}

double EulerianFluid1D::totalMass() const {
    const auto n = static_cast<std::ptrdiff_t>(m_density.cellCount());
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        total += m_density[i];
    }
    return total * m_density.spacing();
}

double EulerianFluid1D::totalMomentum() const {
    const auto n = static_cast<std::ptrdiff_t>(m_momentum.cellCount());
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        total += m_momentum[i];
    }
    return total * m_momentum.spacing();
}

double EulerianFluid1D::totalEnergy() const {
    const auto n = static_cast<std::ptrdiff_t>(m_energy.cellCount());
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        total += m_energy[i];
    }
    return total * m_energy.spacing();
}

}  // namespace ysq
