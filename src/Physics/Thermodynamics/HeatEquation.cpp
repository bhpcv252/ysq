#include <Physics/Thermodynamics/HeatEquation.hpp>

#include <cstddef>
#include <vector>

namespace ysq {

HeatEquation1D::HeatEquation1D(std::size_t cellCount, double spacing, double diffusivity)
    : m_diffusivity(diffusivity), m_temperature(cellCount, spacing, 1) {}

std::size_t HeatEquation1D::cellCount() const noexcept {
    return m_temperature.cellCount();
}

double HeatEquation1D::spacing() const noexcept {
    return m_temperature.spacing();
}

double HeatEquation1D::diffusivity() const noexcept {
    return m_diffusivity;
}

void HeatEquation1D::setTemperature(std::size_t cell, double value) {
    m_temperature[static_cast<std::ptrdiff_t>(cell)] = value;
}

double HeatEquation1D::temperature(std::size_t cell) const {
    return m_temperature[static_cast<std::ptrdiff_t>(cell)];
}

double HeatEquation1D::stableTimeStep(double safetyFactor) const {
    const double dx = m_temperature.spacing();
    return safetyFactor * 0.5 * dx * dx / m_diffusivity;
}

void HeatEquation1D::step(double dt) {
    m_temperature.applyPeriodicBoundary();

    const auto n = static_cast<std::ptrdiff_t>(m_temperature.cellCount());
    const double dx = m_temperature.spacing();
    const double factor = m_diffusivity * dt / (dx * dx);

    std::vector<double> next(static_cast<std::size_t>(n));
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        next[static_cast<std::size_t>(i)] =
            m_temperature[i] + factor * (m_temperature[i + 1] - 2.0 * m_temperature[i] +
                                         m_temperature[i - 1]);
    }
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        m_temperature[i] = next[static_cast<std::size_t>(i)];
    }
}

double HeatEquation1D::totalHeat() const {
    const auto n = static_cast<std::ptrdiff_t>(m_temperature.cellCount());
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        total += m_temperature[i];
    }
    return total * m_temperature.spacing();
}

}  // namespace ysq
