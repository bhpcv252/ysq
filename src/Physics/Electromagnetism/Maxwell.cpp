#include <Physics/Electromagnetism/Maxwell.hpp>

#include <Physics/Electromagnetism/Field.hpp>
#include <Units/Constants.hpp>

namespace ysq {

MaxwellField1D::MaxwellField1D(std::size_t cellCount, double spacing)
    : m_electric(cellCount, spacing, 1), m_magnetic(cellCount, spacing, 1) {}

std::size_t MaxwellField1D::cellCount() const noexcept {
    return m_electric.cellCount();
}

double MaxwellField1D::spacing() const noexcept {
    return m_electric.spacing();
}

double MaxwellField1D::electricField(std::size_t cell) const {
    return m_electric[static_cast<std::ptrdiff_t>(cell)];
}

void MaxwellField1D::setElectricField(std::size_t cell, double value) {
    m_electric[static_cast<std::ptrdiff_t>(cell)] = value;
}

double MaxwellField1D::magneticField(std::size_t cell) const {
    return m_magnetic[static_cast<std::ptrdiff_t>(cell)];
}

void MaxwellField1D::setMagneticField(std::size_t cell, double value) {
    m_magnetic[static_cast<std::ptrdiff_t>(cell)] = value;
}

void MaxwellField1D::step(double dt) {
    const double c = constants::speedOfLight.value();
    const double dx = m_electric.spacing();
    const auto n = static_cast<std::ptrdiff_t>(m_electric.cellCount());

    // Bz half-step: needs Ey's ghost cells, since cell n-1's update reads
    // Ey[n], the wrapped-around copy of Ey[0].
    m_electric.applyPeriodicBoundary();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        m_magnetic[i] -= (dt / dx) * (m_electric[i + 1] - m_electric[i]);
    }

    // Ey full step, using the just-updated Bz: needs Bz's ghost cells, since
    // cell 0's update reads Bz[-1], the wrapped-around copy of Bz[n-1].
    m_magnetic.applyPeriodicBoundary();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        m_electric[i] -= c * c * (dt / dx) * (m_magnetic[i] - m_magnetic[i - 1]);
    }
}

double MaxwellField1D::totalEnergy() const {
    const double epsilon0 = constants::vacuumPermittivity.value();
    const double mu0 = constants::vacuumPermeability.value();
    const auto n = static_cast<std::ptrdiff_t>(m_electric.cellCount());

    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const double bPrev = m_magnetic[(i == 0) ? (n - 1) : (i - 1)];
        const double bAtEy = 0.5 * (m_magnetic[i] + bPrev);
        total += 0.5 * (epsilon0 * m_electric[i] * m_electric[i] + bAtEy * bAtEy / mu0);
    }
    return total * m_electric.spacing();
}

double magicTimeStep(double spacing) {
    return spacing / constants::speedOfLight.value();
}

}  // namespace ysq
