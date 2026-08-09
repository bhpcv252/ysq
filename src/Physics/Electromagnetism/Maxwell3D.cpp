#include <Physics/Electromagnetism/Maxwell3D.hpp>

#include <Physics/Electromagnetism/Field.hpp>
#include <Units/Constants.hpp>

#include <cmath>
#include <cstddef>

namespace ysq {

MaxwellField3D::MaxwellField3D(std::size_t cellCountX, std::size_t cellCountY,
                               std::size_t cellCountZ, double spacing)
    : m_ex(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_ey(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_ez(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_bx(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_by(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_bz(cellCountX, cellCountY, cellCountZ, spacing, 1) {}

std::size_t MaxwellField3D::cellCountX() const noexcept {
    return m_ex.cellCountX();
}

std::size_t MaxwellField3D::cellCountY() const noexcept {
    return m_ex.cellCountY();
}

std::size_t MaxwellField3D::cellCountZ() const noexcept {
    return m_ex.cellCountZ();
}

double MaxwellField3D::spacing() const noexcept {
    return m_ex.spacing();
}

namespace {

std::ptrdiff_t idx(std::size_t i) {
    return static_cast<std::ptrdiff_t>(i);
}

}  // namespace

double MaxwellField3D::electricFieldX(std::size_t i, std::size_t j, std::size_t k) const {
    return m_ex(idx(i), idx(j), idx(k));
}
void MaxwellField3D::setElectricFieldX(std::size_t i, std::size_t j, std::size_t k,
                                       double value) {
    m_ex(idx(i), idx(j), idx(k)) = value;
}
double MaxwellField3D::electricFieldY(std::size_t i, std::size_t j, std::size_t k) const {
    return m_ey(idx(i), idx(j), idx(k));
}
void MaxwellField3D::setElectricFieldY(std::size_t i, std::size_t j, std::size_t k,
                                       double value) {
    m_ey(idx(i), idx(j), idx(k)) = value;
}
double MaxwellField3D::electricFieldZ(std::size_t i, std::size_t j, std::size_t k) const {
    return m_ez(idx(i), idx(j), idx(k));
}
void MaxwellField3D::setElectricFieldZ(std::size_t i, std::size_t j, std::size_t k,
                                       double value) {
    m_ez(idx(i), idx(j), idx(k)) = value;
}

double MaxwellField3D::magneticFieldX(std::size_t i, std::size_t j, std::size_t k) const {
    return m_bx(idx(i), idx(j), idx(k));
}
void MaxwellField3D::setMagneticFieldX(std::size_t i, std::size_t j, std::size_t k,
                                       double value) {
    m_bx(idx(i), idx(j), idx(k)) = value;
}
double MaxwellField3D::magneticFieldY(std::size_t i, std::size_t j, std::size_t k) const {
    return m_by(idx(i), idx(j), idx(k));
}
void MaxwellField3D::setMagneticFieldY(std::size_t i, std::size_t j, std::size_t k,
                                       double value) {
    m_by(idx(i), idx(j), idx(k)) = value;
}
double MaxwellField3D::magneticFieldZ(std::size_t i, std::size_t j, std::size_t k) const {
    return m_bz(idx(i), idx(j), idx(k));
}
void MaxwellField3D::setMagneticFieldZ(std::size_t i, std::size_t j, std::size_t k,
                                       double value) {
    m_bz(idx(i), idx(j), idx(k)) = value;
}

double MaxwellField3D::stableTimeStep(double courantFactor) const {
    const double c = constants::speedOfLight.value();
    return courantFactor * m_ex.spacing() / (c * std::sqrt(3.0));
}

void MaxwellField3D::step(double dt) {
    const double c = constants::speedOfLight.value();
    const auto nx = static_cast<std::ptrdiff_t>(m_ex.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_ex.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_ex.cellCountZ());
    const double h = m_ex.spacing();

    // B half-step: needs each E field's ghost cells on the +1 side, since
    // (e.g.) cell n-1's Bx update reads Ez[n], the wrapped-around copy of
    // Ez[0].
    m_ex.applyPeriodicBoundary();
    m_ey.applyPeriodicBoundary();
    m_ez.applyPeriodicBoundary();
    const double bFactor = dt / h;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                m_bx(i, j, k) -= bFactor * ((m_ez(i, j + 1, k) - m_ez(i, j, k)) -
                                            (m_ey(i, j, k + 1) - m_ey(i, j, k)));
                m_by(i, j, k) -= bFactor * ((m_ex(i, j, k + 1) - m_ex(i, j, k)) -
                                            (m_ez(i + 1, j, k) - m_ez(i, j, k)));
                m_bz(i, j, k) -= bFactor * ((m_ey(i + 1, j, k) - m_ey(i, j, k)) -
                                            (m_ex(i, j + 1, k) - m_ex(i, j, k)));
            }
        }
    }

    // E full step, using the just-updated B: needs each B field's ghost
    // cells on the -1 side, since (e.g.) cell 0's Ex update reads By[-1],
    // the wrapped-around copy of By[n-1].
    m_bx.applyPeriodicBoundary();
    m_by.applyPeriodicBoundary();
    m_bz.applyPeriodicBoundary();
    const double eFactor = c * c * dt / h;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                m_ex(i, j, k) += eFactor * ((m_bz(i, j, k) - m_bz(i, j - 1, k)) -
                                            (m_by(i, j, k) - m_by(i, j, k - 1)));
                m_ey(i, j, k) += eFactor * ((m_bx(i, j, k) - m_bx(i, j, k - 1)) -
                                            (m_bz(i, j, k) - m_bz(i - 1, j, k)));
                m_ez(i, j, k) += eFactor * ((m_by(i, j, k) - m_by(i - 1, j, k)) -
                                            (m_bx(i, j, k) - m_bx(i, j - 1, k)));
            }
        }
    }
}

double MaxwellField3D::totalEnergy() const {
    const double epsilon0 = constants::vacuumPermittivity.value();
    const double mu0 = constants::vacuumPermeability.value();
    const auto nx = static_cast<std::ptrdiff_t>(m_ex.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_ex.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_ex.cellCountZ());
    const double h = m_ex.spacing();

    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                const double exAt = 0.5 * (m_ex(i, j, k) + m_ex(i - 1, j, k));
                const double eyAt = 0.5 * (m_ey(i, j, k) + m_ey(i, j - 1, k));
                const double ezAt = 0.5 * (m_ez(i, j, k) + m_ez(i, j, k - 1));

                const double bxAt = 0.25 * (m_bx(i, j, k) + m_bx(i, j - 1, k) +
                                            m_bx(i, j, k - 1) + m_bx(i, j - 1, k - 1));
                const double byAt = 0.25 * (m_by(i, j, k) + m_by(i - 1, j, k) +
                                            m_by(i, j, k - 1) + m_by(i - 1, j, k - 1));
                const double bzAt = 0.25 * (m_bz(i, j, k) + m_bz(i - 1, j, k) +
                                            m_bz(i, j - 1, k) + m_bz(i - 1, j - 1, k));

                const double eSquared = exAt * exAt + eyAt * eyAt + ezAt * ezAt;
                const double bSquared = bxAt * bxAt + byAt * byAt + bzAt * bzAt;
                total += 0.5 * (epsilon0 * eSquared + bSquared / mu0);
            }
        }
    }
    return total * h * h * h;
}

}  // namespace ysq
