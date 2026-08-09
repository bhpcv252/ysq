#include <Physics/QuantumMechanics/WavePacket3D.hpp>

#include <Math/FFT.hpp>
#include <Math/Scalar.hpp>

#include <utility>

namespace ysq {

TimeDependentWavefunction3D::TimeDependentWavefunction3D(
    std::vector<Complex<double>> initialWavefunction, std::vector<double> potential,
    std::size_t nx, std::size_t ny, std::size_t nz, double spacing, double mass,
    double hbar)
    : m_psi(std::move(initialWavefunction)),
      m_potential(std::move(potential)),
      m_nx(nx),
      m_ny(ny),
      m_nz(nz),
      m_spacing(spacing),
      m_mass(mass),
      m_hbar(hbar) {}

std::size_t TimeDependentWavefunction3D::idx(std::size_t i, std::size_t j,
                                             std::size_t k) const noexcept {
    return (i * m_ny + j) * m_nz + k;
}

std::size_t TimeDependentWavefunction3D::size() const noexcept {
    return m_psi.size();
}

Complex<double> TimeDependentWavefunction3D::amplitude(std::size_t i, std::size_t j,
                                                       std::size_t k) const {
    return m_psi[idx(i, j, k)];
}

double TimeDependentWavefunction3D::probabilityDensity(std::size_t i, std::size_t j,
                                                       std::size_t k) const {
    const Complex<double>& value = m_psi[idx(i, j, k)];
    return value.re * value.re + value.im * value.im;
}

namespace {

/// The spatial-frequency mapping `Optics/Diffraction.hpp` and
/// `WavePacket.hpp` both already use: `k = 2 pi p / (N spacing)`, negative
/// for `p > N/2` (the upper half of the FFT's output represents negative
/// frequencies).
double wavenumberFor(std::size_t p, std::size_t n, double spacing) {
    const auto signedP =
        (p <= n / 2) ? static_cast<long>(p) : static_cast<long>(p) - static_cast<long>(n);
    return 2.0 * kPi<double> * static_cast<double>(signedP) /
           (static_cast<double>(n) * spacing);
}

}  // namespace

void TimeDependentWavefunction3D::step(double dt) {
    const double halfDt = dt / 2.0;

    for (std::size_t p = 0; p < m_psi.size(); ++p) {
        const double phase = -m_potential[p] * halfDt / m_hbar;
        m_psi[p] *= Complex<double>::polar(1.0, phase);
    }

    fft3D(m_psi, m_nx, m_ny, m_nz);
    for (std::size_t i = 0; i < m_nx; ++i) {
        const double kx = wavenumberFor(i, m_nx, m_spacing);
        for (std::size_t j = 0; j < m_ny; ++j) {
            const double ky = wavenumberFor(j, m_ny, m_spacing);
            for (std::size_t k = 0; k < m_nz; ++k) {
                const double kz = wavenumberFor(k, m_nz, m_spacing);
                const double kSquared = kx * kx + ky * ky + kz * kz;
                const double kineticPhase = -m_hbar * kSquared * dt / (2.0 * m_mass);
                m_psi[idx(i, j, k)] *= Complex<double>::polar(1.0, kineticPhase);
            }
        }
    }
    ifft3D(m_psi, m_nx, m_ny, m_nz);

    for (std::size_t p = 0; p < m_psi.size(); ++p) {
        const double phase = -m_potential[p] * halfDt / m_hbar;
        m_psi[p] *= Complex<double>::polar(1.0, phase);
    }
}

double TimeDependentWavefunction3D::totalProbability() const {
    double total = 0.0;
    for (std::size_t i = 0; i < m_nx; ++i) {
        for (std::size_t j = 0; j < m_ny; ++j) {
            for (std::size_t k = 0; k < m_nz; ++k) {
                total += probabilityDensity(i, j, k);
            }
        }
    }
    return total * m_spacing * m_spacing * m_spacing;
}

double TimeDependentWavefunction3D::expectationEnergy() const {
    const double kineticCoefficient =
        (m_hbar * m_hbar) / (2.0 * m_mass * m_spacing * m_spacing);
    const Complex<double> zero{0.0, 0.0};

    double kinetic = 0.0;
    double potential = 0.0;
    for (std::size_t i = 0; i < m_nx; ++i) {
        for (std::size_t j = 0; j < m_ny; ++j) {
            for (std::size_t k = 0; k < m_nz; ++k) {
                const std::size_t here = idx(i, j, k);
                Complex<double> neighborSum = zero;
                neighborSum += (i > 0) ? m_psi[idx(i - 1, j, k)] : zero;
                neighborSum += (i + 1 < m_nx) ? m_psi[idx(i + 1, j, k)] : zero;
                neighborSum += (j > 0) ? m_psi[idx(i, j - 1, k)] : zero;
                neighborSum += (j + 1 < m_ny) ? m_psi[idx(i, j + 1, k)] : zero;
                neighborSum += (k > 0) ? m_psi[idx(i, j, k - 1)] : zero;
                neighborSum += (k + 1 < m_nz) ? m_psi[idx(i, j, k + 1)] : zero;

                const Complex<double> laplacian = neighborSum - m_psi[here] * 6.0;
                const Complex<double> kineticTerm =
                    conj(m_psi[here]) * (-kineticCoefficient) * laplacian;

                kinetic += kineticTerm.re;
                potential += probabilityDensity(i, j, k) * m_potential[here];
            }
        }
    }

    return (kinetic + potential) * m_spacing * m_spacing * m_spacing;
}

}  // namespace ysq
