#include <Physics/QuantumMechanics/WavePacket.hpp>

#include <Math/FFT.hpp>
#include <Math/Scalar.hpp>

#include <utility>

namespace ysq {

TimeDependentWavefunction1D::TimeDependentWavefunction1D(
    std::vector<Complex<double>> initialWavefunction, std::vector<double> potential,
    double spacing, double mass, double hbar)
    : m_psi(std::move(initialWavefunction)),
      m_potential(std::move(potential)),
      m_spacing(spacing),
      m_mass(mass),
      m_hbar(hbar) {}

std::size_t TimeDependentWavefunction1D::size() const noexcept {
    return m_psi.size();
}

Complex<double> TimeDependentWavefunction1D::amplitude(std::size_t i) const {
    return m_psi[i];
}

double TimeDependentWavefunction1D::probabilityDensity(std::size_t i) const {
    return m_psi[i].re * m_psi[i].re + m_psi[i].im * m_psi[i].im;
}

void TimeDependentWavefunction1D::step(double dt) {
    const std::size_t n = m_psi.size();
    const double halfDt = dt / 2.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double phase = -m_potential[i] * halfDt / m_hbar;
        m_psi[i] *= Complex<double>::polar(1.0, phase);
    }

    fft(m_psi);
    for (std::size_t j = 0; j < n; ++j) {
        const auto signedJ = (j <= n / 2) ? static_cast<long>(j)
                                          : static_cast<long>(j) - static_cast<long>(n);
        const double k = 2.0 * kPi<double> * static_cast<double>(signedJ) /
                         (static_cast<double>(n) * m_spacing);
        const double kineticPhase = -m_hbar * k * k * dt / (2.0 * m_mass);
        m_psi[j] *= Complex<double>::polar(1.0, kineticPhase);
    }
    ifft(m_psi);

    for (std::size_t i = 0; i < n; ++i) {
        const double phase = -m_potential[i] * halfDt / m_hbar;
        m_psi[i] *= Complex<double>::polar(1.0, phase);
    }
}

double TimeDependentWavefunction1D::totalProbability() const {
    double total = 0.0;
    for (std::size_t i = 0; i < m_psi.size(); ++i) {
        total += probabilityDensity(i);
    }
    return total * m_spacing;
}

double TimeDependentWavefunction1D::expectationPosition() const {
    double total = 0.0;
    for (std::size_t i = 0; i < m_psi.size(); ++i) {
        const double x = static_cast<double>(i) * m_spacing;
        total += x * probabilityDensity(i);
    }
    return total * m_spacing;
}

double TimeDependentWavefunction1D::expectationEnergy() const {
    const std::size_t n = m_psi.size();
    const double kineticCoefficient =
        (m_hbar * m_hbar) / (2.0 * m_mass * m_spacing * m_spacing);

    double kinetic = 0.0;
    double potential = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const Complex<double> left = (i == 0) ? Complex<double>{0.0, 0.0} : m_psi[i - 1];
        const Complex<double> right =
            (i + 1 == n) ? Complex<double>{0.0, 0.0} : m_psi[i + 1];
        const Complex<double> laplacian = left - m_psi[i] * 2.0 + right;
        const Complex<double> kineticTerm =
            conj(m_psi[i]) * (-kineticCoefficient) * laplacian;

        kinetic += kineticTerm.re;
        potential += probabilityDensity(i) * m_potential[i];
    }

    return (kinetic + potential) * m_spacing;
}

}  // namespace ysq
