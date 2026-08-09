#pragma once

#include <Math/Complex.hpp>

#include <cstddef>
#include <vector>

namespace ysq {

/// Time-dependent evolution under the 3D Schrödinger equation, the direct
/// generalization of `WavePacket.hpp`'s split-step Fourier method: half a
/// potential-phase step, a full kinetic-phase step done in momentum space
/// via `Math/FFT.hpp`'s `fft3D`/`ifft3D`, then the other half
/// potential-phase step. The 3D kinetic operator is diagonal in momentum
/// space exactly like the 1D one is, mode by mode, since
/// `kx^2 + ky^2 + kz^2` separates additively — the same reason this method
/// works in any dimension once the FFT it leans on does.
///
/// `hbar` and `mass` are explicit, the same standing `WavePacket.hpp` has.
/// Not built on `Grid3D`: `Complex` does not satisfy `Numeric`
/// (`Math/Complex.hpp`'s own doc comment), so storage is a flat
/// `std::vector`, matching `TimeDependentWavefunction1D`'s own choice for
/// the identical reason.
class TimeDependentWavefunction3D {
public:
    /// `initialWavefunction` and `potential` are flat, row-major, size
    /// `nx * ny * nz`, `idx(i, j, k) = (i * ny + j) * nz + k` (matching
    /// `Schrodinger3D.hpp`'s convention). `nx`, `ny`, `nz` must each
    /// independently be a power of two, `Math/FFT.hpp`'s own precondition.
    TimeDependentWavefunction3D(std::vector<Complex<double>> initialWavefunction,
                                std::vector<double> potential, std::size_t nx,
                                std::size_t ny, std::size_t nz, double spacing,
                                double mass, double hbar);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] Complex<double> amplitude(std::size_t i, std::size_t j,
                                            std::size_t k) const;
    [[nodiscard]] double probabilityDensity(std::size_t i, std::size_t j,
                                            std::size_t k) const;

    /// One split-step Fourier evolution step of size `dt`.
    void step(double dt);

    /// `sum |psi|^2 spacing^3`: should stay 1 (to numerical precision),
    /// unitarity being structural here, the same as
    /// `TimeDependentWavefunction1D::totalProbability`.
    [[nodiscard]] double totalProbability() const;

    /// `<H> = <T> + <V>`, via the same 7-point finite-difference
    /// discretization `Schrodinger3D.hpp`'s eigenvalue problem uses (not
    /// the FFT `step` uses internally) — an independent check on the
    /// propagator, the same reason `TimeDependentWavefunction1D`'s own
    /// version exists.
    [[nodiscard]] double expectationEnergy() const;

private:
    std::vector<Complex<double>> m_psi;
    std::vector<double> m_potential;
    std::size_t m_nx;
    std::size_t m_ny;
    std::size_t m_nz;
    double m_spacing;
    double m_mass;
    double m_hbar;

    [[nodiscard]] std::size_t idx(std::size_t i, std::size_t j,
                                  std::size_t k) const noexcept;
};

}  // namespace ysq
