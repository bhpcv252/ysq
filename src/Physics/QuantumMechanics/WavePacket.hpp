#pragma once

#include <Math/Complex.hpp>

#include <cstddef>
#include <vector>

namespace ysq {

/// Time-dependent evolution under the Schrödinger equation,
/// `i hbar d(psi)/dt = [-hbar^2/(2m) d^2/dx^2 + V(x)] psi`, by the
/// second-order Strang-split split-step Fourier method (Feit, Fleck &
/// Steiger 1982): half a potential-phase step, a full kinetic-phase step
/// done in momentum space via `Math/FFT.hpp`, then the other half
/// potential-phase step.
///
/// **This is the reason `Math/FFT.hpp` exists in this engine at all**,
/// ahead of whichever other consumer might have asked for it first: the
/// kinetic operator is diagonal in momentum space (a pure per-mode phase),
/// where the same operator in position space would need a dense,
/// expensive convolution instead, so transforming there and back every
/// step is what makes this method both simple and fast. `psi.size()` must
/// be a power of two, `Math/FFT.hpp`'s own precondition.
///
/// `hbar` and `mass` are explicit, never defaulted, the same standing
/// `Schrodinger.hpp`'s solver has: real SI quantum mechanics and
/// natural/atomic units are equally valid choices this makes no
/// assumption between.
class TimeDependentWavefunction1D {
public:
    TimeDependentWavefunction1D(std::vector<Complex<double>> initialWavefunction,
                                std::vector<double> potential, double spacing,
                                double mass, double hbar);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] Complex<double> amplitude(std::size_t i) const;
    [[nodiscard]] double probabilityDensity(std::size_t i) const;

    /// One split-step Fourier evolution step of size `dt`.
    void step(double dt);

    /// `sum |psi_i|^2 spacing`: should stay 1 (to numerical precision) for
    /// as long as `step` is called, since every operator this method
    /// applies is a pure phase -- unitarity is structural here, not merely
    /// approximate.
    [[nodiscard]] double totalProbability() const;

    /// `<x> = sum x_i |psi_i|^2 spacing`.
    [[nodiscard]] double expectationPosition() const;

    /// `<H> = <T> + <V>`, computed by the same 3-point finite-difference
    /// discretization `Schrodinger.hpp`'s eigenvalue problem uses to build
    /// its own Hamiltonian -- not via the FFT this class's own `step`
    /// uses internally -- so this is an independent check on the
    /// propagator rather than a tautology: a time-independent potential
    /// must leave this constant over many steps regardless of which
    /// method actually advanced time.
    [[nodiscard]] double expectationEnergy() const;

private:
    std::vector<Complex<double>> m_psi;
    std::vector<double> m_potential;
    double m_spacing;
    double m_mass;
    double m_hbar;
};

}  // namespace ysq
