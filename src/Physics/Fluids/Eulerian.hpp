#pragma once

#include <Math/Grid.hpp>

#include <cstddef>

namespace ysq {

/// The compressible Euler equations, mass, momentum and energy, for an
/// ideal gas of adiabatic index gamma, in one spatial dimension:
///
///     d(rho)/dt   + d(rho u)/dx         = 0
///     d(rho u)/dt + d(rho u^2 + p)/dx   = 0
///     d(E)/dt     + d(u (E + p))/dx     = 0
///
/// solved by a first-order finite-volume method with the Rusanov (local
/// Lax-Friedrichs) numerical flux.
///
/// **Fluid rung 2.** Where SPH (Fluids/SPH.hpp) is Lagrangian, particles
/// carrying the fluid with them, this is Eulerian, the fluid moving through
/// a fixed Math/Grid.hpp mesh: the standard shape for compressible gas
/// dynamics and shocks, which SPH resolves poorly without artificial
/// viscosity, not implemented on that rung.
///
/// **Scope.** One spatial dimension, per Math/Grid.hpp. First-order in
/// space and time: robust and simple to verify exactly, at the cost of
/// smearing a shock or contact discontinuity over several cells rather
/// than resolving it sharply, the standard tradeoff a first-order
/// Godunov-type scheme makes. A higher-order reconstruction and a sharper
/// Riemann solver (HLLC, or an exact one) are natural refinements, not
/// implemented here; see docs/physics.md for what is validated instead of
/// an exact Riemann solution.
///
/// Periodic boundaries throughout, the same as MaxwellField1D: exact
/// conservation of mass, momentum and energy follows directly, and a
/// domain large enough relative to the run length keeps a wave from
/// wrapping around and contaminating the result.
class EulerianFluid1D {
public:
    EulerianFluid1D(std::size_t cellCount, double spacing, double adiabaticIndex);

    [[nodiscard]] std::size_t cellCount() const noexcept;
    [[nodiscard]] double spacing() const noexcept;
    [[nodiscard]] double adiabaticIndex() const noexcept;

    void setState(std::size_t cell, double density, double velocity, double pressure);

    [[nodiscard]] double density(std::size_t cell) const;
    [[nodiscard]] double velocity(std::size_t cell) const;
    [[nodiscard]] double pressure(std::size_t cell) const;

    /// One explicit finite-volume step. `dt` must satisfy the CFL
    /// condition; stableTimeStep() gives a safe value at a given Courant
    /// number (0 < courantNumber <= 1).
    void step(double dt);

    [[nodiscard]] double stableTimeStep(double courantNumber) const;

    [[nodiscard]] double totalMass() const;
    [[nodiscard]] double totalMomentum() const;
    [[nodiscard]] double totalEnergy() const;

private:
    double m_gamma;
    Grid1D<double> m_density;
    Grid1D<double> m_momentum;
    Grid1D<double> m_energy;
};

}  // namespace ysq
