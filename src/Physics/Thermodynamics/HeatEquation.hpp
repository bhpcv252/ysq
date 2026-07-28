#pragma once

#include <Math/Grid.hpp>

#include <cstddef>

namespace ysq {

/// Thermodynamics' second rung: the heat (diffusion) equation in one
/// spatial dimension,
///
///     dT/dt = alpha d^2T/dx^2
///
/// `alpha` the thermal diffusivity (k / (rho c_p)), solved by the explicit
/// forward-time, centred-space (FTCS) finite-difference scheme:
///
///     T_i^(n+1) = T_i^n + alpha dt/dx^2 (T_(i+1)^n - 2 T_i^n + T_(i-1)^n)
///
/// **Scope.** One spatial dimension, per Math/Grid.hpp; a 3D solver, needed
/// for a genuinely shaped heat source or boundary, is future work. FTCS is
/// explicit and conditionally stable rather than an implicit scheme (Crank-
/// Nicolson, say), which trades an unconditionally stable, more expensive
/// implicit solve for this scheme's simplicity and its very direct
/// validation against the exact spreading Gaussian solution; see
/// docs/physics.md.
///
/// Periodic boundaries, the same choice Maxwell and Eulerian make: total
/// heat, `sum T dx`, is exactly conserved, and a domain large enough
/// relative to the run keeps a spreading pulse from wrapping around and
/// contaminating the result.
class HeatEquation1D {
public:
    HeatEquation1D(std::size_t cellCount, double spacing, double diffusivity);

    [[nodiscard]] std::size_t cellCount() const noexcept;
    [[nodiscard]] double spacing() const noexcept;
    [[nodiscard]] double diffusivity() const noexcept;

    void setTemperature(std::size_t cell, double value);
    [[nodiscard]] double temperature(std::size_t cell) const;

    /// One explicit FTCS step. `dt` must satisfy the stability condition
    /// `diffusivity * dt / spacing()^2 <= 0.5`; stableTimeStep() gives a
    /// safe value at a given safety factor (0 < factor <= 1).
    void step(double dt);
    [[nodiscard]] double stableTimeStep(double safetyFactor) const;

    [[nodiscard]] double totalHeat() const;

private:
    double m_diffusivity;
    Grid1D<double> m_temperature;
};

}  // namespace ysq
