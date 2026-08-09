#pragma once

#include <Math/Grid3D.hpp>

#include <cstddef>

namespace ysq {

/// The heat (diffusion) equation in three spatial dimensions,
///
///     dT/dt = alpha (d^2T/dx^2 + d^2T/dy^2 + d^2T/dz^2)
///
/// the direct generalization of `HeatEquation1D`'s explicit FTCS scheme:
/// the same 3-point stencil applied along each axis and summed, the
/// standard 7-point 3D Laplacian, still second order (this is a scope
/// extension, not also an accuracy upgrade — `Math/FiniteDifference.hpp`'s
/// fourth-order stencils are a different, unrelated tool built for
/// `Physics/Spacetime`'s BSSN evolution, not this).
///
///     T_ijk^(n+1) = T_ijk^n + alpha dt/h^2 (T_(i+1)jk + T_(i-1)jk +
///                    T_i(j+1)k + T_i(j-1)k + T_ij(k+1) + T_ij(k-1) - 6 T_ijk)
///
/// Periodic boundaries, the same choice `HeatEquation1D` makes: total heat
/// is exactly conserved, and a domain large enough relative to the run
/// keeps a spreading pulse from wrapping around and contaminating the
/// result.
class HeatEquation3D {
public:
    HeatEquation3D(std::size_t cellCountX, std::size_t cellCountY, std::size_t cellCountZ,
                   double spacing, double diffusivity);

    [[nodiscard]] std::size_t cellCountX() const noexcept;
    [[nodiscard]] std::size_t cellCountY() const noexcept;
    [[nodiscard]] std::size_t cellCountZ() const noexcept;
    [[nodiscard]] double spacing() const noexcept;
    [[nodiscard]] double diffusivity() const noexcept;

    void setTemperature(std::size_t i, std::size_t j, std::size_t k, double value);
    [[nodiscard]] double temperature(std::size_t i, std::size_t j, std::size_t k) const;

    /// One explicit FTCS step. `dt` must satisfy the 3D stability condition
    /// `diffusivity * dt / spacing()^2 <= 1/6` (three times tighter than
    /// `HeatEquation1D`'s `<= 1/2`, since the 3D Laplacian picks up twice
    /// as many neighbor terms per axis, three axes instead of one);
    /// stableTimeStep() gives a safe value at a given safety factor
    /// (0 < factor <= 1).
    void step(double dt);
    [[nodiscard]] double stableTimeStep(double safetyFactor) const;

    [[nodiscard]] double totalHeat() const;

private:
    double m_diffusivity;
    Grid3D<double> m_temperature;
};

}  // namespace ysq
