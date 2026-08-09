#pragma once

#include <Math/Grid3D.hpp>

#include <cstddef>

namespace ysq {

/// Linear acoustics in three spatial dimensions, homogeneous medium of
/// density `rho0` and sound speed `c`, pressure perturbation `p` and
/// velocity vector `u = (ux, uy, uz)`:
///
///     dp/dt = -rho0 c^2 (dux/dx + duy/dy + duz/dz)
///     dux/dt = -(1/rho0) dp/dx    duy/dt = -(1/rho0) dp/dy    duz/dt = -(1/rho0) dp/dz
///
/// The direct 3D generalization of `AcousticField1D`: a staggered
/// (marker-and-cell) grid, `p` at cell centers and each velocity component
/// at its own face centers (`ux` at `x_i + spacing/2`, `uy` at
/// `y_j + spacing/2`, `uz` at `z_k + spacing/2`), updated by the same
/// leapfrog structure (a velocity half-step from pressure's gradient, then
/// a pressure full-step from the just-updated velocity's divergence) with
/// the same 2-point differences per axis `AcousticField1D` already uses —
/// no cross-axis coupling, unlike `MaxwellField3D`'s curl, since divergence
/// and gradient are separable per component.
///
/// **No exact "magic timestep" in 3D.** `AcousticField1D`'s 1D leapfrog
/// scheme has a special `dt` with zero numerical dispersion, a coincidence
/// of the 1D discretization exactly matching the continuum wave equation
/// at that step; no such step exists on a 3D Cartesian grid, where
/// dispersion becomes direction-dependent. `stableTimeStep()` gives the
/// CFL limit only.
class AcousticField3D {
public:
    AcousticField3D(std::size_t cellCountX, std::size_t cellCountY,
                    std::size_t cellCountZ, double spacing, double mediumDensity,
                    double soundSpeed);

    [[nodiscard]] std::size_t cellCountX() const noexcept;
    [[nodiscard]] std::size_t cellCountY() const noexcept;
    [[nodiscard]] std::size_t cellCountZ() const noexcept;
    [[nodiscard]] double spacing() const noexcept;
    [[nodiscard]] double mediumDensity() const noexcept;
    [[nodiscard]] double soundSpeed() const noexcept;

    [[nodiscard]] double pressure(std::size_t i, std::size_t j, std::size_t k) const;
    void setPressure(std::size_t i, std::size_t j, std::size_t k, double value);

    [[nodiscard]] double velocityX(std::size_t i, std::size_t j, std::size_t k) const;
    void setVelocityX(std::size_t i, std::size_t j, std::size_t k, double value);
    [[nodiscard]] double velocityY(std::size_t i, std::size_t j, std::size_t k) const;
    void setVelocityY(std::size_t i, std::size_t j, std::size_t k, double value);
    [[nodiscard]] double velocityZ(std::size_t i, std::size_t j, std::size_t k) const;
    void setVelocityZ(std::size_t i, std::size_t j, std::size_t k, double value);

    /// One leapfrog cycle. `dt` must satisfy the 3D CFL condition
    /// `dt <= spacing() / (soundSpeed() sqrt(3))`; `stableTimeStep()` gives
    /// a safe value at a given Courant factor (0 < courantFactor <= 1).
    void step(double dt);
    [[nodiscard]] double stableTimeStep(double courantFactor) const;

    /// `(1/2) sum (p^2/(rho0 c^2) + rho0 |u|^2) h^3`: the 3D analog of
    /// `AcousticField1D::totalEnergy`, each velocity component averaged
    /// from its two staggered neighbors to approximate its value at `p`'s
    /// grid point, the same staggering correction the 1D version makes.
    [[nodiscard]] double totalEnergy() const;

private:
    Grid3D<double> m_pressure;
    Grid3D<double> m_velocityX;
    Grid3D<double> m_velocityY;
    Grid3D<double> m_velocityZ;
    double m_density;
    double m_soundSpeed;
};

}  // namespace ysq
