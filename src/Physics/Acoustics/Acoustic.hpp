#pragma once

#include <Math/Grid.hpp>

#include <cstddef>

namespace ysq {

/// A one-dimensional finite-difference time-domain (FDTD) solver for the
/// linearized acoustic wave equations, in a homogeneous medium of density
/// `rho0` and sound speed `c`, pressure perturbation `p` and particle
/// velocity `u`:
///
///     dp/dt = -rho0 c^2 du/dx
///     du/dt = -(1/rho0) dp/dx
///
/// **Structurally identical to `Electromagnetism/Maxwell.hpp`.** These
/// equations are `Maxwell.hpp`'s own 1D vacuum equations,
/// `dEy/dt = -c^2 dBz/dx`, `dBz/dt = -dEy/dx`, with `p <-> Ey`, `u <-> Bz`,
/// `rho0 c^2 <-> c^2`, `1/rho0 <-> 1`: the same linear wave operator, one
/// more physical interpretation of it. Every property that makes
/// `MaxwellField1D` correct -- the Yee/leapfrog staggering, the CFL
/// stability limit, the exact zero-numerical-dispersion "magic" time step,
/// the symplectic-style energy conservation -- is a property of that
/// discretized operator itself, not of which fields are involved, so it
/// carries over unchanged. `mediumDensity`/`soundSpeed` are constructor
/// parameters rather than a fixed constant the way vacuum `c` is for
/// light: sound speed and medium density vary by material, so there is no
/// single physical constant to reach for here.
///
/// `p` lives at the grid's integer points, `u` at the half-integer points
/// in between (the same Yee staggering `Maxwell.hpp` uses):
/// `velocity(i)` is `u` at `x_i + spacing/2`, not at `x_i`.
class AcousticField1D {
public:
    AcousticField1D(std::size_t cellCount, double spacing, double mediumDensity,
                    double soundSpeed);

    [[nodiscard]] std::size_t cellCount() const noexcept;
    [[nodiscard]] double spacing() const noexcept;
    [[nodiscard]] double mediumDensity() const noexcept;
    [[nodiscard]] double soundSpeed() const noexcept;

    [[nodiscard]] double pressure(std::size_t cell) const;
    void setPressure(std::size_t cell, double value);

    [[nodiscard]] double velocity(std::size_t cell) const;
    void setVelocity(std::size_t cell, double value);

    /// One leapfrog cycle. `dt` must satisfy the CFL condition
    /// `dt <= spacing() / soundSpeed()`; `magicAcousticTimeStep()` gives the
    /// value that makes this scheme exact, with no numerical dispersion at
    /// all.
    void step(double dt);

    /// `(1/2) sum (p^2 / (rho0 c^2) + rho0 u^2) spacing`: kinetic
    /// (`(1/2) rho0 u^2`) plus compressional potential
    /// (`(1/2) p^2 / (rho0 c^2)`) energy density, the acoustic analog of
    /// `MaxwellField1D::totalEnergy`'s electric-plus-magnetic energy
    /// density. `u` is averaged from its two neighbours to approximate its
    /// value at `p`'s grid points, the same staggering correction
    /// `totalEnergy()` there makes for `Bz`.
    [[nodiscard]] double totalEnergy() const;

private:
    Grid1D<double> m_pressure;
    Grid1D<double> m_velocity;
    double m_density;
    double m_soundSpeed;
};

/// `spacing / soundSpeed`: the step size at which this scheme has no
/// numerical dispersion, so a wave packet propagates with its shape exactly
/// preserved rather than merely approximately -- the acoustic analog of
/// `Maxwell.hpp`'s `magicTimeStep`.
[[nodiscard]] double magicAcousticTimeStep(double spacing, double soundSpeed);

}  // namespace ysq
