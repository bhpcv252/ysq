#pragma once

#include <Math/Grid3D.hpp>

#include <cstddef>

namespace ysq {

/// The full three-dimensional finite-difference time-domain (FDTD) solver
/// for the vacuum Maxwell equations, on a standard Yee grid:
///
///     dE/dt = c^2 curl(B)
///     dB/dt = -curl(E)
///
/// The direct 3D generalization of `MaxwellField1D`'s own `Ey`/`Bz`
/// leapfrog, with two curl terms per update instead of one and six field
/// components instead of two. Each component keeps its own Yee staggering
/// (`Ex` at `(i+1/2, j, k)`, `Ey` at `(i, j+1/2, k)`, `Ez` at
/// `(i, j, k+1/2)`; `Bx` at `(i, j+1/2, k+1/2)`, `By` at `(i+1/2, j, k+1/2)`,
/// `Bz` at `(i+1/2, j+1/2, k)`), chosen so every curl term a component
/// needs is a plain adjacent-index difference of another component already
/// staggered to exactly the right position — the same "no numerical idea
/// beyond a 2-point difference" property `MaxwellField1D` has, just with
/// more bookkeeping. `B` updates read a forward difference of `E`
/// (matching `MaxwellField1D`'s `Bz -= dt/dx (Ey[i+1] - Ey[i])`); `E`
/// updates then read a backward difference of the just-updated `B`
/// (matching `Ey -= c^2 dt/dx (Bz[i] - Bz[i-1])`).
///
/// **No exact "magic timestep" in 3D** — the same point
/// `Physics/Acoustics/Acoustic3D.hpp` makes for the identical reason
/// (`stableTimeStep()` gives the CFL limit only).
class MaxwellField3D {
public:
    MaxwellField3D(std::size_t cellCountX, std::size_t cellCountY, std::size_t cellCountZ,
                   double spacing);

    [[nodiscard]] std::size_t cellCountX() const noexcept;
    [[nodiscard]] std::size_t cellCountY() const noexcept;
    [[nodiscard]] std::size_t cellCountZ() const noexcept;
    [[nodiscard]] double spacing() const noexcept;

    [[nodiscard]] double electricFieldX(std::size_t i, std::size_t j,
                                        std::size_t k) const;
    void setElectricFieldX(std::size_t i, std::size_t j, std::size_t k, double value);
    [[nodiscard]] double electricFieldY(std::size_t i, std::size_t j,
                                        std::size_t k) const;
    void setElectricFieldY(std::size_t i, std::size_t j, std::size_t k, double value);
    [[nodiscard]] double electricFieldZ(std::size_t i, std::size_t j,
                                        std::size_t k) const;
    void setElectricFieldZ(std::size_t i, std::size_t j, std::size_t k, double value);

    [[nodiscard]] double magneticFieldX(std::size_t i, std::size_t j,
                                        std::size_t k) const;
    void setMagneticFieldX(std::size_t i, std::size_t j, std::size_t k, double value);
    [[nodiscard]] double magneticFieldY(std::size_t i, std::size_t j,
                                        std::size_t k) const;
    void setMagneticFieldY(std::size_t i, std::size_t j, std::size_t k, double value);
    [[nodiscard]] double magneticFieldZ(std::size_t i, std::size_t j,
                                        std::size_t k) const;
    void setMagneticFieldZ(std::size_t i, std::size_t j, std::size_t k, double value);

    /// One leapfrog cycle. `dt` must satisfy the 3D CFL condition
    /// `dt <= spacing() / (c sqrt(3))`; `stableTimeStep()` gives a safe
    /// value at a given Courant factor (0 < courantFactor <= 1).
    void step(double dt);
    [[nodiscard]] double stableTimeStep(double courantFactor) const;

    /// `(1/2) sum (epsilon0 |E|^2 + |B|^2/mu0) h^3`: each field component
    /// bilinearly interpolated from its own staggered positions onto the
    /// grid's integer vertices (`E`'s two neighbors along its own stagger
    /// axis; `B`'s four neighbors, since each `B` component is staggered
    /// along two axes at once) before combining, the 3D generalization of
    /// `MaxwellField1D::totalEnergy`'s single-neighbor average.
    [[nodiscard]] double totalEnergy() const;

private:
    Grid3D<double> m_ex, m_ey, m_ez;
    Grid3D<double> m_bx, m_by, m_bz;
};

}  // namespace ysq
