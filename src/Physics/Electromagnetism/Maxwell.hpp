#pragma once

#include <Math/Grid.hpp>

#include <cstddef>

namespace ysq {

/// A one-dimensional finite-difference time-domain (FDTD) solver for the
/// vacuum Maxwell equations, restricted to a transverse electromagnetic
/// wave propagating along x with components Ey and Bz:
///
///     dEy/dt = -c^2 dBz/dx
///     dBz/dt = -dEy/dx
///
/// **The second rung of the Electromagnetism ladder.** Field.hpp's rung is
/// quasi-static: a source's field is what its present position and velocity
/// say, instantaneously. This rung is the opposite regime, a field that
/// actually propagates at c because it is evolved from Maxwell's equations
/// rather than assumed. Neither replaces the other; an application picks
/// whichever matches its regime, the same way it picks a rung of the
/// gravity ladder.
///
/// **Scope: one spatial dimension**, per Math/Grid.hpp. A full 3D Yee-grid
/// solver, which is what a genuine radiating source (a dipole, say) needs,
/// is future work; this rung validates against what 1D vacuum
/// electrodynamics actually predicts: a wave travelling at exactly c, and a
/// closed system's energy staying constant. See docs/physics.md.
///
/// Ey lives at the grid's integer points, Bz at the half-integer points in
/// between (the Yee staggering): magneticField(i) is Bz at x_i +
/// spacing/2, not at x_i. Time is staggered the same way, via leapfrog:
/// step() advances Bz by half a step and then Ey by a full step using the
/// updated Bz, which is what makes the scheme second-order accurate and
/// exactly energy-conserving in the sense a symplectic integrator is (see
/// Math/Integrators/Symplectic.hpp for the general idea; this is that
/// structure applied to a field instead of a particle).
class MaxwellField1D {
public:
    MaxwellField1D(std::size_t cellCount, double spacing);

    [[nodiscard]] std::size_t cellCount() const noexcept;
    [[nodiscard]] double spacing() const noexcept;

    [[nodiscard]] double electricField(std::size_t cell) const;
    void setElectricField(std::size_t cell, double value);

    [[nodiscard]] double magneticField(std::size_t cell) const;
    void setMagneticField(std::size_t cell, double value);

    /// One leapfrog cycle. `dt` must satisfy the CFL condition
    /// `dt <= spacing() / c`; magicTimeStep() gives the value that makes
    /// this particular scheme exact in 1D vacuum, with no numerical
    /// dispersion at all.
    void step(double dt);

    /// (1/2) sum (epsilon0 Ey^2 + Bz^2 / mu0) spacing: the field's total
    /// energy, for a conservation check. Bz is averaged from its two
    /// neighbours to approximate its value at Ey's grid points, since the
    /// two are staggered by half a cell.
    [[nodiscard]] double totalEnergy() const;

private:
    Grid1D<double> m_electric;
    Grid1D<double> m_magnetic;
};

/// spacing / c: the step size at which this scheme has no numerical
/// dispersion in 1D vacuum, so a wave packet propagates with its shape
/// exactly preserved rather than merely approximately.
[[nodiscard]] double magicTimeStep(double spacing);

}  // namespace ysq
