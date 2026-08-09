#pragma once

#include <Math/Grid3D.hpp>

#include <cstddef>

namespace ysq {

/// The compressible Euler equations, mass, momentum (now a 3-vector) and
/// energy, for an ideal gas of adiabatic index gamma, in three spatial
/// dimensions, by **dimensional (Godunov) splitting**: a full x-sweep,
/// then a full y-sweep, then a full z-sweep, each `dt`, each being
/// `EulerianFluid1D`'s own first-order finite-volume Rusanov update
/// generalized to carry the two transverse momentum components through
/// every flux passively (`flux = rho u v`, `rho u w`, no pressure term,
/// since a 1D-normal Riemann problem along one axis does not see the
/// other two) while the Rusanov dissipation itself
/// (`0.5 maxSpeed (right - left)`) still applies uniformly across all five
/// conserved quantities. Pressure always comes from *total* kinetic
/// energy (`u^2+v^2+w^2`), not just the sweep-normal component. Splitting
/// does not add or remove accuracy order here: both the split-off pieces
/// and the original scheme are already first order, the same "robust,
/// simple to verify, at the cost of smearing a shock" character
/// `EulerianFluid1D`'s own doc comment already commits to.
///
/// Periodic boundaries throughout, the same as `EulerianFluid1D`: exact
/// conservation of mass, momentum and energy follows directly.
class EulerianFluid3D {
public:
    EulerianFluid3D(std::size_t cellCountX, std::size_t cellCountY,
                    std::size_t cellCountZ, double spacing, double adiabaticIndex);

    [[nodiscard]] std::size_t cellCountX() const noexcept;
    [[nodiscard]] std::size_t cellCountY() const noexcept;
    [[nodiscard]] std::size_t cellCountZ() const noexcept;
    [[nodiscard]] double spacing() const noexcept;
    [[nodiscard]] double adiabaticIndex() const noexcept;

    void setState(std::size_t i, std::size_t j, std::size_t k, double density,
                  double velocityX, double velocityY, double velocityZ, double pressure);

    [[nodiscard]] double density(std::size_t i, std::size_t j, std::size_t k) const;
    [[nodiscard]] double velocityX(std::size_t i, std::size_t j, std::size_t k) const;
    [[nodiscard]] double velocityY(std::size_t i, std::size_t j, std::size_t k) const;
    [[nodiscard]] double velocityZ(std::size_t i, std::size_t j, std::size_t k) const;
    [[nodiscard]] double pressure(std::size_t i, std::size_t j, std::size_t k) const;

    /// One explicit x-then-y-then-z finite-volume step. `dt` must satisfy
    /// the CFL condition; `stableTimeStep()` gives a safe value at a given
    /// Courant number (0 < courantNumber <= 1).
    void step(double dt);
    [[nodiscard]] double stableTimeStep(double courantNumber) const;

    [[nodiscard]] double totalMass() const;
    [[nodiscard]] double totalMomentumX() const;
    [[nodiscard]] double totalMomentumY() const;
    [[nodiscard]] double totalMomentumZ() const;
    [[nodiscard]] double totalEnergy() const;

private:
    double m_gamma;
    Grid3D<double> m_density;
    Grid3D<double> m_momentumX;
    Grid3D<double> m_momentumY;
    Grid3D<double> m_momentumZ;
    Grid3D<double> m_energy;
};

}  // namespace ysq
