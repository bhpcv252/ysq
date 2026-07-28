#pragma once

#include <Math/Vector3.hpp>

#include <span>
#include <vector>

namespace ysq {

/// One SPH particle: mass, kinematic state, and the two fields the kernel
/// sum produces. density and pressure are outputs of
/// computeDensityAndPressure, not independent state; they are stored on the
/// particle because pressureAccelerations needs both of every neighbour,
/// and recomputing them per pair would cost the kernel sum twice over.
struct SPHParticle {
    double mass = 0.0;
    Vec3 position{};
    Vec3 velocity{};
    double density = 0.0;
    double pressure = 0.0;
};

/// Smoothed Particle Hydrodynamics: fluid rung 1, particle-based, no grid.
/// The Lagrangian counterpart to Physics/Gravity's direct summation and
/// Barnes-Hut, and built to reuse the same shape: a span of particles in,
/// accelerations out, ready for one of Math's integrators.
///
/// The cubic spline kernel (Monaghan and Lattanzio, "A refined particle
/// method for astrophysical problems", Astron. Astrophys. 149 (1985),
/// 135-143), normalized for 3D:
///
///     W(r, h) = sigma / h^3 * { 1 - 1.5 q^2 + 0.75 q^3   0 <= q < 1
///                                0.25 (2 - q)^3           1 <= q < 2
///                                0                        q >= 2 }
///     q = r / h,  sigma = 1 / pi
///
/// compact support at 2h, so a particle only feels neighbours within that
/// radius. See docs/physics.md for the density estimate, the equation of
/// state, and the symmetric pressure force's conservation properties.
///
/// **Scope.** No artificial viscosity, so this is suited to smooth, low
/// Mach-number flows rather than anything with a shock; Physics/Fluids'
/// Eulerian rung is built for that regime instead. No self-gravity: an
/// application couples this to Physics/Gravity's own accelerations if a
/// scenario needs both.

[[nodiscard]] double cubicSplineKernel(double r, double smoothingLength);
[[nodiscard]] Vec3 cubicSplineKernelGradient(const Vec3& separation,
                                             double smoothingLength);

/// Updates every particle's density (the kernel sum over every other
/// particle, including itself) and pressure (a polytropic equation of
/// state, P = equationOfStateK * density^polytropicIndex) in place.
void computeDensityAndPressure(std::span<SPHParticle> particles, double smoothingLength,
                               double equationOfStateK, double polytropicIndex);

/// The symmetric SPH pressure-gradient acceleration on every particle,
///
///     a_i = - sum_j  m_j (P_i / rho_i^2 + P_j / rho_j^2) grad_i W_ij
///
/// which conserves momentum exactly: the same reasoning as Newtonian
/// gravity's pairwise antisymmetry, since W_ij's gradient with respect to
/// r_i is exactly the negative of its gradient with respect to r_j.
/// Densities and pressures must already be current; call
/// computeDensityAndPressure first.
[[nodiscard]] std::vector<Vec3>
pressureAccelerations(std::span<const SPHParticle> particles, double smoothingLength);

}  // namespace ysq
