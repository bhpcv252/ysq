#pragma once

#include <Physics/Body.hpp>
#include <Units/Constants.hpp>
#include <Units/Electromagnetism.hpp>
#include <Units/Force.hpp>

#include <span>

namespace ysq {

namespace dim {

using VacuumPermeability = Div<Force, Raise<Current, 2>>;
using VacuumPermittivity = Inverse<Mul<VacuumPermeability, Raise<Velocity, 2>>>;
using CoulombConstant = Inverse<VacuumPermittivity>;

}  // namespace dim

using VacuumPermeability = Quantity<dim::VacuumPermeability>;
using VacuumPermittivity = Quantity<dim::VacuumPermittivity>;
using CoulombConstant = Quantity<dim::CoulombConstant>;

namespace constants {

/// Vacuum permeability. Measured, not exact, since the 2019 SI
/// redefinition tied the ampere to the elementary charge rather than
/// defining mu0 outright: before then it was exactly 4 pi x 10^-7 by
/// definition, and the measured value today is extremely close to that but
/// not equal to it. Lives here, not in Units/Constants.hpp, for the same
/// reason G does in Physics/Gravity/Newtonian.hpp: it parameterizes one
/// interaction rather than the vocabulary physics is written in. CODATA
/// 2018/2022.
inline constexpr VacuumPermeability vacuumPermeability{1.25663706212e-6};

/// Computed from vacuumPermeability and the (exact) speed of light rather
/// than typed independently, so the two cannot drift apart from a retyped
/// digit.
inline constexpr VacuumPermittivity vacuumPermittivity{
    1.0 / (vacuumPermeability.value() * speedOfLight.value() * speedOfLight.value())};

/// Coulomb's constant, 1 / (4 pi epsilon0), likewise computed rather than
/// typed.
inline constexpr CoulombConstant coulombConstant{
    1.0 / (4.0 * kPi<double> * vacuumPermittivity.value())};

}  // namespace constants

/// Electric and magnetic fields, this rung of the ladder built from point
/// charges by direct superposition: Coulomb's law for E, the point-charge
/// form of Biot-Savart for B. Quasi-static, meaning it is each source's
/// present position and velocity that matter, not where it was one
/// light-travel-time ago; a field that actually propagates, sourced by
/// Maxwell's equations rather than assumed instantaneously, is the next rung
/// and is not implemented here. See src/Physics/README.md.

/// The electric field at `at` due to every point charge in `sources`,
/// superposed. Undefined, and skipped, at zero separation from a source.
[[nodiscard]] ElectricField3 electricField(const Length3& at,
                                           std::span<const Body> sources);

/// The magnetic field at `at` due to every moving point charge in
/// `sources`, superposed.
[[nodiscard]] MagneticFluxDensity3 magneticField(const Length3& at,
                                                 std::span<const Body> sources);

}  // namespace ysq
