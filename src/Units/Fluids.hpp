#pragma once

#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

/// Shear stress per velocity gradient: pressure times time, the form
/// Newton's law of viscosity, `tau = mu (dv/dy)`, requires so that `tau`
/// comes out with units of pressure.
using DynamicViscosity = Mul<Pressure, Time>;

/// Dynamic viscosity divided through by density: what is left once a fluid's
/// own inertia is factored out, the quantity that appears directly in the
/// Navier-Stokes equations' diffusive term.
using KinematicViscosity = Div<Area, Time>;

/// Energy per unit area, equivalently force per unit length: the energy cost
/// of extending a liquid surface, which is why a surface under tension pulls
/// itself into the shape of least area.
using SurfaceTension = Div<Force, Length>;

}  // namespace dim

using DynamicViscosity = Quantity<dim::DynamicViscosity>;
using KinematicViscosity = Quantity<dim::KinematicViscosity>;
using SurfaceTension = Quantity<dim::SurfaceTension>;

namespace units {

inline constexpr DynamicViscosity pascalSecond{1.0};
inline constexpr KinematicViscosity squareMetrePerSecond{1.0};
inline constexpr SurfaceTension newtonPerMetre{1.0};

}  // namespace units

}  // namespace ysq
