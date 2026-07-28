#pragma once

#include <Units/Constants.hpp>
#include <Units/Force.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

namespace dim {

/// Force per charge: what a charge experiences, independent of its own
/// charge.
using ElectricField = Div<Force, ElectricCharge>;

/// Force per charge per speed: what the Lorentz force's magnetic term
/// divides out to.
using MagneticFluxDensity = Div<Force, Mul<ElectricCharge, Velocity>>;

}  // namespace dim

using ElectricField = Quantity<dim::ElectricField>;
using ElectricField3 = Quantity<dim::ElectricField, Vec3>;

using MagneticFluxDensity = Quantity<dim::MagneticFluxDensity>;
using MagneticFluxDensity3 = Quantity<dim::MagneticFluxDensity, Vec3>;

namespace units {

inline constexpr ElectricField voltPerMetre{1.0};
inline constexpr MagneticFluxDensity tesla{1.0};

/// The CGS unit of magnetic flux density, 1e-4 T. Still met in older
/// astrophysics and geophysics literature, the only reason it is here.
inline constexpr MagneticFluxDensity gauss{1.0e-4};

}  // namespace units

}  // namespace ysq
