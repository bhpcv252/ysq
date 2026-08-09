#pragma once

#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Time.hpp>
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

/// Energy per charge: the work needed to move a unit charge between two
/// points, independent of the charge actually moved.
using ElectricPotential = Div<Energy, ElectricCharge>;

/// Charge stored per unit potential difference.
using Capacitance = Div<ElectricCharge, ElectricPotential>;

/// Potential difference per current: Ohm's law, V = IR, solved for R.
using Resistance = Div<ElectricPotential, Current>;

/// The reciprocal of resistance: current per potential difference.
using Conductance = Inverse<Resistance>;

/// Potential times time: the quantity Faraday's law of induction relates to
/// a changing current, `dPhi/dt`, so it is the natural unit for the flux
/// itself.
using MagneticFlux = Mul<ElectricPotential, Time>;

/// Magnetic flux per current: what relates a circuit's own current to the
/// flux it induces through itself (self-inductance) or through another
/// circuit (mutual inductance).
using Inductance = Div<MagneticFlux, Current>;

}  // namespace dim

using ElectricField = Quantity<dim::ElectricField>;
using ElectricField3 = Quantity<dim::ElectricField, Vec3>;

using MagneticFluxDensity = Quantity<dim::MagneticFluxDensity>;
using MagneticFluxDensity3 = Quantity<dim::MagneticFluxDensity, Vec3>;

using ElectricPotential = Quantity<dim::ElectricPotential>;
using Capacitance = Quantity<dim::Capacitance>;
using Resistance = Quantity<dim::Resistance>;
using Conductance = Quantity<dim::Conductance>;
using MagneticFlux = Quantity<dim::MagneticFlux>;
using Inductance = Quantity<dim::Inductance>;

namespace units {

inline constexpr ElectricField voltPerMetre{1.0};
inline constexpr MagneticFluxDensity tesla{1.0};

/// The CGS unit of magnetic flux density, 1e-4 T. Still met in older
/// astrophysics and geophysics literature, the only reason it is here.
inline constexpr MagneticFluxDensity gauss{1.0e-4};

inline constexpr ElectricPotential volt{1.0};
inline constexpr Capacitance farad{1.0};
inline constexpr Resistance ohm{1.0};
inline constexpr Conductance siemens{1.0};
inline constexpr MagneticFlux weber{1.0};
inline constexpr Inductance henry{1.0};

}  // namespace units

}  // namespace ysq
