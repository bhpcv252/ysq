#pragma once

#include <Units/Energy.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

using HeatCapacity = Div<Energy, Temperature>;

/// Power per unit length per unit temperature: Fourier's law of heat
/// conduction, `q = -k grad(T)`, needs this so the heat flux `q` comes out
/// in power per area.
using ThermalConductivity = Div<Power, Mul<Length, Temperature>>;

/// Area per time: the same dimension as `Units/Fluids.hpp`'s
/// `KinematicViscosity` (thermal diffusivity and momentum diffusivity are
/// the same kind of quantity, one governing how heat spreads and the other
/// how momentum does), declared independently here rather than depending on
/// that module for a `using` alias, the same relationship `HeatCapacity` and
/// `Entropy` already have below.
using ThermalDiffusivity = Div<Area, Time>;

/// Heat capacity per unit mass: the intensive form, a property of a
/// material independent of how much of it there is, unlike `HeatCapacity`
/// itself.
using SpecificHeatCapacity = Div<HeatCapacity, Mass>;

}  // namespace dim

using Temperature = Quantity<dim::Temperature>;

/// Entropy and heat capacity share a dimension, both being an energy per
/// kelvin, so these are two names for one type. Same limitation as torque and
/// energy; see src/Units/README.md.
using HeatCapacity = Quantity<dim::HeatCapacity>;
using Entropy = Quantity<dim::HeatCapacity>;

using ThermalConductivity = Quantity<dim::ThermalConductivity>;
using ThermalDiffusivity = Quantity<dim::ThermalDiffusivity>;
using SpecificHeatCapacity = Quantity<dim::SpecificHeatCapacity>;

namespace units {

inline constexpr Temperature kelvin{1.0};
inline constexpr HeatCapacity joulePerKelvin{1.0};
inline constexpr ThermalConductivity wattPerMetreKelvin{1.0};
inline constexpr SpecificHeatCapacity joulePerKilogramKelvin{1.0};

/// No unit constant of its own: `ThermalDiffusivity` shares its dimension
/// with `Units/Fluids.hpp`'s `KinematicViscosity` (see that type's own
/// comment above), so `units::squareMetrePerSecond` there already names it
/// -- the same one-name-for-one-dimension relationship `joulePerKelvin`
/// above already has for both `HeatCapacity` and `Entropy`.

}  // namespace units

/// Celsius and Fahrenheit are affine, not scaled, so they cannot be unit
/// constants like everything else in this module.
///
/// A unit constant works because conversion is multiplication: metres and
/// kilometres share an origin, so one is a fixed multiple of the other. Zero
/// Celsius is not zero kelvin, so there is no factor that converts between
/// them, and the arithmetic that makes `5.0 * units::kilometre` correct would
/// make `5.0 * units::celsius` silently wrong.
///
/// These are also why the module has no temperature-interval type. A
/// difference of two Celsius temperatures is a difference in kelvin, but the
/// type system here cannot tell an interval from an absolute value, so a sum
/// of two absolute temperatures compiles and is meaningless. Distinguishing
/// them needs the same machinery as separating torque from energy.
[[nodiscard]] constexpr Temperature fromCelsius(double degrees) noexcept {
    return Temperature{degrees + 273.15};
}

[[nodiscard]] constexpr double toCelsius(Temperature temperature) noexcept {
    return temperature.value() - 273.15;
}

[[nodiscard]] constexpr Temperature fromFahrenheit(double degrees) noexcept {
    return Temperature{(degrees - 32.0) * (5.0 / 9.0) + 273.15};
}

[[nodiscard]] constexpr double toFahrenheit(Temperature temperature) noexcept {
    return (temperature.value() - 273.15) * (9.0 / 5.0) + 32.0;
}

namespace literals {

[[nodiscard]] constexpr Temperature operator""_K(long double value) {
    return Temperature{static_cast<double>(value)};
}

[[nodiscard]] constexpr Temperature operator""_K(unsigned long long value) {
    return Temperature{static_cast<double>(value)};
}

}  // namespace literals

}  // namespace ysq
