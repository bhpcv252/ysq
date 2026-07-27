#pragma once

#include <Units/Energy.hpp>
#include <Units/Length.hpp>
#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

/// Radiometry: energy per unit time, and how it is spread over area.
using RadiantPower = Power;
using Irradiance = Div<Power, Area>;

/// Photometry: the same physics weighted by the response of the human eye,
/// which is why luminous intensity is a base quantity of its own rather than
/// something derivable from watts. The weighting is a choice about observers,
/// not about light, and no amount of dimensional algebra recovers it.
using LuminousFlux = LuminousIntensity;
using Illuminance = Div<LuminousIntensity, Area>;

}  // namespace dim

using RadiantPower = Quantity<dim::RadiantPower>;
using Irradiance = Quantity<dim::Irradiance>;

/// Radiance is power per unit area per unit solid angle, and the steradian is
/// dimensionless, so radiance and irradiance share a dimension. Luminous flux
/// and luminous intensity do the same, for the same reason: the lumen is a
/// candela-steradian. Solid angle is a genuine physical distinction that a
/// dimension cannot carry; see src/Units/README.md.
using Radiance = Quantity<dim::Irradiance>;

using LuminousIntensity = Quantity<dim::LuminousIntensity>;
using LuminousFlux = Quantity<dim::LuminousFlux>;
using Illuminance = Quantity<dim::Illuminance>;

namespace units {

// No watt of its own here. RadiantPower *is* Power, so the unit is
// units::watt in Energy.hpp, and a second name for one constant of one type
// would be exactly the catalogue-of-everything-nameable this module's README
// rules out. The RadiantPower alias earns its place because it states intent;
// a duplicate unit constant would not.

/// IAU 2015 Resolution B3 nominal solar luminosity. A convention fixed to a
/// round number, chosen so that published stellar luminosities do not shift
/// every time the Sun is remeasured.
inline constexpr RadiantPower solarLuminosity{3.828e26};

inline constexpr LuminousIntensity candela{1.0};
inline constexpr LuminousFlux lumen{1.0};
inline constexpr Illuminance lux{1.0};

}  // namespace units

namespace literals {

[[nodiscard]] constexpr RadiantPower operator""_Lsun(long double value) {
    return units::solarLuminosity * static_cast<double>(value);
}

[[nodiscard]] constexpr RadiantPower operator""_Lsun(unsigned long long value) {
    return units::solarLuminosity * static_cast<double>(value);
}

[[nodiscard]] constexpr LuminousIntensity operator""_cd(long double value) {
    return LuminousIntensity{static_cast<double>(value)};
}

[[nodiscard]] constexpr LuminousIntensity operator""_cd(unsigned long long value) {
    return LuminousIntensity{static_cast<double>(value)};
}

}  // namespace literals

}  // namespace ysq
