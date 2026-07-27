#pragma once

#include <Units/Length.hpp>
#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

using Density = Div<Mass, Volume>;
using SurfaceDensity = Div<Mass, Area>;
using LinearDensity = Div<Mass, Length>;

}  // namespace dim

using Mass = Quantity<dim::Mass>;
using Density = Quantity<dim::Density>;
using SurfaceDensity = Quantity<dim::SurfaceDensity>;
using LinearDensity = Quantity<dim::LinearDensity>;

namespace units {

inline constexpr Mass kilogram{1.0};
inline constexpr Mass gram{1.0e-3};
inline constexpr Mass tonne{1.0e3};

/// The atomic mass constant, one twelfth the mass of a free carbon-12 atom at
/// rest in its ground state. Measured, CODATA 2022.
inline constexpr Mass atomicMassUnit{1.66053906892e-27};

/// Nominal solar and terrestrial masses.
///
/// These are the one place in this module where a unit is not exact and not
/// definable without leaving it. What the IAU fixes exactly is the mass
/// *parameter* GM, not the mass; recovering kilograms means dividing by a
/// measured G, so these carry G's uncertainty of roughly two parts in 100000.
/// The exact parameters are in Constants.hpp and are what an orbit should
/// actually be integrated with. docs/units.md records the G used here.
inline constexpr Mass solarMass{1.988409870698051e30};
inline constexpr Mass earthMass{5.972167867791379e24};

inline constexpr Density kilogramPerCubicMetre{1.0};

}  // namespace units

namespace literals {

[[nodiscard]] constexpr Mass operator""_kg(long double value) {
    return Mass{static_cast<double>(value)};
}

[[nodiscard]] constexpr Mass operator""_kg(unsigned long long value) {
    return Mass{static_cast<double>(value)};
}

[[nodiscard]] constexpr Mass operator""_g(long double value) {
    return units::gram * static_cast<double>(value);
}

[[nodiscard]] constexpr Mass operator""_g(unsigned long long value) {
    return units::gram * static_cast<double>(value);
}

[[nodiscard]] constexpr Mass operator""_Msun(long double value) {
    return units::solarMass * static_cast<double>(value);
}

[[nodiscard]] constexpr Mass operator""_Msun(unsigned long long value) {
    return units::solarMass * static_cast<double>(value);
}

[[nodiscard]] constexpr Mass operator""_Mearth(long double value) {
    return units::earthMass * static_cast<double>(value);
}

[[nodiscard]] constexpr Mass operator""_Mearth(unsigned long long value) {
    return units::earthMass * static_cast<double>(value);
}

}  // namespace literals

}  // namespace ysq
