#pragma once

#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

/// The same dimension as Torque, declared there as well. Both names are kept
/// because code that means one and not the other should be able to say which,
/// even though the type system cannot hold them apart.
using Energy = Mul<Force, Length>;
using Power = Div<Energy, Time>;
using Action = Mul<Energy, Time>;
using SpecificEnergy = Div<Energy, Mass>;

}  // namespace dim

using Energy = Quantity<dim::Energy>;
using Power = Quantity<dim::Power>;
using Action = Quantity<dim::Action>;
using SpecificEnergy = Quantity<dim::SpecificEnergy>;

namespace units {

inline constexpr Energy joule{1.0};

/// The CGS unit of energy, 1e-7 J, exact.
inline constexpr Energy erg{1.0e-7};

/// The electronvolt: the elementary charge times one volt, so it is exact,
/// because the elementary charge is one of the constants that defines the SI.
inline constexpr Energy electronvolt{1.602176634e-19};
inline constexpr Energy kiloelectronvolt{1.602176634e-16};
inline constexpr Energy megaelectronvolt{1.602176634e-13};
inline constexpr Energy gigaelectronvolt{1.602176634e-10};

inline constexpr Power watt{1.0};

}  // namespace units

namespace literals {

[[nodiscard]] constexpr Energy operator""_J(long double value) {
    return Energy{static_cast<double>(value)};
}

[[nodiscard]] constexpr Energy operator""_J(unsigned long long value) {
    return Energy{static_cast<double>(value)};
}

[[nodiscard]] constexpr Energy operator""_eV(long double value) {
    return units::electronvolt * static_cast<double>(value);
}

[[nodiscard]] constexpr Energy operator""_eV(unsigned long long value) {
    return units::electronvolt * static_cast<double>(value);
}

[[nodiscard]] constexpr Energy operator""_MeV(long double value) {
    return units::megaelectronvolt * static_cast<double>(value);
}

[[nodiscard]] constexpr Energy operator""_MeV(unsigned long long value) {
    return units::megaelectronvolt * static_cast<double>(value);
}

[[nodiscard]] constexpr Power operator""_W(long double value) {
    return Power{static_cast<double>(value)};
}

[[nodiscard]] constexpr Power operator""_W(unsigned long long value) {
    return Power{static_cast<double>(value)};
}

}  // namespace literals

}  // namespace ysq
