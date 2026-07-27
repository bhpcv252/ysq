#pragma once

#include <Units/Acceleration.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

namespace dim {

using Force = Mul<Mass, Acceleration>;
using Momentum = Mul<Mass, Velocity>;
using AngularMomentum = Mul<Momentum, Length>;
using Pressure = Div<Force, Area>;

/// Torque is r x F, so it carries the dimension of an energy. That is not a
/// mistake in the table: the two genuinely are dimensionally identical, and no
/// system built on dimensions alone can separate them. See Energy.hpp, where
/// the same dimension is named again, and src/Units/README.md for why the
/// formatter refuses to print either as a joule.
using Torque = Mul<Force, Length>;

}  // namespace dim

using Force = Quantity<dim::Force>;
using Force2 = Quantity<dim::Force, Vec2>;
using Force3 = Quantity<dim::Force, Vec3>;
using Force4 = Quantity<dim::Force, Vec4>;

using Momentum = Quantity<dim::Momentum>;
using Momentum2 = Quantity<dim::Momentum, Vec2>;
using Momentum3 = Quantity<dim::Momentum, Vec3>;
using Momentum4 = Quantity<dim::Momentum, Vec4>;

using AngularMomentum = Quantity<dim::AngularMomentum>;
using AngularMomentum3 = Quantity<dim::AngularMomentum, Vec3>;

using Torque = Quantity<dim::Torque>;
using Torque3 = Quantity<dim::Torque, Vec3>;

using Pressure = Quantity<dim::Pressure>;

namespace units {

inline constexpr Force newton{1.0};

/// The CGS unit of force, 1 g cm/s^2. Still met in older astrophysics
/// literature, which is the only reason it is here.
inline constexpr Force dyne{1.0e-5};

inline constexpr Pressure pascal{1.0};
inline constexpr Pressure bar{1.0e5};

/// The standard atmosphere, exact by definition (CGPM 1954).
inline constexpr Pressure atmosphere{101325.0};

inline constexpr Momentum kilogramMetrePerSecond{1.0};

}  // namespace units

namespace literals {

[[nodiscard]] constexpr Force operator""_N(long double value) {
    return Force{static_cast<double>(value)};
}

[[nodiscard]] constexpr Force operator""_N(unsigned long long value) {
    return Force{static_cast<double>(value)};
}

[[nodiscard]] constexpr Pressure operator""_Pa(long double value) {
    return Pressure{static_cast<double>(value)};
}

[[nodiscard]] constexpr Pressure operator""_Pa(unsigned long long value) {
    return Pressure{static_cast<double>(value)};
}

[[nodiscard]] constexpr Pressure operator""_bar(long double value) {
    return units::bar * static_cast<double>(value);
}

[[nodiscard]] constexpr Pressure operator""_bar(unsigned long long value) {
    return units::bar * static_cast<double>(value);
}

}  // namespace literals

}  // namespace ysq
