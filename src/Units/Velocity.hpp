#pragma once

#include <Units/Length.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

using Velocity = Div<Length, Time>;

}  // namespace dim

/// The scalar is a Speed and the vector is a Velocity, which is the
/// distinction physics already draws in words. Everything else in the module
/// separates the two with a numeric suffix, because English does not supply a
/// second word for most quantities.
using Speed = Quantity<dim::Velocity>;
using Velocity2 = Quantity<dim::Velocity, Vec2>;
using Velocity3 = Quantity<dim::Velocity, Vec3>;
using Velocity4 = Quantity<dim::Velocity, Vec4>;

namespace units {

inline constexpr Speed metrePerSecond{1.0};
inline constexpr Speed kilometrePerSecond{1.0e3};
inline constexpr Speed kilometrePerHour{1.0e3 / 3600.0};

/// The speed of light in vacuum, exact by definition since it is what fixes
/// the metre. Repeated in Constants.hpp, where it is a defining constant
/// rather than a unit of speed; both spellings are the same number and the
/// distinction is what it is being used for.
inline constexpr Speed speedOfLight{299792458.0};

}  // namespace units

namespace literals {

[[nodiscard]] constexpr Speed operator""_mps(long double value) {
    return Speed{static_cast<double>(value)};
}

[[nodiscard]] constexpr Speed operator""_mps(unsigned long long value) {
    return Speed{static_cast<double>(value)};
}

[[nodiscard]] constexpr Speed operator""_kmps(long double value) {
    return units::kilometrePerSecond * static_cast<double>(value);
}

[[nodiscard]] constexpr Speed operator""_kmps(unsigned long long value) {
    return units::kilometrePerSecond * static_cast<double>(value);
}

}  // namespace literals

}  // namespace ysq
