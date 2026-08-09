#pragma once

#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

namespace dim {

using Acceleration = Div<Velocity, Time>;
using Jerk = Div<Acceleration, Time>;

/// Angular velocity per time: the same relationship `Acceleration` has to
/// `Velocity` above, one rotational step up.
using AngularAcceleration = Div<Frequency, Time>;

}  // namespace dim

using Acceleration = Quantity<dim::Acceleration>;
using Acceleration2 = Quantity<dim::Acceleration, Vec2>;
using Acceleration3 = Quantity<dim::Acceleration, Vec3>;
using Acceleration4 = Quantity<dim::Acceleration, Vec4>;

using Jerk = Quantity<dim::Jerk>;
using Jerk3 = Quantity<dim::Jerk, Vec3>;

using AngularAcceleration = Quantity<dim::AngularAcceleration>;
using AngularAcceleration3 = Quantity<dim::AngularAcceleration, Vec3>;

namespace units {

inline constexpr Acceleration metrePerSecondSquared{1.0};
inline constexpr AngularAcceleration radianPerSecondSquared{1.0};

/// Standard gravity: the conventional acceleration used to define the
/// kilogram-force and to quote g-loads. Exact by convention (CGPM 1901), and
/// not the actual acceleration anywhere in particular on Earth.
inline constexpr Acceleration standardGravity{9.80665};

}  // namespace units

namespace literals {

[[nodiscard]] constexpr Acceleration operator""_mps2(long double value) {
    return Acceleration{static_cast<double>(value)};
}

[[nodiscard]] constexpr Acceleration operator""_mps2(unsigned long long value) {
    return Acceleration{static_cast<double>(value)};
}

[[nodiscard]] constexpr Acceleration operator""_g0(long double value) {
    return units::standardGravity * static_cast<double>(value);
}

[[nodiscard]] constexpr Acceleration operator""_g0(unsigned long long value) {
    return units::standardGravity * static_cast<double>(value);
}

}  // namespace literals

}  // namespace ysq
