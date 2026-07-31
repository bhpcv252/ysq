#pragma once

#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

using Frequency = Inverse<Time>;

}  // namespace dim

using Time = Quantity<dim::Time>;

/// Frequency and angular velocity are the same type, both being inverse time.
///
/// They are different physical quantities and a dimension-only system cannot
/// tell them apart, so an angular velocity handed to something expecting a
/// frequency compiles and is wrong by a factor of 2*pi. Distinguishing them
/// needs quantity kinds, which are deliberately out of scope; see
/// src/Units/README.md. The alias exists so that code that means an angular
/// velocity can at least say so.
using Frequency = Quantity<dim::Frequency>;
using AngularVelocity = Quantity<dim::Frequency>;
using AngularVelocity3 = Quantity<dim::Frequency, Vec3>;

namespace units {

inline constexpr Time second{1.0};
inline constexpr Time millisecond{1.0e-3};
inline constexpr Time microsecond{1.0e-6};
inline constexpr Time nanosecond{1.0e-9};

inline constexpr Time minute{60.0};
inline constexpr Time hour{3600.0};
inline constexpr Time day{86400.0};

/// The Julian year, exactly 365.25 days. This is the year astronomy uses for
/// intervals, and the one the light-year is defined against. It is not the
/// tropical year and not a calendar year, neither of which is a fixed number
/// of seconds.
inline constexpr Time year{31557600.0};
inline constexpr Time megayear{3.15576e13};
inline constexpr Time gigayear{3.15576e16};

inline constexpr Frequency hertz{1.0};

}  // namespace units

namespace literals {

// These suffixes collide with std::chrono_literals, which owns _s, _min, _h,
// _d and _y. Opening both namespaces unqualified in one scope is an ambiguity
// error, which is loud and takes seconds to fix, so the natural spelling is
// worth more than the avoidance. Engine headers never open this namespace.

[[nodiscard]] constexpr Time operator""_s(long double value) {
    return Time{static_cast<double>(value)};
}

[[nodiscard]] constexpr Time operator""_s(unsigned long long value) {
    return Time{static_cast<double>(value)};
}

[[nodiscard]] constexpr Time operator""_ms(long double value) {
    return units::millisecond * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_ms(unsigned long long value) {
    return units::millisecond * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_min(long double value) {
    return units::minute * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_min(unsigned long long value) {
    return units::minute * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_h(long double value) {
    return units::hour * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_h(unsigned long long value) {
    return units::hour * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_day(long double value) {
    return units::day * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_day(unsigned long long value) {
    return units::day * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_yr(long double value) {
    return units::year * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_yr(unsigned long long value) {
    return units::year * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_Myr(long double value) {
    return units::megayear * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_Myr(unsigned long long value) {
    return units::megayear * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_Gyr(long double value) {
    return units::gigayear * static_cast<double>(value);
}

[[nodiscard]] constexpr Time operator""_Gyr(unsigned long long value) {
    return units::gigayear * static_cast<double>(value);
}

[[nodiscard]] constexpr Frequency operator""_Hz(long double value) {
    return Frequency{static_cast<double>(value)};
}

[[nodiscard]] constexpr Frequency operator""_Hz(unsigned long long value) {
    return Frequency{static_cast<double>(value)};
}

}  // namespace literals

}  // namespace ysq
