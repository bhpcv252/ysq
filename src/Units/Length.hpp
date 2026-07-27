#pragma once

#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

using Area = Raise<Length, 2>;
using Volume = Raise<Length, 3>;
using WaveNumber = Inverse<Length>;

}  // namespace dim

using Length = Quantity<dim::Length>;
using Length2 = Quantity<dim::Length, Vec2>;
using Length3 = Quantity<dim::Length, Vec3>;
using Length4 = Quantity<dim::Length, Vec4>;

using Area = Quantity<dim::Area>;
using Volume = Quantity<dim::Volume>;
using WaveNumber = Quantity<dim::WaveNumber>;

namespace units {

/// Unit constants are double-valued, deliberately, even though Quantity is
/// templated on its value type.
///
/// A conversion factor is a definition rather than a measurement, and a
/// definition rendered at float precision has been damaged before it is ever
/// used. A float quantity is still perfectly constructible; it just does not
/// get to spell its magnitude with a degraded copy of an exact number.

inline constexpr Length metre{1.0};
inline constexpr Length kilometre{1.0e3};
inline constexpr Length centimetre{1.0e-2};
inline constexpr Length millimetre{1.0e-3};
inline constexpr Length micrometre{1.0e-6};
inline constexpr Length nanometre{1.0e-9};

/// Exactly 1e-10 m. Not an SI unit, and the working unit of atomic spacing.
inline constexpr Length angstrom{1.0e-10};

/// Exact by definition since IAU 2012 Resolution B2, which redefined the
/// astronomical unit as a fixed number of metres rather than as something
/// derived from the Gaussian gravitational constant.
inline constexpr Length astronomicalUnit{1.495978707e11};

/// The distance at which one astronomical unit subtends one arcsecond:
/// exactly (648000 / pi) au. Irrational, so unlike the au it does not survive
/// a round trip bit-for-bit.
inline constexpr Length parsec{3.0856775814913673e16};

/// The distance light travels in one Julian year, both factors exact, so this
/// is exact: 299792458 m/s * 31557600 s.
inline constexpr Length lightYear{9.4607304725808e15};

/// IAU 2015 Resolution B3 nominal solar radius. A convention, not a
/// measurement of the actual Sun, which is neither spherical nor constant.
inline constexpr Length solarRadius{6.957e8};

inline constexpr Area squareMetre{1.0};
inline constexpr Volume cubicMetre{1.0};

}  // namespace units

namespace literals {

[[nodiscard]] constexpr Length operator""_m(long double value) {
    return Length{static_cast<double>(value)};
}

[[nodiscard]] constexpr Length operator""_m(unsigned long long value) {
    return Length{static_cast<double>(value)};
}

[[nodiscard]] constexpr Length operator""_km(long double value) {
    return units::kilometre * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_km(unsigned long long value) {
    return units::kilometre * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_cm(long double value) {
    return units::centimetre * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_cm(unsigned long long value) {
    return units::centimetre * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_mm(long double value) {
    return units::millimetre * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_mm(unsigned long long value) {
    return units::millimetre * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_au(long double value) {
    return units::astronomicalUnit * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_au(unsigned long long value) {
    return units::astronomicalUnit * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_pc(long double value) {
    return units::parsec * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_pc(unsigned long long value) {
    return units::parsec * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_ly(long double value) {
    return units::lightYear * static_cast<double>(value);
}

[[nodiscard]] constexpr Length operator""_ly(unsigned long long value) {
    return units::lightYear * static_cast<double>(value);
}

}  // namespace literals

}  // namespace ysq
