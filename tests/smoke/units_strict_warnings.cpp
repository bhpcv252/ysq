/// Compiles every Units header under ysq::warnings_strict, which adds
/// -Wconversion -Wsign-conversion -Wdouble-promotion (and the MSVC
/// equivalents). In CI those are errors.
///
/// Units is an INTERFACE library and cannot carry the strict set itself
/// without pushing it onto Renderer, UI and Applications, which the root
/// README's Warnings section rules out. So the check lives here, exactly as
/// it does for Math in math_strict_warnings.cpp.
///
/// The explicit instantiations are the point. An uninstantiated template is
/// barely checked at all, so without them this file would compile clean no
/// matter what the headers said. float is not optional coverage:
/// -Wdouble-promotion only has anything to say when the value type is
/// narrower than double, so what the float half catches is a plain `2.0`
/// written inside a template where `T{2}` was meant. That is not
/// hypothetical: injecting one into Quantity's operator* makes this file fail
/// to compile, which is how the claim was checked rather than assumed.
///
/// It is specifically *not* the float-quantity-meets-double-unit-constant
/// case. That one never reaches a warning at all, because the two are
/// different types and the type system rejects it outright.
///
/// Including each header first, alone, is also what proves it is
/// self-contained rather than quietly relying on a sibling.

#include <Units/Acceleration.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Format.hpp>
#include <Units/Length.hpp>
#include <Units/Luminosity.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <format>
#include <type_traits>

template class ysq::Quantity<ysq::dim::Length, float>;
template class ysq::Quantity<ysq::dim::Length, double>;
template class ysq::Quantity<ysq::dim::Mass, float>;
template class ysq::Quantity<ysq::dim::Mass, double>;
template class ysq::Quantity<ysq::dim::Dimensionless, float>;
template class ysq::Quantity<ysq::dim::Dimensionless, double>;
template class ysq::Quantity<ysq::dim::Length, ysq::Vector2<float>>;
template class ysq::Quantity<ysq::dim::Length, ysq::Vector2<double>>;
template class ysq::Quantity<ysq::dim::Length, ysq::Vector3<float>>;
template class ysq::Quantity<ysq::dim::Length, ysq::Vector3<double>>;
template class ysq::Quantity<ysq::dim::Length, ysq::Vector4<float>>;
template class ysq::Quantity<ysq::dim::Length, ysq::Vector4<double>>;

namespace {

/// Calls everything on scalar-valued quantities. The result is accumulated
/// and returned only so nothing is discarded and the whole body has to be
/// generated.
template <class T>
T exerciseScalarQuantities() {
    using Length = ysq::Quantity<ysq::dim::Length, T>;
    using Time = ysq::Quantity<ysq::dim::Time, T>;
    using Mass = ysq::Quantity<ysq::dim::Mass, T>;
    using Area = ysq::Quantity<ysq::dim::Area, T>;
    using Volume = ysq::Quantity<ysq::dim::Volume, T>;
    using Force = ysq::Quantity<ysq::dim::Force, T>;

    T acc{};

    Length a{T{3}};
    const Length b{T{4}};
    const Time t{T{2}};
    const Mass m{T{5}};

    a += b;
    a -= b;
    a *= T{2};
    a /= T{2};

    acc += (+a - -a + b).value();
    acc += (a * T{2}).value() + (T{2} * a).value() + (a / T{2}).value();
    acc += (a / t).value();
    acc += (m * (a / t / t)).value();
    acc += (a * b).value();
    acc += (a * b * b).value();
    acc += (T{1} / t).value();
    acc += a.in(b);
    acc += a.value() + Length::zero().value();

    acc += ysq::raised<2>(a).value() + ysq::raised<3>(a).value();
    acc += ysq::raised<-1>(a).value() + ysq::raised<0>(a);
    acc += ysq::sqrt(Area{T{16}}).value();
    acc += ysq::root<3>(Volume{T{27}}).value();

    acc += ysq::min(a, b).value() + ysq::max(a, b).value();
    acc += ysq::clamp(a, Length{T{0}}, b).value();
    acc += ysq::abs(-a).value() + ysq::sign(a);
    acc += ysq::lerp(a, b, T{0.5}).value();
    acc += ysq::approxEqual(a, b) ? T{1} : T{0};
    acc += ysq::approxEqual(a, b, ysq::kDefaultRelTol<T>, Length{T{1}}) ? T{1} : T{0};
    acc += ysq::isNearZero(Length{T{0}}) ? T{1} : T{0};
    acc += ysq::isNearZero(a, Length{T{1}}) ? T{1} : T{0};

    acc += (a < b) || (a > b) || (a <= b) || (a >= b) || (a == b) ? T{1} : T{0};

    // The dimensionless quantity converts to its number, and only that way.
    const ysq::Quantity<ysq::dim::Dimensionless, T> ratio = a / b;
    acc += static_cast<T>(ratio);
    acc += ratio;

    // A law, written the way Physics will write one.
    const Force force = m * (a / (t * t));
    acc += force.value();

    acc += static_cast<T>(std::format("{:.3f}", a).size());
    acc += static_cast<T>(std::format("{:.3e}", force).size());
    acc += static_cast<T>(std::format("{}", ratio).size());

    return acc;
}

/// The same for vector-valued quantities, over all three widths.
template <class T>
T exerciseVectorQuantities() {
    using V2 = ysq::Vector2<T>;
    using V3 = ysq::Vector3<T>;
    using V4 = ysq::Vector4<T>;

    using Length = ysq::Quantity<ysq::dim::Length, T>;
    using Length2 = ysq::Quantity<ysq::dim::Length, V2>;
    using Length3 = ysq::Quantity<ysq::dim::Length, V3>;
    using Length4 = ysq::Quantity<ysq::dim::Length, V4>;
    using Force2 = ysq::Quantity<ysq::dim::Force, V2>;
    using Force3 = ysq::Quantity<ysq::dim::Force, V3>;
    using Time = ysq::Quantity<ysq::dim::Time, T>;

    T acc{};

    Length3 a{V3{T{3}, T{4}, T{0}}};
    const Length3 b{V3{T{1}, T{2}, T{3}}};
    const Force3 f{V3{T{0}, T{1}, T{0}}};
    const Time t{T{2}};

    a += b;
    a -= b;
    a *= T{2};
    a /= T{2};

    acc += (+a - -a + b).value().x;
    acc += (a * T{2}).value().y + (T{2} * a).value().z + (a / T{2}).value().x;
    acc += (a * t).value().x + (t * a).value().y + (a / t).value().z;
    acc += ysq::dot(a, f).value();
    acc += ysq::cross(a, f).value().z;
    acc += ysq::length(a).value() + ysq::lengthSquared(a).value();
    acc += ysq::normalized(a).x;
    acc += ysq::tryNormalized(a).value_or(V3::zero()).y;
    acc += ysq::distance(a, b).value() + ysq::distanceSquared(a, b).value();
    acc += ysq::lerp(a, b, T{0.5}).value().x;
    acc += ysq::abs(a).value().z;
    acc += (a == b) ? T{1} : T{0};
    acc += a.in(Length{T{1}}).x;

    // A magnitude times a direction, which is how a force law assembles one.
    const Length magnitude = ysq::length(a);
    acc += (magnitude * ysq::normalized(a)).value().x;
    acc += (ysq::normalized(a) * magnitude).value().y;

    const Length2 planar{V2{T{3}, T{4}}};
    const Force2 planarForce{V2{T{0}, T{1}}};
    acc += ysq::length(planar).value();
    acc += ysq::dot(planar, planarForce).value();
    acc += ysq::cross(planar, planarForce).value();
    acc += ysq::normalized(planar).x;

    const Length4 fourVector{V4{T{1}, T{2}, T{2}, T{4}}};
    acc += ysq::length(fourVector).value();
    acc += ysq::dot(fourVector, fourVector).value();
    acc += ysq::normalized(fourVector).w;

    acc += static_cast<T>(std::format("{:.2f}", a).size());
    acc += static_cast<T>(std::format("{:.2f}", planar).size());
    acc += static_cast<T>(std::format("{:.2f}", fourVector).size());

    return acc;
}

/// The named aliases and unit catalogue, which are double-valued and so are
/// exercised once rather than for both precisions.
double exerciseTheCatalogue() {
    using namespace ysq;
    using namespace ysq::literals;

    double acc = 0.0;

    acc += (1.0_m + 1.0_km + 1.0_cm + 1.0_mm + 1.0_au + 1.0_pc + 1.0_ly).value();
    acc += (1.0_kg + 1.0_g + 1.0_Msun + 1.0_Mearth).value();
    acc +=
        (1.0_s + 1.0_ms + 1.0_min + 1.0_h + 1.0_day + 1.0_yr + 1.0_Myr + 1.0_Gyr).value();
    acc += (1.0_Hz).value() + (1.0_mps + 1.0_kmps).value();
    acc += (1.0_mps2 + 1.0_g0).value();
    acc += (1.0_N).value() + (1.0_Pa + 1.0_bar).value();
    acc += (1.0_J + 1.0_eV + 1.0_MeV).value() + (1.0_W).value();
    acc += (1.0_K).value() + (1.0_Lsun).value() + (1.0_cd).value();
    acc += (1_m + 2_km + 3_au).value() + (1_kg).value() + (1_s + 2_yr).value();

    acc += units::metre.value() + units::kilometre.value();
    acc += units::centimetre.value() + units::millimetre.value();
    acc += units::micrometre.value() + units::nanometre.value();
    acc += units::angstrom.value() + units::astronomicalUnit.value();
    acc += units::parsec.value() + units::lightYear.value();
    acc += units::solarRadius.value() + units::squareMetre.value();
    acc += units::cubicMetre.value();
    acc += units::kilogram.value() + units::gram.value() + units::tonne.value();
    acc += units::atomicMassUnit.value() + units::solarMass.value();
    acc += units::earthMass.value() + units::kilogramPerCubicMetre.value();
    acc += units::second.value() + units::millisecond.value();
    acc += units::microsecond.value() + units::nanosecond.value();
    acc += units::minute.value() + units::hour.value() + units::day.value();
    acc += units::year.value() + units::megayear.value() + units::gigayear.value();
    acc += units::hertz.value();
    acc += units::metrePerSecond.value() + units::kilometrePerSecond.value();
    acc += units::kilometrePerHour.value() + units::speedOfLight.value();
    acc += units::metrePerSecondSquared.value() + units::standardGravity.value();
    acc += units::newton.value() + units::dyne.value() + units::pascal.value();
    acc += units::bar.value() + units::atmosphere.value();
    acc += units::kilogramMetrePerSecond.value();
    acc += units::joule.value() + units::erg.value() + units::electronvolt.value();
    acc += units::kiloelectronvolt.value() + units::megaelectronvolt.value();
    acc += units::gigaelectronvolt.value() + units::watt.value();
    acc += units::kelvin.value() + units::joulePerKelvin.value();
    acc += units::solarLuminosity.value();
    acc += units::candela.value() + units::lumen.value() + units::lux.value();

    acc += constants::caesiumHyperfineFrequency.value();
    acc += constants::speedOfLight.value() + constants::planckConstant.value();
    acc += constants::elementaryCharge.value();
    acc += constants::boltzmannConstant.value();
    acc += constants::avogadroConstant.value();
    acc += constants::luminousEfficacy.value();
    acc += constants::reducedPlanckConstant.value();
    acc += constants::nominalSolarMassParameter.value();
    acc += constants::nominalEarthMassParameter.value();

    acc += fromCelsius(20.0).value() + toCelsius(Temperature{300.0});
    acc += fromFahrenheit(70.0).value() + toFahrenheit(Temperature{300.0});

    // Every named alias instantiated at least once.
    acc += Length{1.0}.value() + Area{1.0}.value() + Volume{1.0}.value();
    acc += WaveNumber{1.0}.value() + Density{1.0}.value();
    acc += SurfaceDensity{1.0}.value() + LinearDensity{1.0}.value();
    acc += Frequency{1.0}.value() + AngularVelocity{1.0}.value();
    acc += Speed{1.0}.value() + Acceleration{1.0}.value() + Jerk{1.0}.value();
    acc += Force{1.0}.value() + Momentum{1.0}.value() + Torque{1.0}.value();
    acc += AngularMomentum{1.0}.value() + Pressure{1.0}.value();
    acc += Energy{1.0}.value() + Power{1.0}.value() + Action{1.0}.value();
    acc += SpecificEnergy{1.0}.value() + Temperature{1.0}.value();
    acc += HeatCapacity{1.0}.value() + Entropy{1.0}.value();
    acc += RadiantPower{1.0}.value() + Irradiance{1.0}.value();
    acc += Radiance{1.0}.value() + LuminousIntensity{1.0}.value();
    acc += LuminousFlux{1.0}.value() + Illuminance{1.0}.value();
    acc += ElectricCharge{1.0}.value() + GravitationalParameter{1.0}.value();
    acc += LuminousEfficacy{1.0}.value();
    acc += Length2{Vec2::unitX()}.value().x + Length3{Vec3::unitX()}.value().x;
    acc += Length4{Vec4::unitX()}.value().x;
    acc += Velocity2{Vec2::unitX()}.value().x + Velocity3{Vec3::unitX()}.value().x;
    acc += Velocity4{Vec4::unitX()}.value().x;
    acc += Acceleration2{Vec2::unitX()}.value().x;
    acc += Acceleration3{Vec3::unitX()}.value().x;
    acc += Acceleration4{Vec4::unitX()}.value().x + Jerk3{Vec3::unitX()}.value().x;
    acc += Force2{Vec2::unitX()}.value().x + Force3{Vec3::unitX()}.value().x;
    acc += Force4{Vec4::unitX()}.value().x + Momentum2{Vec2::unitX()}.value().x;
    acc += Momentum3{Vec3::unitX()}.value().x + Momentum4{Vec4::unitX()}.value().x;
    acc += AngularMomentum3{Vec3::unitX()}.value().x;
    acc += Torque3{Vec3::unitX()}.value().x;

    return acc;
}

}  // namespace

TEST(UnitsStrictWarnings, EveryTemplateInstantiatesForFloatAndDouble) {
    // The compile is the assertion. Calling both keeps the instantiations from
    // being dead-stripped and gives the case something to report.
    EXPECT_TRUE(std::isfinite(exerciseScalarQuantities<float>()));
    EXPECT_TRUE(std::isfinite(exerciseScalarQuantities<double>()));
    EXPECT_TRUE(std::isfinite(exerciseVectorQuantities<float>()));
    EXPECT_TRUE(std::isfinite(exerciseVectorQuantities<double>()));
    EXPECT_TRUE(std::isfinite(exerciseTheCatalogue()));
}

TEST(UnitsStrictWarnings, QuantitiesAreLaidOutForDirectGpuUpload) {
    // Compute takes arrays of these. A dimension is a compile-time property
    // and must cost nothing in storage, so an array of quantities has to be
    // memcpy-able into a GPU buffer exactly as the underlying values are.
    static_assert(std::is_standard_layout_v<ysq::Length>);
    static_assert(std::is_trivially_copyable_v<ysq::Length>);
    static_assert(std::is_standard_layout_v<ysq::Length3>);
    static_assert(std::is_trivially_copyable_v<ysq::Length3>);

    static_assert(sizeof(ysq::Length) == sizeof(double));
    static_assert(sizeof(ysq::Length3) == sizeof(ysq::Vec3));
    static_assert(sizeof(ysq::Quantity<ysq::dim::Length, float>) == sizeof(float));
    static_assert(sizeof(ysq::Quantity<ysq::dim::Length, ysq::Vec3f>) ==
                  sizeof(ysq::Vec3f));
    static_assert(alignof(ysq::Length3) == alignof(ysq::Vec3));

    // A dimension carries no storage of its own at all.
    static_assert(sizeof(ysq::Quantity<ysq::dim::Dim<9, -9, 9, -9, 9, -9, 9>>) ==
                  sizeof(double));

    SUCCEED();
}
