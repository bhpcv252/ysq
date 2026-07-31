/// Compiles every Math header under ysq::warnings_strict, which adds
/// -Wconversion -Wsign-conversion -Wdouble-promotion (and the MSVC
/// equivalents). In CI those are errors.
///
/// Math is an INTERFACE library and cannot carry the strict set itself without
/// pushing it onto Renderer, UI and Applications, which the root README's
/// Warnings section rules out. So the check lives here instead.
///
/// The explicit instantiations are the point. An uninstantiated template is
/// barely checked at all, so without them this file would compile clean no
/// matter what the headers said. Every class template is instantiated for both
/// float and double, and exercise<T>() calls every free function so the
/// function templates are instantiated too. float is not optional coverage:
/// -Wdouble-promotion only has anything to say when the element type is
/// narrower than double.
///
/// Including each header first, alone, is also what proves it is self-contained
/// rather than quietly relying on whatever a sibling header dragged in.

#include <Math/Calculus.hpp>
#include <Math/Complex.hpp>
#include <Math/CoordinateSystems.hpp>
#include <Math/Dual.hpp>
#include <Math/Format.hpp>
#include <Math/Grid.hpp>
#include <Math/Integrators/Adaptive.hpp>
#include <Math/Integrators/Euler.hpp>
#include <Math/Integrators/RK4.hpp>
#include <Math/Integrators/Symplectic.hpp>
#include <Math/Interpolation.hpp>
#include <Math/Intersection.hpp>
#include <Math/Matrix2.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Matrix4.hpp>
#include <Math/ODE.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Scalar.hpp>
#include <Math/Statistics.hpp>
#include <Math/Tensor.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <type_traits>
#include <utility>

template struct ysq::Vector2<float>;
template struct ysq::Vector2<double>;
template struct ysq::Vector3<float>;
template struct ysq::Vector3<double>;
template struct ysq::Vector4<float>;
template struct ysq::Vector4<double>;
template struct ysq::Matrix2<float>;
template struct ysq::Matrix2<double>;
template struct ysq::Matrix3<float>;
template struct ysq::Matrix3<double>;
template struct ysq::Matrix4<float>;
template struct ysq::Matrix4<double>;
template struct ysq::Quaternion<float>;
template struct ysq::Quaternion<double>;
template struct ysq::Complex<float>;
template struct ysq::Complex<double>;
template struct ysq::Dual<float>;
template struct ysq::Dual<double>;
template struct ysq::Tensor<float, 2, 4>;
template struct ysq::Tensor<double, 2, 4>;
template struct ysq::Tensor<double, 4, 4>;
template class ysq::Grid1D<float>;
template class ysq::Grid1D<double>;

// The composition this whole design exists for: a vector over a dual scalar.
// If Dual ever stops satisfying Numeric, this is where it stops compiling.
template struct ysq::Vector3<ysq::Dual<double>>;
template struct ysq::Matrix3<ysq::Dual<double>>;
template struct ysq::Complex<ysq::Dual<double>>;

namespace {

/// Calls everything. The result is accumulated and returned only so nothing is
/// discarded and the whole body has to be generated.
template <class T>
T exercise() {
    using V2 = ysq::Vector2<T>;
    using V3 = ysq::Vector3<T>;
    using V4 = ysq::Vector4<T>;

    T acc{};

    acc += ysq::radians(T{45});
    acc += ysq::degrees(T{1});
    acc += ysq::clamp(T{2}, T{0}, T{1});
    acc += ysq::sign(T{-3});
    acc += ysq::approxEqual(T{1}, T{1}) ? T{1} : T{0};
    acc += ysq::approxEqual(T{1}, T{2}, T{1}, T{1}) ? T{1} : T{0};
    acc += ysq::isNearZero(T{0}) ? T{1} : T{0};
    acc += ysq::kPi<T> + ysq::kTau<T> + ysq::kE<T>;
    acc += ysq::kDefaultRelTol<T> + ysq::kDefaultAbsTol<T>;

    V2 a2{T{1}, T{2}};
    const V2 b2{T{3}, T{4}};
    a2 += b2;
    a2 -= b2;
    a2 *= T{2};
    a2 /= T{2};
    acc += ysq::dot(+a2 + b2 - a2, -b2);
    acc += ysq::dot(a2 * T{2}, T{2} * b2 / T{2});
    acc += ysq::cross(a2, b2);
    acc += ysq::lengthSquared(ysq::perpendicular(a2));
    acc += ysq::length(a2);
    acc += ysq::normalized(a2).x;
    acc += ysq::tryNormalized(a2).value_or(V2::zero()).y;
    acc += ysq::distance(a2, b2) + ysq::distanceSquared(a2, b2);
    acc += ysq::lerp(a2, b2, T{0.5}).x;
    acc += ysq::project(a2, b2).x + ysq::reject(a2, b2).y;
    acc += ysq::reflect(a2, V2::unitX()).x;
    acc += ysq::hadamard(a2, b2).x;
    acc += ysq::min(a2, b2).x + ysq::max(a2, b2).y;
    acc += ysq::abs(a2).x;
    acc += ysq::angleBetween(a2, b2);
    acc += a2[0] + std::as_const(a2)[1];
    acc += V2::splat(T{1}).x + V2::unitY().y + V2::zero().x;
    acc += (a2 == b2) ? T{1} : T{0};
    acc += static_cast<T>(V2::size());

    V3 a3{T{1}, T{2}, T{3}};
    const V3 b3{T{4}, T{5}, T{6}};
    a3 += b3;
    a3 -= b3;
    a3 *= T{2};
    a3 /= T{2};
    acc += ysq::dot(+a3 + b3 - a3, -b3);
    acc += ysq::dot(a3 * T{2}, T{2} * b3 / T{2});
    acc += ysq::cross(a3, b3).x;
    acc += ysq::scalarTriple(a3, b3, V3::unitZ());
    acc += ysq::lengthSquared(a3) + ysq::length(a3);
    acc += ysq::normalized(a3).x;
    acc += ysq::tryNormalized(a3).value_or(V3::zero()).y;
    acc += ysq::distance(a3, b3) + ysq::distanceSquared(a3, b3);
    acc += ysq::lerp(a3, b3, T{0.5}).x;
    acc += ysq::project(a3, b3).x + ysq::reject(a3, b3).y;
    acc += ysq::reflect(a3, V3::unitX()).x;
    acc += ysq::hadamard(a3, b3).x;
    acc += ysq::min(a3, b3).x + ysq::max(a3, b3).y;
    acc += ysq::abs(a3).z;
    acc += ysq::angleBetween(a3, b3);
    acc += ysq::rotateAbout(a3, V3::unitZ(), ysq::kPi<T>).x;
    acc += a3[0] + std::as_const(a3)[2];
    acc += a3.xy().x;
    acc += V3::splat(T{1}).x + V3::unitX().x + V3::unitY().y + V3::unitZ().z;
    acc += V3::zero().x;
    acc += (a3 == b3) ? T{1} : T{0};
    acc += static_cast<T>(V3::size());

    V4 a4{T{1}, T{2}, T{3}, T{4}};
    const V4 b4{T{5}, T{6}, T{7}, T{8}};
    a4 += b4;
    a4 -= b4;
    a4 *= T{2};
    a4 /= T{2};
    acc += ysq::dot(+a4 + b4 - a4, -b4);
    acc += ysq::dot(a4 * T{2}, T{2} * b4 / T{2});
    acc += ysq::lengthSquared(a4) + ysq::length(a4);
    acc += ysq::normalized(a4).x;
    acc += ysq::tryNormalized(a4).value_or(V4::zero()).y;
    acc += ysq::distance(a4, b4) + ysq::distanceSquared(a4, b4);
    acc += ysq::lerp(a4, b4, T{0.5}).x;
    acc += ysq::project(a4, b4).x + ysq::reject(a4, b4).y;
    acc += ysq::hadamard(a4, b4).x;
    acc += ysq::min(a4, b4).x + ysq::max(a4, b4).y;
    acc += ysq::abs(a4).w;
    acc += ysq::perspectiveDivide(a4).x;
    acc += a4[0] + std::as_const(a4)[3];
    acc += a4.xy().x + a4.xyz().z;
    acc += V4::point(a3).w + V4::direction(a3).w;
    acc +=
        V4::splat(T{1}).x + V4::unitX().x + V4::unitY().y + V4::unitZ().z + V4::unitW().w;
    acc += V4::zero().x;
    acc += (a4 == b4) ? T{1} : T{0};
    acc += static_cast<T>(V4::size());

    acc += static_cast<T>(std::format("{}", a2).size());
    acc += static_cast<T>(std::format("{:.3f}", a3).size());
    acc += static_cast<T>(std::format("{:+.2e}", a4).size());

    return acc;
}

template <class T>
T exerciseMatrices() {
    using V2 = ysq::Vector2<T>;
    using V3 = ysq::Vector3<T>;
    using V4 = ysq::Vector4<T>;
    using M2 = ysq::Matrix2<T>;
    using M3 = ysq::Matrix3<T>;
    using M4 = ysq::Matrix4<T>;

    T acc{};

    M2 a2 = M2::fromRows({T{2}, T{1}}, {T{1}, T{3}});
    const M2 b2 = M2::rotation(ysq::radians(T{30}));
    a2 += b2;
    a2 -= b2;
    a2 *= T{2};
    a2 /= T{2};
    a2 *= b2;
    acc += (+a2 - -a2 + b2)(0, 0);
    acc += (a2 * T{2})(0, 1) + (T{2} * a2)(1, 0) + (a2 / T{2})(1, 1);
    acc += (a2 * b2)(0, 0) + (a2 * V2::unitX()).y;
    acc += a2[0].x + std::as_const(a2)[1].y + a2.row(0).x;
    acc += ysq::transpose(a2)(0, 1) + ysq::determinant(a2) + ysq::trace(a2);
    acc += ysq::inverse(b2)(0, 0);
    acc += ysq::tryInverse(b2).value_or(M2::identity())(1, 1);
    acc += ysq::solve(b2, V2::unitX()).value_or(V2::zero()).x;
    acc += M2::zero()(0, 0) + M2::identity()(0, 0) + M2::diagonal(V2::unitX())(0, 0);
    acc += M2::scale(V2::splat(T{2}))(0, 0);
    acc += (a2 == b2) ? T{1} : T{0};
    acc += static_cast<T>(M2::rows() + M2::cols());

    M3 a3 = M3::fromRows({T{2}, T{1}, T{0}}, {T{1}, T{3}, T{1}}, {T{0}, T{1}, T{4}});
    const M3 b3 = M3::rotation(V3::unitZ(), ysq::radians(T{30}));
    a3 += b3;
    a3 -= b3;
    a3 *= T{2};
    a3 /= T{2};
    a3 *= b3;
    acc += (+a3 - -a3 + b3)(0, 0);
    acc += (a3 * T{2})(0, 1) + (T{2} * a3)(1, 0) + (a3 / T{2})(1, 1);
    acc += (a3 * b3)(0, 0) + (a3 * V3::unitX()).y;
    acc += a3[0].x + std::as_const(a3)[1].y + a3.row(0).x + a3.upperLeft2x2()(0, 0);
    acc += ysq::transpose(a3)(0, 1) + ysq::determinant(a3) + ysq::trace(a3);
    acc += ysq::inverse(b3)(0, 0) + ysq::inverseOrthogonal(b3)(0, 0);
    acc += ysq::tryInverse(b3).value_or(M3::identity())(1, 1);
    acc += ysq::solve(b3, V3::unitX()).value_or(V3::zero()).x;
    acc += M3::zero()(0, 0) + M3::identity()(0, 0) + M3::diagonal(V3::unitX())(0, 0);
    acc += M3::scale(V3::splat(T{2}))(0, 0);
    acc += M3::outerProduct(V3::unitX(), V3::unitY())(0, 1);
    acc += M3::crossMatrix(V3::unitZ())(0, 1);
    acc +=
        M3::rotationX(T{1})(1, 1) + M3::rotationY(T{1})(0, 0) + M3::rotationZ(T{1})(0, 0);
    acc += (a3 == b3) ? T{1} : T{0};
    acc += static_cast<T>(M3::rows() + M3::cols());

    M4 a4 = M4::fromRows({T{2}, T{1}, T{0}, T{0}}, {T{1}, T{3}, T{1}, T{0}},
                         {T{0}, T{1}, T{4}, T{0}}, {T{0}, T{0}, T{0}, T{1}});
    const M4 b4 = M4::rotation(V3::unitZ(), ysq::radians(T{30}));
    a4 += b4;
    a4 -= b4;
    a4 *= T{2};
    a4 /= T{2};
    a4 *= b4;
    acc += (+a4 - -a4 + b4)(0, 0);
    acc += (a4 * T{2})(0, 1) + (T{2} * a4)(1, 0) + (a4 / T{2})(1, 1);
    acc += (a4 * b4)(0, 0) + (a4 * V4::unitX()).y;
    acc += a4[0].x + std::as_const(a4)[1].y + a4.row(0).x;
    acc += a4.upperLeft3x3()(0, 0) + a4.translationPart().x;
    acc += ysq::transpose(a4)(0, 1) + ysq::determinant(a4) + ysq::trace(a4);
    acc += ysq::adjugate(a4)(0, 0) + ysq::inverse(b4)(0, 0);
    acc += ysq::inverseAffine(b4)(0, 0);
    acc += ysq::tryInverse(b4).value_or(M4::identity())(1, 1);
    acc += ysq::solve(b4, V4::unitX()).value_or(V4::zero()).x;
    acc += ysq::transformPoint(b4, V3::unitX()).x;
    acc += ysq::transformDirection(b4, V3::unitX()).y;
    acc += ysq::projectPoint(b4, V3::unitX()).z;
    acc += M4::zero()(0, 0) + M4::identity()(0, 0) + M4::diagonal(V4::unitX())(0, 0);
    acc += M4::translation(V3::unitX())(0, 3) + M4::scale(V3::splat(T{2}))(0, 0);
    acc += M4::fromLinear(b3)(0, 0) + M4::fromLinearTranslation(b3, V3::unitX())(0, 3);
    acc +=
        M4::rotationX(T{1})(1, 1) + M4::rotationY(T{1})(0, 0) + M4::rotationZ(T{1})(0, 0);
    acc += M4::lookAt(V3::splat(T{3}), V3::zero(), V3::unitY())(0, 0);
    acc += M4::perspective(ysq::radians(T{60}), T{2}, T{1}, T{100})(0, 0);
    acc += M4::orthographic(T{-1}, T{1}, T{-1}, T{1}, T{1}, T{100})(0, 0);
    acc += (a4 == b4) ? T{1} : T{0};
    acc += static_cast<T>(M4::rows() + M4::cols());

    acc += static_cast<T>(std::format("{}", a2).size());
    acc += static_cast<T>(std::format("{:.3f}", a3).size());
    acc += static_cast<T>(std::format("{:+.2e}", a4).size());

    return acc;
}

template <class T>
T exerciseQuaternions() {
    using V3 = ysq::Vector3<T>;
    using Q = ysq::Quaternion<T>;

    T acc{};

    Q a = Q::fromAxisAngle(V3::unitZ(), ysq::radians(T{30}));
    const Q b = Q::fromEulerZYX(T{1}, ysq::radians(T{20}), T{-1});
    a += b;
    a -= b;
    a *= T{2};
    a /= T{2};
    a = ysq::normalized(a);
    a *= b;
    acc += (+a - -a + b).w;
    acc += (a * T{2}).x + (T{2} * a).y + (a / T{2}).z;
    acc += (a * b).w + ysq::dot(a, b);
    acc += a[0] + std::as_const(a)[3] + a.xyz().x;
    acc += ysq::lengthSquared(a) + ysq::length(a);
    acc += ysq::conjugate(a).x + ysq::inverse(a).y + ysq::inverseUnit(a).z;
    acc += ysq::tryNormalized(a).value_or(Q::identity()).w;
    acc += ysq::rotate(a, V3::unitX()).x;
    acc += ysq::toMatrix3(a)(0, 0);
    acc += Q::fromRotationMatrix(ysq::toMatrix3(a)).w;
    acc += Q::fromRotationMatrix(ysq::Matrix3<T>::rotationX(ysq::kPi<T>)).x;
    acc += ysq::toAxisAngle(a).axis.x + ysq::toAxisAngle(a).angle;
    acc += ysq::toEulerZYX(a).yaw + ysq::toEulerZYX(a).pitch + ysq::toEulerZYX(a).roll;
    acc += ysq::angleBetween(a, b);
    acc += ysq::slerp(a, b, T{0.25}).w + ysq::nlerp(a, b, T{0.25}).x;
    acc += Q::identity().w + Q::zero().w;
    acc += Q::fromScalarVector(T{1}, V3::unitX()).x;
    acc += (a == b) ? T{1} : T{0};
    acc += static_cast<T>(Q::size());

    acc += static_cast<T>(std::format("{:.3f}", a).size());

    return acc;
}

template <class T>
T exerciseScalars() {
    using C = ysq::Complex<T>;
    using D = ysq::Dual<T>;

    T acc{};

    C a = C{T{3}, T{4}};
    const C b = C{T{1}, T{-2}};
    a += b;
    a -= b;
    a *= T{2};
    a /= T{2};
    a *= b;
    a /= b;
    acc += (+a - -a + b).re;
    acc += (a * T{2}).im + (T{2} * a).re + (a / T{2}).im;
    acc += (a + T{1}).re + (T{1} + a).re + (a - T{1}).im + (T{1} - a).im;
    acc += (a * b).re + (a / b).im;
    acc += a[0] + std::as_const(a)[1];
    acc += ysq::conj(a).im + ysq::lengthSquared(a) + ysq::abs(a) + ysq::length(a);
    acc += ysq::arg(a) + ysq::inverse(a).re + ysq::normalized(a).re;
    acc += ysq::tryNormalized(a).value_or(C::one()).re;
    acc += ysq::exp(a).re + ysq::log(a).im + ysq::sqrt(a).re;
    acc += ysq::sqrt(C{T{-4}, T{0}}).im + ysq::sqrt(C::zero()).re;
    acc += ysq::pow(a, b).re + ysq::pow(a, T{2}).im;
    acc += C::polar(T{2}, T{1}).re + C::real(T{1}).re + C::imaginary(T{1}).im;
    acc += C::zero().re + C::one().re + C::i().im;
    acc += (a == b) ? T{1} : T{0};
    acc += static_cast<T>(C::size());

    D x = D::variable(T{2});
    const D y = D::constant(T{3});
    x += y;
    x -= y;
    x *= y;
    x /= y;
    acc += (+x - -x + y).value;
    acc += (x + T{1}).value + (T{1} + x).derivative;
    acc += (x - T{1}).value + (T{1} - x).derivative;
    acc += (x * T{2}).value + (T{2} * x).derivative;
    acc += (x / T{2}).value + (T{2} / x).derivative;
    acc += x[0] + std::as_const(x)[1];
    acc += ysq::sqrt(x).derivative + ysq::exp(x).derivative + ysq::log(x).derivative;
    acc += ysq::sin(x).derivative + ysq::cos(x).derivative + ysq::tan(x).derivative;
    acc += ysq::asin(D{T{0}, T{1}}).derivative + ysq::acos(D{T{0}, T{1}}).derivative;
    acc += ysq::atan(x).derivative + ysq::atan2(x, y).derivative;
    acc += ysq::sinh(x).derivative + ysq::cosh(x).derivative + ysq::tanh(x).derivative;
    acc += ysq::asinh(x).derivative + ysq::acosh(x).derivative;
    acc += ysq::atanh(D{T{0}, T{1}}).derivative;
    acc += ysq::abs(x).derivative + ysq::hypot(x, y).derivative;
    acc += ysq::pow(x, y).derivative + ysq::pow(x, T{2}).derivative +
           ysq::pow(T{2}, x).derivative;
    acc += ysq::valueOf(x);
    acc += ysq::identical(x, y) ? T{1} : T{0};
    acc += ysq::derivative([](auto v) { return v * v; }, T{3});
    acc += ysq::valueAndDerivative([](auto v) { return v * v; }, T{3}).second;
    acc += ysq::secondDerivative([](auto v) { return v * v * v; }, T{3});
    acc += (x < y) || (x > y) || (x <= y) || (x >= y) || (x == y) ? T{1} : T{0};
    acc += static_cast<T>(D::size());

    // The composition the whole design is for.
    ysq::Vector3<D> vd{D::variable(T{1}), D::constant(T{2}), D::constant(T{3})};
    acc += ysq::length(vd).derivative + ysq::dot(vd, vd).derivative;
    acc += ysq::normalized(vd).x.derivative;
    acc += ysq::cross(vd, vd).x.value;

    acc += static_cast<T>(std::format("{:.3f}", a).size());
    acc += static_cast<T>(std::format("{:.3f}", x).size());

    return acc;
}

template <class T>
T exerciseTensors() {
    using T24 = ysq::Tensor<T, 2, 4>;
    using T14 = ysq::Tensor<T, 1, 4>;
    using T44 = ysq::Tensor<T, 4, 4>;

    T acc{};

    T24 g = T24::delta();
    const T24 h = T24::filled(T{2});
    g += h;
    g -= h;
    g *= T{2};
    g /= T{2};
    acc += (+g - -g + h)(0, 0);
    acc += (g * T{2})(1, 1) + (T{2} * g)(2, 2) + (g / T{2})(3, 3);
    acc += g(0, 1) + std::as_const(g)(1, 0) + g[0] + std::as_const(g)[1];
    acc += ysq::trace(g);
    acc += ysq::traceOver<0, 1>(g)();
    acc += ysq::transposeIndices<0, 1>(g)(0, 1);
    acc += ysq::symmetrize<0, 1>(g)(0, 1) + ysq::antisymmetrize<0, 1>(g)(0, 1);
    acc += static_cast<T>(T24::rank() + T24::dimension() + T24::size());
    acc += (g == h) ? T{1} : T{0};
    acc += T24::zero()(0, 0);

    const T14 v = ysq::toTensor(ysq::Vector4<T>::unitX());
    acc += ysq::outerProduct(v, v)(0, 0);
    acc += ysq::contract<0, 0>(v, v)();
    acc += ysq::contract<1, 0>(g, v)(0);
    acc += ysq::toVector4(v).x;
    acc += ysq::toVector3(ysq::toTensor(ysq::Vector3<T>::unitX())).x;
    acc += ysq::toMatrix4(g)(0, 0);
    acc += ysq::toMatrix3(ysq::toTensor(ysq::Matrix3<T>::identity()))(0, 0);

    const T44 riemann = T44::filled(T{1});
    acc += ysq::traceOver<0, 2>(riemann)(0, 0);
    acc += ysq::contract<3, 0>(riemann, v)(0, 0, 0);

    acc += static_cast<T>(std::format("{:.1f}", v).size());

    return acc;
}

template <class T>
T exerciseNumerics() {
    using V3 = ysq::Vector3<T>;

    T acc{};

    const std::array<T, 5> data{T{1}, T{2}, T{4}, T{8}, T{16}};
    const std::array<T, 5>& values = data;

    acc += ysq::sum(values) + ysq::naiveSum(values) + ysq::mean(values);
    acc += ysq::variance(values) + ysq::sampleVariance(values);
    acc += ysq::standardDeviation(values) + ysq::sampleStandardDeviation(values);
    acc += ysq::minimum(values) + ysq::maximum(values) + ysq::range(values);
    acc += ysq::median(values) + ysq::quantile(values, T{0.25});
    acc += ysq::covariance(values, values) + ysq::correlation(values, values);
    acc += ysq::linearFit(values, values).slope;
    acc += ysq::linearFit(values, values).intercept;
    acc += ysq::linearFit(values, values).rSquared;
    acc += static_cast<T>(ysq::histogram(values, 4, T{0}, T{16}).size());

    ysq::RunningStatistics<T> running;
    ysq::RunningStatistics<T> other;
    for (const T value : data) {
        running.add(value);
        other.add(value * T{2});
    }
    running.merge(other);
    acc += running.mean() + running.variance() + running.sampleVariance();
    acc += running.standardDeviation() + running.sampleStandardDeviation();
    acc += running.minimum() + running.maximum() + running.range();
    acc += static_cast<T>(running.count());
    running.reset();

    acc += ysq::lerp(T{1}, T{2}, T{0.5}) + ysq::inverseLerp(T{1}, T{2}, T{1.5});
    acc += ysq::remap(T{1}, T{0}, T{2}, T{0}, T{10});
    acc += ysq::smoothstep(T{0}, T{1}, T{0.5}) + ysq::smootherstep(T{0}, T{1}, T{0.5});
    acc += ysq::bilinear(T{0}, T{1}, T{2}, T{3}, T{0.5}, T{0.5});
    acc += ysq::trilinear(T{0}, T{1}, T{2}, T{3}, T{4}, T{5}, T{6}, T{7}, T{0.5}, T{0.5},
                          T{0.5});
    acc += ysq::cubicHermite(T{0}, T{1}, T{1}, T{1}, T{0.5});
    acc += ysq::catmullRom(T{0}, T{1}, T{2}, T{3}, T{0.5});
    acc += ysq::cubicBezier(T{0}, T{1}, T{2}, T{3}, T{0.5});
    acc += ysq::catmullRom(V3::zero(), V3::unitX(), V3::unitY(), V3::unitZ(), T{0.5}).x;
    acc += ysq::interpolateTable(values, values, T{3}).value_or(T{0});

    const auto spline = ysq::CubicSpline<T>::natural(values, values);
    if (spline) {
        acc += (*spline)(T{3}) + spline->derivative(T{3});
        acc += spline->lowerBound() + spline->upperBound();
        acc += static_cast<T>(spline->size());
    }

    const auto square = [](T x) { return x * x; };
    acc += ysq::onesidedStep(T{1}) + ysq::centralStep(T{1});
    acc += ysq::forwardDifference(square, T{2}) +
           ysq::forwardDifference(square, T{2}, T{1} / T{1024});
    acc += ysq::backwardDifference(square, T{2}) +
           ysq::backwardDifference(square, T{2}, T{1} / T{1024});
    acc += ysq::centralDifference(square, T{2}) +
           ysq::centralDifference(square, T{2}, T{1} / T{1024});
    acc += ysq::secondCentralDifference(square, T{2}) +
           ysq::secondCentralDifference(square, T{2}, T{1} / T{64});
    acc += ysq::richardsonDerivative(square, T{2}) +
           ysq::richardsonDerivative(square, T{2}, T{1} / T{8});

    const auto field = [](const auto& v) { return dot(v, v); };
    const auto flow = [](const auto& v) { return v * v.x; };
    acc += ysq::gradient(field, V3{T{1}, T{2}, T{3}}).x;
    acc += ysq::jacobian(flow, V3{T{1}, T{2}, T{3}})(0, 0);
    acc += ysq::hessian(field, V3{T{1}, T{2}, T{3}})(0, 0);
    acc += ysq::numericalGradient(field, V3{T{1}, T{2}, T{3}}).x;
    acc += ysq::numericalJacobian(flow, V3{T{1}, T{2}, T{3}})(0, 0);
    acc += ysq::numericalHessian(field, V3{T{1}, T{2}, T{3}})(0, 0);

    acc += ysq::trapezoid(square, T{0}, T{1}, 8);
    acc += ysq::simpson(square, T{0}, T{1}, 8);
    acc += ysq::adaptiveSimpson(square, T{0}, T{1}, T{1} / T{1024});
    acc += ysq::romberg(square, T{0}, T{1});
    acc += ysq::gaussLegendre<2>(square, T{0}, T{1});
    acc += ysq::gaussLegendre<3>(square, T{0}, T{1});
    acc += ysq::gaussLegendre<4>(square, T{0}, T{1});
    acc += ysq::gaussLegendre<5>(square, T{0}, T{1});

    const ysq::Spherical<T> sphere{T{2}, T{1}, T{-1}};
    const ysq::Cylindrical<T> cylinder{T{2}, T{1}, T{3}};
    const ysq::Polar<T> polar{T{2}, T{1}};

    acc += ysq::toCartesian(sphere).x + ysq::toCartesian(cylinder).y;
    acc += ysq::toCartesian(polar).x;
    acc += ysq::toSpherical(V3{T{1}, T{2}, T{3}}).polar;
    acc += ysq::toCylindrical(V3{T{1}, T{2}, T{3}}).azimuth;
    acc += ysq::toPolar(ysq::Vector2<T>{T{1}, T{2}}).angle;
    acc += ysq::sphericalBasis(sphere)(0, 0) + ysq::cylindricalBasis(cylinder)(0, 0);
    acc += ysq::polarBasis(polar)(0, 0);
    acc += ysq::sphericalJacobian(sphere)(0, 0);
    acc += ysq::cylindricalJacobian(cylinder)(0, 0);
    acc += ysq::sphericalComponentsToCartesian(sphere, V3::unitX()).x;
    acc += ysq::cartesianComponentsToSpherical(sphere, V3::unitX()).x;
    acc += ysq::cylindricalComponentsToCartesian(cylinder, V3::unitX()).x;
    acc += ysq::cartesianComponentsToCylindrical(cylinder, V3::unitX()).x;
    acc += sphere[0] + cylinder[1] + polar[0];
    acc += (sphere == sphere) ? T{1} : T{0};

    acc += static_cast<T>(std::format("{:.2f}", sphere).size());
    acc += static_cast<T>(std::format("{:.2f}", cylinder).size());
    acc += static_cast<T>(std::format("{:.2f}", polar).size());

    return acc;
}

template <class T>
T exerciseIntersection() {
    using V3 = ysq::Vector3<T>;

    T acc{};

    const ysq::Ray3<T> ray{V3{T{-5}, T{0}, T{0}}, V3{T{1}, T{0}, T{0}}};
    const ysq::Sphere3<T> sphere{V3::zero(), T{2}};

    acc += ysq::intersect(ray, sphere).value_or(T{0});
    acc +=
        ysq::segmentIntersectsSphere(V3{T{-5}, T{0}, T{0}}, V3{T{5}, T{0}, T{0}}, sphere)
            ? T{1}
            : T{0};

    return acc;
}

template <class T>
T exerciseGrid() {
    T acc{};

    ysq::Grid1D<T> grid(4, 0.5, 1);
    for (std::size_t i = 0; i < grid.cellCount(); ++i) {
        grid[static_cast<std::ptrdiff_t>(i)] = static_cast<T>(i);
    }
    grid.applyPeriodicBoundary();

    const ysq::Grid1D<T>& constGrid = grid;
    acc += constGrid[-1] + constGrid[0] +
           constGrid[static_cast<std::ptrdiff_t>(constGrid.cellCount())];
    acc += static_cast<T>(constGrid.cellCount() + constGrid.ghostCells());
    acc += static_cast<T>(constGrid.spacing());

    return acc;
}

template <class T>
T exerciseIntegrators() {
    using V3 = ysq::Vector3<T>;
    using Phase = ysq::PhaseState<V3>;

    T acc{};

    // dy/dt = -y, and the same problem written as a second-order system.
    const auto decay = [](T, const V3& y) { return y * T{-1}; };
    const auto springAcceleration = [](T, const V3& q) { return q * T{-1}; };
    const auto springSystem = ysq::asPhaseSystem(springAcceleration);

    const V3 start{T{1}, T{0}, T{0}};
    const Phase phaseStart{start, V3::zero()};
    const T step = T{1} / T{64};

    ysq::ExplicitEulerStepper<V3> euler;
    ysq::MidpointStepper<V3> midpoint;
    ysq::HeunStepper<V3> heun;
    ysq::Rk4Stepper<V3> rk4;
    ysq::DormandPrince54Stepper<V3> dormandPrince;

    acc += ysq::integrate(euler, decay, start, T{0}, T{1}, step).x;
    acc += ysq::integrate(midpoint, decay, start, T{0}, T{1}, step).x;
    acc += ysq::integrate(heun, decay, start, T{0}, T{1}, step).x;
    acc += ysq::integrate(rk4, decay, start, T{0}, T{1}, step).x;
    acc += ysq::integrate(dormandPrince, decay, start, T{0}, T{1}, step).x;

    acc +=
        static_cast<T>(euler.evaluations() + midpoint.evaluations() + heun.evaluations() +
                       rk4.evaluations() + dormandPrince.evaluations());
    acc +=
        static_cast<T>(ysq::ExplicitEulerStepper<V3>::order + ysq::Rk4Stepper<V3>::order +
                       ysq::DormandPrince54Stepper<V3>::embeddedOrder);

    ysq::SemiImplicitEulerStepper<V3> symplecticEuler;
    ysq::VelocityVerletStepper<V3> verlet;
    ysq::ForestRuthStepper<V3> forestRuth;
    ysq::PefrlStepper<V3> pefrl;

    acc +=
        ysq::integrate(symplecticEuler, springAcceleration, phaseStart, T{0}, T{1}, step)
            .position.x;
    acc += ysq::integrate(verlet, springAcceleration, phaseStart, T{0}, T{1}, step)
               .velocity.x;
    acc += ysq::integrate(forestRuth, springAcceleration, phaseStart, T{0}, T{1}, step)
               .position.x;
    acc += ysq::integrate(pefrl, springAcceleration, phaseStart, T{0}, T{1}, step)
               .position.x;
    acc += static_cast<T>(verlet.evaluations() + forestRuth.evaluations() +
                          pefrl.evaluations() + symplecticEuler.evaluations());

    // The same problem through an explicit method, via the phase-space wrapper.
    ysq::Rk4Stepper<Phase> phaseRk4;
    acc +=
        ysq::integrate(phaseRk4, springSystem, phaseStart, T{0}, T{1}, step).position.x;

    // An observer, and the step count the driver will use.
    T sampled{};
    acc += ysq::integrate(rk4, decay, start, T{0}, T{1}, step, [&](T, const V3& y) {
               sampled += y.x;
           }).x;
    acc += sampled;
    acc += static_cast<T>(ysq::stepCount(T{0}, T{1}, step));

    ysq::AdaptiveSettings<T> settings;
    settings.absoluteTolerance = static_cast<T>(1e-6);
    settings.relativeTolerance = static_cast<T>(1e-6);
    const auto adaptive =
        ysq::integrateAdaptive(dormandPrince, decay, start, T{0}, T{1}, step, settings);
    acc += adaptive.state.x + adaptive.time;
    acc += static_cast<T>(adaptive.acceptedSteps + adaptive.rejectedSteps +
                          adaptive.evaluations);
    acc += adaptive.succeeded ? T{1} : T{0};
    dormandPrince.reset();

    acc += ysq::errorNorm(start, start, start, T{1}, T{1});
    acc += ysq::errorNorm(T{1}, T{1}, T{1}, T{1}, T{1});

    // A dynamically sized state.
    using Dynamic = ysq::StateVector<T>;
    Dynamic dynamic(3, T{1});
    dynamic[0] = T{2};
    dynamic += dynamic;
    dynamic -= dynamic;
    dynamic *= T{2};
    dynamic /= T{2};
    acc += (dynamic + dynamic)[0] + (dynamic - dynamic)[1];
    acc += (dynamic * T{2})[0] + (T{2} * dynamic)[1] + (dynamic / T{2})[2];
    acc += (-dynamic)[0];
    acc += static_cast<T>(dynamic.size()) + (dynamic.empty() ? T{1} : T{0});
    acc += (dynamic == dynamic) ? T{1} : T{0};
    acc += *dynamic.begin() + *std::as_const(dynamic).begin() + dynamic.data()[0];
    dynamic.resize(4);

    ysq::Rk4Stepper<Dynamic> dynamicRk4;
    const auto dynamicDecay = [](T, const Dynamic& y) { return y * T{-1}; };
    acc += ysq::integrate(dynamicRk4, dynamicDecay, Dynamic{T{1}, T{2}, T{3}}, T{0}, T{1},
                          step)[0];

    // Phase-space arithmetic in its own right.
    Phase phase = phaseStart;
    phase += phaseStart;
    phase -= phaseStart;
    phase *= T{2};
    acc += (phase + phaseStart).position.x + (phase - phaseStart).velocity.x;
    acc += (phase * T{2}).position.x + (T{2} * phase).position.x;
    acc += (-phase).position.x + phase[0].x + std::as_const(phase)[1].x;
    acc += static_cast<T>(Phase::size());
    acc += (phase == phaseStart) ? T{1} : T{0};

    return acc;
}

}  // namespace

TEST(MathStrictWarnings, EveryTemplateInstantiatesForFloatAndDouble) {
    // The compile is the assertion. Calling both keeps the instantiations from
    // being dead-stripped and gives the case something to report.
    EXPECT_TRUE(std::isfinite(exercise<float>()));
    EXPECT_TRUE(std::isfinite(exercise<double>()));
    EXPECT_TRUE(std::isfinite(exerciseMatrices<float>()));
    EXPECT_TRUE(std::isfinite(exerciseMatrices<double>()));
    EXPECT_TRUE(std::isfinite(exerciseQuaternions<float>()));
    EXPECT_TRUE(std::isfinite(exerciseQuaternions<double>()));
    EXPECT_TRUE(std::isfinite(exerciseScalars<float>()));
    EXPECT_TRUE(std::isfinite(exerciseScalars<double>()));
    EXPECT_TRUE(std::isfinite(exerciseTensors<float>()));
    EXPECT_TRUE(std::isfinite(exerciseTensors<double>()));
    EXPECT_TRUE(std::isfinite(exerciseNumerics<float>()));
    EXPECT_TRUE(std::isfinite(exerciseNumerics<double>()));
    EXPECT_TRUE(std::isfinite(exerciseGrid<float>()));
    EXPECT_TRUE(std::isfinite(exerciseGrid<double>()));
    EXPECT_TRUE(std::isfinite(exerciseIntersection<float>()));
    EXPECT_TRUE(std::isfinite(exerciseIntersection<double>()));
    EXPECT_TRUE(std::isfinite(exerciseIntegrators<float>()));
    EXPECT_TRUE(std::isfinite(exerciseIntegrators<double>()));
}

TEST(MathStrictWarnings, VectorsAreLaidOutForDirectGpuUpload) {
    // An array of these has to be memcpy-able into a GPU buffer with no
    // repacking, which is what standard layout plus no padding buys.
    static_assert(std::is_standard_layout_v<ysq::Vec3>);
    static_assert(std::is_trivially_copyable_v<ysq::Vec3>);
    static_assert(std::is_standard_layout_v<ysq::Vec3f>);
    static_assert(std::is_trivially_copyable_v<ysq::Vec3f>);

    static_assert(sizeof(ysq::Vec2) == 2 * sizeof(double));
    static_assert(sizeof(ysq::Vec3) == 3 * sizeof(double));
    static_assert(sizeof(ysq::Vec4) == 4 * sizeof(double));
    static_assert(sizeof(ysq::Vec2f) == 2 * sizeof(float));
    static_assert(sizeof(ysq::Vec3f) == 3 * sizeof(float));
    static_assert(sizeof(ysq::Vec4f) == 4 * sizeof(float));

    // Default construction zeroes rather than leaving garbage.
    static_assert(ysq::Vec3{} == ysq::Vec3::zero());

    SUCCEED();
}
