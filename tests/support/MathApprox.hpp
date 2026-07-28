#pragma once

#include <Math/Format.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <format>
#include <ostream>
#include <type_traits>

/// Approximate comparison for Math values, as GoogleTest predicate formats.
///
/// EXPECT_NEAR compares one scalar with an absolute tolerance, which is the
/// wrong tool twice over here: comparing two matrices element by element gives
/// a failure message that names neither matrix, and an absolute tolerance is
/// meaningless once magnitudes leave the neighbourhood of 1. These print both
/// operands, name the offending component, and use the mixed relative and
/// absolute rule from Math/Scalar.hpp.
///
///     EXPECT_VEC_APPROX(cross(a, b), expected);
///     EXPECT_VEC_NEAR(integrated, analytic, 1e-9);
///     EXPECT_MAT_APPROX(m * inverse(m), Mat4::identity());
///     EXPECT_APPROX(length(v), 5.0);
///
/// VEC covers anything with a static size() and operator[], which includes
/// Quaternion. MAT covers anything with static rows() and cols() and
/// operator()(row, col).

/// Formatters for the integrator state types.
///
/// These live here rather than in Math/Format.hpp because nothing in the
/// engine formats one: printing an integrator state is a debugging concern,
/// and putting them in the engine's formatting header would make it depend on
/// ODE.hpp and drag <vector> into every translation unit that prints a vector.
/// The generic VectorFormatter they reuse is public for exactly this.
template <class S, class CharT>
struct std::formatter<ysq::PhaseState<S>, CharT>
    : ysq::detail::VectorFormatter<ysq::PhaseState<S>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::StateVector<T>, CharT>
    : ysq::detail::VectorFormatter<ysq::StateVector<T>, CharT> {};

namespace ysq {

namespace test::detail {

/// True when following value_type down eventually reaches a floating-point
/// type.
///
/// This exists to keep the streaming operator below off std::string, which has
/// size(), operator[] and value_type and would otherwise match it and collide
/// with the standard one. A string bottoms out at char; every Math value
/// bottoms out at float or double, however deeply it is nested.
template <class T, class = void>
struct BottomsOutInAScalar : std::bool_constant<std::floating_point<T>> {};

template <class T>
struct BottomsOutInAScalar<T, std::void_t<typename T::value_type>>
    : std::bool_constant<std::floating_point<T> ||
                         BottomsOutInAScalar<typename T::value_type>::value> {};

}  // namespace test::detail

/// Streaming, not GoogleTest's PrintTo, because the two are reached by
/// different paths: PrintTo covers what EXPECT_EQ prints on failure, but
/// `EXPECT_TRUE(x) << someVector` goes through Message::operator<< and needs a
/// real operator<<. One overload set covers both.
///
/// These live in the test support header rather than in Math/Format.hpp
/// because <ostream> is heavier than <format> and no engine code needs it.
/// They are in namespace ysq so ADL finds them.
template <class V>
    requires requires(const V& v) {
        v.size();
        v[std::size_t{0}];
        typename V::value_type;
    } && test::detail::BottomsOutInAScalar<V>::value
std::ostream& operator<<(std::ostream& os, const V& value) {
    return os << std::format("{}", value);
}

template <class M>
    requires requires(const M& m) {
        M::rows();
        M::cols();
        m(std::size_t{0}, std::size_t{0});
        typename M::value_type;
    }
std::ostream& operator<<(std::ostream& os, const M& value) {
    return os << std::format("{}", value);
}

}  // namespace ysq

namespace ysq::test {

template <class V>
[[nodiscard]] ::testing::AssertionResult
componentsNear(const char* aExpr, const char* bExpr, const char* tolExpr, const V& a,
               const V& b, typename V::value_type tolerance) {
    using T = typename V::value_type;

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (approxEqual(a[i], b[i], tolerance, tolerance)) {
            continue;
        }
        const T difference = (a[i] < b[i]) ? (b[i] - a[i]) : (a[i] - b[i]);
        return ::testing::AssertionFailure()
               << std::format("\n  {} = {}\n  {} = {}\ncomponent {} differs by {:.6g}, "
                              "tolerance {:.6g} ({})",
                              aExpr, a, bExpr, b, i, difference, tolerance, tolExpr);
    }
    return ::testing::AssertionSuccess();
}

template <class V>
[[nodiscard]] ::testing::AssertionResult
componentsApprox(const char* aExpr, const char* bExpr, const V& a, const V& b) {
    return componentsNear(aExpr, bExpr, "default", a, b,
                          kDefaultRelTol<typename V::value_type>);
}

template <class M>
[[nodiscard]] ::testing::AssertionResult
elementsNear(const char* aExpr, const char* bExpr, const char* tolExpr, const M& a,
             const M& b, typename M::value_type tolerance) {
    using T = typename M::value_type;

    for (std::size_t r = 0; r < M::rows(); ++r) {
        for (std::size_t c = 0; c < M::cols(); ++c) {
            if (approxEqual(a(r, c), b(r, c), tolerance, tolerance)) {
                continue;
            }
            const T difference =
                (a(r, c) < b(r, c)) ? (b(r, c) - a(r, c)) : (a(r, c) - b(r, c));
            return ::testing::AssertionFailure() << std::format(
                       "\n  {} = {}\n  {} = {}\nelement ({}, {}) differs by "
                       "{:.6g}, tolerance {:.6g} ({})",
                       aExpr, a, bExpr, b, r, c, difference, tolerance, tolExpr);
        }
    }
    return ::testing::AssertionSuccess();
}

template <class M>
[[nodiscard]] ::testing::AssertionResult
elementsApprox(const char* aExpr, const char* bExpr, const M& a, const M& b) {
    return elementsNear(aExpr, bExpr, "default", a, b,
                        kDefaultRelTol<typename M::value_type>);
}

template <std::floating_point T>
[[nodiscard]] ::testing::AssertionResult scalarNear(const char* aExpr, const char* bExpr,
                                                    const char* tolExpr, T a, T b,
                                                    T tolerance) {
    if (approxEqual(a, b, tolerance, tolerance)) {
        return ::testing::AssertionSuccess();
    }
    const T difference = (a < b) ? (b - a) : (a - b);
    return ::testing::AssertionFailure() << std::format(
               "\n  {} = {:.17g}\n  {} = {:.17g}\ndiffers by {:.6g}, tolerance "
               "{:.6g} ({})",
               aExpr, a, bExpr, b, difference, tolerance, tolExpr);
}

template <std::floating_point T>
[[nodiscard]] ::testing::AssertionResult scalarApprox(const char* aExpr,
                                                      const char* bExpr, T a, T b) {
    return scalarNear(aExpr, bExpr, "default", a, b, kDefaultRelTol<T>);
}

}  // namespace ysq::test

#define EXPECT_VEC_NEAR(a, b, tol)                                                       \
    EXPECT_PRED_FORMAT3(::ysq::test::componentsNear, a, b, tol)
#define ASSERT_VEC_NEAR(a, b, tol)                                                       \
    ASSERT_PRED_FORMAT3(::ysq::test::componentsNear, a, b, tol)

#define EXPECT_VEC_APPROX(a, b) EXPECT_PRED_FORMAT2(::ysq::test::componentsApprox, a, b)
#define ASSERT_VEC_APPROX(a, b) ASSERT_PRED_FORMAT2(::ysq::test::componentsApprox, a, b)

// Tensor stores its components flat and exposes size() and operator[], so the
// VEC comparators already cover it. These are aliases purely so a tensor test
// does not read as if it were about vectors.
#define EXPECT_TENSOR_NEAR(a, b, tol) EXPECT_VEC_NEAR(a, b, tol)
#define EXPECT_TENSOR_APPROX(a, b) EXPECT_VEC_APPROX(a, b)

#define EXPECT_MAT_NEAR(a, b, tol)                                                       \
    EXPECT_PRED_FORMAT3(::ysq::test::elementsNear, a, b, tol)
#define ASSERT_MAT_NEAR(a, b, tol)                                                       \
    ASSERT_PRED_FORMAT3(::ysq::test::elementsNear, a, b, tol)

#define EXPECT_MAT_APPROX(a, b) EXPECT_PRED_FORMAT2(::ysq::test::elementsApprox, a, b)
#define ASSERT_MAT_APPROX(a, b) ASSERT_PRED_FORMAT2(::ysq::test::elementsApprox, a, b)

#define EXPECT_NEAR_REL(a, b, tol) EXPECT_PRED_FORMAT3(::ysq::test::scalarNear, a, b, tol)

#define EXPECT_APPROX(a, b) EXPECT_PRED_FORMAT2(::ysq::test::scalarApprox, a, b)
#define ASSERT_APPROX(a, b) ASSERT_PRED_FORMAT2(::ysq::test::scalarApprox, a, b)
