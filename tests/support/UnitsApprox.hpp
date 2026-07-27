#pragma once

#include <Units/Format.hpp>
#include <Units/Unit.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <format>
#include <ostream>

/// Approximate comparison for dimensioned quantities, as GoogleTest predicate
/// formats.
///
///     EXPECT_QUANTITY_APPROX(kinetic, expected);
///     EXPECT_QUANTITY_NEAR(orbit, analytic, 1.0 * units::kilometre);
///     EXPECT_QUANTITY_VEC_APPROX(force, expected);
///
/// Separate from MathApprox.hpp rather than bolted onto it, so that a test
/// exercising only Math does not acquire a dependency on Units.
///
/// The tolerance is itself a quantity. "Within a kilometre" is a length, and
/// writing it as 1000.0 would leave the reader to remember which unit the
/// number is in, which is the class of mistake this whole module exists to
/// remove. It is dimension-checked like everything else, so a length tolerance
/// on a mass comparison does not compile.
///
/// A Quantity has no size() and no operator[], so it matches none of the
/// generic comparators in MathApprox.hpp and none of its streaming operators.
/// That is deliberate; see src/Units/Unit.hpp.

namespace ysq {

template <class D, class V>
std::ostream& operator<<(std::ostream& os, const Quantity<D, V>& value) {
    return os << std::format("{}", value);
}

}  // namespace ysq

namespace ysq::test {

template <class Q>
[[nodiscard]] ::testing::AssertionResult quantitiesNear(const char* aExpr,
                                                        const char* bExpr,
                                                        const char* tolExpr, const Q& a,
                                                        const Q& b, const Q& tolerance) {
    using T = typename Q::scalar_type;
    if (approxEqual(a, b, kDefaultRelTol<T>, tolerance)) {
        return ::testing::AssertionSuccess();
    }
    const Q difference = abs(a - b);
    return ::testing::AssertionFailure()
           << std::format("\n  {} = {}\n  {} = {}\ndiffers by {}, tolerance {} ({})",
                          aExpr, a, bExpr, b, difference, tolerance, tolExpr);
}

template <class Q>
[[nodiscard]] ::testing::AssertionResult quantitiesApprox(const char* aExpr,
                                                          const char* bExpr, const Q& a,
                                                          const Q& b) {
    using T = typename Q::scalar_type;
    return quantitiesNear(aExpr, bExpr, "default", a, b,
                          Q{kDefaultAbsTol<T>});
}

/// Componentwise, on the same terms, for a vector-valued quantity. The
/// tolerance is the scalar-valued quantity of the same dimension, since a
/// per-component tolerance is a magnitude rather than a direction.
template <class Q, class Tol>
[[nodiscard]] ::testing::AssertionResult quantityComponentsNear(
    const char* aExpr, const char* bExpr, const char* tolExpr, const Q& a, const Q& b,
    const Tol& tolerance) {
    using T = typename Q::scalar_type;
    static_assert(std::same_as<typename Q::dimension, typename Tol::dimension>,
                  "The tolerance must have the same dimension as the operands.");

    const auto& av = a.value();
    const auto& bv = b.value();
    for (std::size_t i = 0; i < av.size(); ++i) {
        if (approxEqual(av[i], bv[i], kDefaultRelTol<T>, tolerance.value())) {
            continue;
        }
        const T difference = (av[i] < bv[i]) ? (bv[i] - av[i]) : (av[i] - bv[i]);
        return ::testing::AssertionFailure() << std::format(
                   "\n  {} = {}\n  {} = {}\ncomponent {} differs by {:.6g}, "
                   "tolerance {} ({})",
                   aExpr, a, bExpr, b, i, difference, tolerance, tolExpr);
    }
    return ::testing::AssertionSuccess();
}

template <class Q>
[[nodiscard]] ::testing::AssertionResult quantityComponentsApprox(const char* aExpr,
                                                                  const char* bExpr,
                                                                  const Q& a,
                                                                  const Q& b) {
    using T = typename Q::scalar_type;
    using Scalar = Quantity<typename Q::dimension, T>;
    return quantityComponentsNear(aExpr, bExpr, "default", a, b,
                                  Scalar{kDefaultAbsTol<T>});
}

}  // namespace ysq::test

#define EXPECT_QUANTITY_NEAR(a, b, tol) \
    EXPECT_PRED_FORMAT3(::ysq::test::quantitiesNear, a, b, tol)
#define ASSERT_QUANTITY_NEAR(a, b, tol) \
    ASSERT_PRED_FORMAT3(::ysq::test::quantitiesNear, a, b, tol)

#define EXPECT_QUANTITY_APPROX(a, b) \
    EXPECT_PRED_FORMAT2(::ysq::test::quantitiesApprox, a, b)
#define ASSERT_QUANTITY_APPROX(a, b) \
    ASSERT_PRED_FORMAT2(::ysq::test::quantitiesApprox, a, b)

#define EXPECT_QUANTITY_VEC_NEAR(a, b, tol) \
    EXPECT_PRED_FORMAT3(::ysq::test::quantityComponentsNear, a, b, tol)

#define EXPECT_QUANTITY_VEC_APPROX(a, b) \
    EXPECT_PRED_FORMAT2(::ysq::test::quantityComponentsApprox, a, b)
