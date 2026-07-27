/// Arithmetic on scalar-valued quantities: that the numbers come out right,
/// and that the whole of it is usable at compile time.

#include <Units/Acceleration.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace {

using namespace ysq;
using namespace ysq::literals;

}  // namespace

TEST(UnitsQuantity, AdditionAndSubtractionActOnTheValue) {
    Length a{3.0};
    const Length b{4.0};

    EXPECT_EQ((a + b).value(), 7.0);
    EXPECT_EQ((a - b).value(), -1.0);
    EXPECT_EQ((-a).value(), -3.0);
    EXPECT_EQ((+a).value(), 3.0);

    a += b;
    EXPECT_EQ(a.value(), 7.0);
    a -= b;
    EXPECT_EQ(a.value(), 3.0);
}

TEST(UnitsQuantity, ScalingByABareNumberKeepsTheDimension) {
    Length a{3.0};

    static_assert(std::is_same_v<decltype(a * 2.0), Length>);
    static_assert(std::is_same_v<decltype(2.0 * a), Length>);
    static_assert(std::is_same_v<decltype(a / 2.0), Length>);

    EXPECT_EQ((a * 2.0).value(), 6.0);
    EXPECT_EQ((2.0 * a).value(), 6.0);
    EXPECT_EQ((a / 2.0).value(), 1.5);

    a *= 2.0;
    EXPECT_EQ(a.value(), 6.0);
    a /= 3.0;
    EXPECT_EQ(a.value(), 2.0);
}

TEST(UnitsQuantity, ProductsAndQuotientsCombineDimensions) {
    const Length distance{100.0};
    const Time elapsed{4.0};
    const Mass mass{2.0};

    const Speed speed = distance / elapsed;
    EXPECT_EQ(speed.value(), 25.0);

    const Acceleration acceleration = speed / elapsed;
    EXPECT_EQ(acceleration.value(), 6.25);

    const Force force = mass * acceleration;
    EXPECT_EQ(force.value(), 12.5);

    const Energy work = force * distance;
    EXPECT_EQ(work.value(), 1250.0);

    // The same energy by another route, since it is the same physics.
    const Energy kinetic = mass * speed * speed * 0.5;
    EXPECT_EQ(kinetic.value(), 625.0);
}

TEST(UnitsQuantity, DividingLikeByLikeGivesANumber) {
    const Length a{10.0};
    const Length b{4.0};

    static_assert(std::is_same_v<decltype(a / b), Dimensionless>);

    // And that number is usable as one, without an accessor.
    const double ratio = a / b;
    EXPECT_EQ(ratio, 2.5);
    EXPECT_TRUE(a / b > 2.0);
    EXPECT_EQ(std::floor(a / b), 2.0);
}

TEST(UnitsQuantity, ANumberOverAQuantityInvertsTheDimension) {
    const Time period{4.0};
    const Frequency frequency = 1.0 / period;

    static_assert(std::is_same_v<decltype(1.0 / period), Frequency>);
    EXPECT_EQ(frequency.value(), 0.25);
    EXPECT_EQ((frequency * period), 1.0);
}

TEST(UnitsQuantity, PowersMultiplyExponentsAndAreExact) {
    const Length side{3.0};

    static_assert(std::is_same_v<decltype(raised<2>(side)), Area>);
    static_assert(std::is_same_v<decltype(raised<3>(side)), Volume>);
    static_assert(std::is_same_v<decltype(raised<0>(side)), Dimensionless>);
    static_assert(std::is_same_v<decltype(raised<-1>(side)), WaveNumber>);

    // Repeated multiplication rather than exp(n log x), so this is exact
    // rather than within a few ulp.
    EXPECT_EQ(raised<2>(side).value(), 9.0);
    EXPECT_EQ(raised<3>(side).value(), 27.0);
    EXPECT_EQ(raised<0>(side), 1.0);
    EXPECT_EQ(raised<-1>(side).value(), 1.0 / 3.0);
    EXPECT_EQ(raised<2>(side), side * side);
}

TEST(UnitsQuantity, RootsHalveOrDivideTheExponents) {
    const Area area{16.0};
    const Volume volume{27.0};

    static_assert(std::is_same_v<decltype(sqrt(area)), Length>);
    static_assert(std::is_same_v<decltype(root<3>(volume)), Length>);

    EXPECT_EQ(sqrt(area).value(), 4.0);
    EXPECT_QUANTITY_NEAR(root<3>(volume), Length{3.0}, Length{1e-12});

    // Round trip.
    const Length side{7.0};
    EXPECT_QUANTITY_APPROX(sqrt(raised<2>(side)), side);
}

TEST(UnitsQuantity, ComparisonsOrderByValue) {
    const Mass light{1.0};
    const Mass heavy{2.0};

    EXPECT_TRUE(light < heavy);
    EXPECT_TRUE(heavy > light);
    EXPECT_TRUE(light <= light);
    EXPECT_TRUE(light >= light);
    EXPECT_TRUE(light == Mass{1.0});
    EXPECT_TRUE(light != heavy);

    EXPECT_EQ(min(light, heavy), light);
    EXPECT_EQ(max(light, heavy), heavy);
    EXPECT_EQ(clamp(Mass{5.0}, light, heavy), heavy);
    EXPECT_EQ(clamp(Mass{0.5}, light, heavy), light);
}

TEST(UnitsQuantity, AbsKeepsTheDimensionAndSignDropsIt) {
    // The magnitude of a length is a length.
    EXPECT_EQ(abs(Length{-3.0}), Length{3.0});
    EXPECT_EQ(abs(Length{3.0}), Length{3.0});
    static_assert(std::is_same_v<decltype(abs(Length{})), Length>);

    // Its sign is not. Minus one metre is not what anyone asking for a sign
    // means, and it would carry a spurious length back into whatever it
    // multiplied.
    static_assert(std::is_same_v<decltype(sign(Length{})), double>);
    EXPECT_EQ(sign(Length{-3.0}), -1.0);
    EXPECT_EQ(sign(Length{3.0}), 1.0);
    EXPECT_EQ(sign(Length{0.0}), 0.0);
}

TEST(UnitsQuantity, LerpIsExactAtBothEnds) {
    const Length a{2.0};
    const Length b{10.0};

    EXPECT_EQ(lerp(a, b, 0.0), a);
    EXPECT_EQ(lerp(a, b, 1.0), b);
    EXPECT_EQ(lerp(a, b, 0.25).value(), 4.0);

    // Extrapolates outside [0, 1], like Math's.
    EXPECT_EQ(lerp(a, b, 2.0).value(), 18.0);
}

TEST(UnitsQuantity, ApproximateComparisonTakesADimensionedTolerance) {
    const Length measured{1000.000001};
    const Length expected{1000.0};

    EXPECT_FALSE(approxEqual(measured, expected));
    EXPECT_TRUE(approxEqual(measured, expected, kDefaultRelTol<double>,
                            Length{1.0e-3}));
    EXPECT_QUANTITY_NEAR(measured, expected, 1.0 * units::millimetre);

    EXPECT_TRUE(isNearZero(Length{0.0}));
    EXPECT_FALSE(isNearZero(Length{1.0}));
    EXPECT_TRUE(isNearZero(Length{1.0e-6}, Length{1.0e-3}));
}

TEST(UnitsQuantity, NonFiniteValuesPropagateRatherThanBeingInvented) {
    // Following Math: the unchecked forms carry a NaN through instead of
    // returning something plausible, because that is the cheaper failure to
    // trace.
    const Length nan{std::numeric_limits<double>::quiet_NaN()};

    EXPECT_TRUE(std::isnan((nan + Length{1.0}).value()));
    EXPECT_TRUE(std::isnan((nan * 2.0).value()));
    EXPECT_FALSE(approxEqual(nan, nan));
    EXPECT_FALSE(nan < Length{1.0});
    EXPECT_FALSE(nan > Length{1.0});

    const Area infinite{std::numeric_limits<double>::infinity()};
    EXPECT_TRUE(std::isinf(sqrt(infinite).value()));
}

TEST(UnitsQuantity, TheWholeAlgebraWorksAtCompileTime) {
    // Not decoration: an initial condition, a unit conversion and a derived
    // constant should all be able to be constant-expressions, so nothing in
    // this module has to run before main.
    constexpr Mass mass{2.0};
    constexpr Length distance{100.0};
    constexpr Time elapsed{4.0};

    constexpr Speed speed = distance / elapsed;
    constexpr Energy kinetic = mass * speed * speed * 0.5;

    static_assert(speed.value() == 25.0);
    static_assert(kinetic.value() == 625.0);
    static_assert(raised<2>(distance).value() == 10000.0);
    static_assert(lerp(Length{0.0}, distance, 0.5).value() == 50.0);
    static_assert(min(Length{1.0}, Length{2.0}).value() == 1.0);
    static_assert(distance / distance == 1.0);
    static_assert(3.0_km == Length{3000.0});

    // sqrt is the exception, and only because std::sqrt is not constexpr
    // before C++26. Math/Scalar.hpp records the same boundary for length().
    EXPECT_EQ(sqrt(Area{4.0}).value(), 2.0);
}

TEST(UnitsQuantity, FloatQuantitiesAreTheirOwnType) {
    using Lengthf = Quantity<dim::Length, float>;

    constexpr Lengthf a{3.0F};
    constexpr Lengthf b{4.0F};

    static_assert(std::is_same_v<Lengthf::scalar_type, float>);
    static_assert(sizeof(Lengthf) == sizeof(float));
    static_assert((a + b).value() == 7.0F);
    static_assert(std::is_same_v<decltype(a * 2.0F), Lengthf>);
    SUCCEED();
}
