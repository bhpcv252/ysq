/// Vector-valued quantities: a dimension wrapped around a Math vector.
///
/// The composition only goes this way round. Vector3<Length> cannot exist,
/// because Length does not satisfy Numeric (multiplication is not closed over
/// it), and Numeric is what Vector, Matrix and Tensor require of their element
/// type. So the dimension sits outside and the vector inside, and every vector
/// operation has to be given a dimensional meaning: dot and cross multiply
/// dimensions, length keeps one, and a direction has none at all.

#include <Units/Acceleration.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

namespace {

using namespace ysq;
using namespace ysq::literals;

constexpr Length3 kThreeFourZero{Vec3{3.0, 4.0, 0.0}};

/// A named concept, not a bare `requires`, for the same reason as in
/// units_dimensions.cpp: outside a template there is no substitution, so Clang
/// reports an invalid requirement as a hard error instead of `false`. That
/// holds inside a function body just as it does at namespace scope.
template <class A, class B>
concept Multipliable = requires(A a, B b) { a * b; };

}  // namespace

TEST(UnitsVector, ComponentwiseArithmeticKeepsTheDimension) {
    Length3 a{Vec3{1.0, 2.0, 3.0}};
    const Length3 b{Vec3{4.0, 5.0, 6.0}};

    EXPECT_VEC_APPROX((a + b).value(), (Vec3{5.0, 7.0, 9.0}));
    EXPECT_VEC_APPROX((a - b).value(), (Vec3{-3.0, -3.0, -3.0}));
    EXPECT_VEC_APPROX((-a).value(), (Vec3{-1.0, -2.0, -3.0}));

    a += b;
    EXPECT_VEC_APPROX(a.value(), (Vec3{5.0, 7.0, 9.0}));
    a -= b;
    EXPECT_VEC_APPROX(a.value(), (Vec3{1.0, 2.0, 3.0}));
    a *= 2.0;
    EXPECT_VEC_APPROX(a.value(), (Vec3{2.0, 4.0, 6.0}));
    a /= 2.0;
    EXPECT_VEC_APPROX(a.value(), (Vec3{1.0, 2.0, 3.0}));
}

TEST(UnitsVector, AScalarQuantityTimesAVectorQuantityIsAVectorQuantity) {
    const Velocity3 velocity{Vec3{10.0, 0.0, 0.0}};
    const Time elapsed{3.0};
    const Mass mass{2.0};

    const Length3 displacement = velocity * elapsed;
    static_assert(std::is_same_v<decltype(velocity * elapsed), Length3>);
    static_assert(std::is_same_v<decltype(elapsed * velocity), Length3>);
    EXPECT_VEC_APPROX(displacement.value(), (Vec3{30.0, 0.0, 0.0}));

    const Momentum3 momentum = mass * velocity;
    static_assert(std::is_same_v<decltype(mass * velocity), Momentum3>);
    EXPECT_VEC_APPROX(momentum.value(), (Vec3{20.0, 0.0, 0.0}));

    // And dividing by a scalar quantity comes back the other way.
    static_assert(std::is_same_v<decltype(displacement / elapsed), Velocity3>);
    EXPECT_QUANTITY_VEC_APPROX(displacement / elapsed, velocity);
}

TEST(UnitsVector, DotMultipliesDimensionsAndReturnsAScalarQuantity) {
    const Length3 displacement{Vec3{2.0, 0.0, 0.0}};
    const Force3 force{Vec3{3.0, 4.0, 0.0}};

    static_assert(std::is_same_v<decltype(dot(force, displacement)), Energy>);

    // Work is the component of force along the displacement, so only the x
    // part of this force does any.
    EXPECT_QUANTITY_APPROX(dot(force, displacement), Energy{6.0});
    EXPECT_QUANTITY_APPROX(dot(displacement, force), Energy{6.0});
}

TEST(UnitsVector, CrossMultipliesDimensionsAndKeepsTheVector) {
    const Length3 lever{Vec3{2.0, 0.0, 0.0}};
    const Force3 force{Vec3{0.0, 3.0, 0.0}};

    static_assert(std::is_same_v<decltype(cross(lever, force)), Torque3>);

    const Torque3 torque = cross(lever, force);
    EXPECT_VEC_APPROX(torque.value(), (Vec3{0.0, 0.0, 6.0}));

    // Antisymmetric, as it must be.
    EXPECT_QUANTITY_VEC_APPROX(cross(force, lever), -torque);

    // In two dimensions the cross product is a scalar, following Math.
    const Quantity<dim::Length, Vec2> planarLever{Vec2{2.0, 0.0}};
    const Quantity<dim::Force, Vec2> planarForce{Vec2{0.0, 3.0}};
    static_assert(std::is_same_v<decltype(cross(planarLever, planarForce)), Torque>);
    EXPECT_QUANTITY_APPROX(cross(planarLever, planarForce), Torque{6.0});
}

TEST(UnitsVector, LengthKeepsTheDimensionAndLengthSquaredDoublesIt) {
    static_assert(std::is_same_v<decltype(length(kThreeFourZero)), Length>);
    static_assert(std::is_same_v<decltype(lengthSquared(kThreeFourZero)), Area>);

    EXPECT_QUANTITY_APPROX(length(kThreeFourZero), Length{5.0});
    EXPECT_QUANTITY_APPROX(lengthSquared(kThreeFourZero), Area{25.0});

    // The identity that ties them together, and the reason sqrt is
    // constrained to even dimensions: an area has a square root, a length has
    // no business having one.
    EXPECT_QUANTITY_APPROX(sqrt(lengthSquared(kThreeFourZero)),
                           length(kThreeFourZero));
}

TEST(UnitsVector, ADirectionIsDimensionless) {
    // normalized returns the bare Math vector, not a quantity. A unit vector
    // is not a length of one; it has no dimension at all, and giving it one
    // would make every direction carry a unit it does not have.
    static_assert(std::is_same_v<decltype(normalized(kThreeFourZero)), Vec3>);

    EXPECT_VEC_APPROX(normalized(kThreeFourZero), (Vec3{0.6, 0.8, 0.0}));

    const auto direction = tryNormalized(kThreeFourZero);
    ASSERT_TRUE(direction.has_value());
    EXPECT_VEC_APPROX(*direction, (Vec3{0.6, 0.8, 0.0}));

    // Failure is reported, not invented, exactly as in Math.
    EXPECT_FALSE(tryNormalized(Length3{Vec3::zero()}).has_value());
    EXPECT_TRUE(std::isnan(normalized(Length3{Vec3::zero()}).x));
}

TEST(UnitsVector, AMagnitudeTimesADirectionRebuildsAVectorQuantity) {
    // The inverse of normalized(): physics produces a magnitude and a
    // direction separately more often than it produces the whole vector, and
    // this is the operation that puts them back together.
    const Force magnitude{10.0};
    const Vec3 direction{0.6, 0.8, 0.0};

    static_assert(std::is_same_v<decltype(magnitude * direction), Force3>);
    static_assert(std::is_same_v<decltype(direction * magnitude), Force3>);

    EXPECT_VEC_APPROX((magnitude * direction).value(), (Vec3{6.0, 8.0, 0.0}));
    EXPECT_QUANTITY_VEC_APPROX(direction * magnitude, magnitude * direction);
    EXPECT_QUANTITY_APPROX(length(magnitude * direction), magnitude);

    // Round trip through both halves.
    const Length3 position{Vec3{3.0, 4.0, 0.0}};
    EXPECT_QUANTITY_VEC_APPROX(length(position) * normalized(position), position);

    // The direction stays a bare vector, so its scalar type has to match; a
    // float direction and a double magnitude do not silently combine.
    static_assert(Multipliable<Force, Vec3>);
    static_assert(!Multipliable<Force, Vec3f>);
}

TEST(UnitsVector, ReadingAVectorQuantityInAUnitDividesComponentwise) {
    const Length3 position{Vec3{1.5e11, -3.0e11, 0.0}};

    static_assert(std::is_same_v<decltype(position.in(units::metre)), Vec3>);
    EXPECT_VEC_APPROX(position.in(units::metre), position.value());
    EXPECT_VEC_APPROX(position.in(units::astronomicalUnit),
                      (Vec3{1.5e11 / 149597870700.0, -3.0e11 / 149597870700.0, 0.0}));
}

TEST(UnitsVector, DistanceIsTheLengthOfTheDifference) {
    const Length3 a{Vec3{1.0, 2.0, 3.0}};
    const Length3 b{Vec3{4.0, 6.0, 3.0}};

    static_assert(std::is_same_v<decltype(distance(a, b)), Length>);
    static_assert(std::is_same_v<decltype(distanceSquared(a, b)), Area>);

    EXPECT_QUANTITY_APPROX(distance(a, b), Length{5.0});
    EXPECT_QUANTITY_APPROX(distanceSquared(a, b), Area{25.0});
}

TEST(UnitsVector, LerpAndAbsWorkComponentwise) {
    const Length3 a{Vec3{0.0, 0.0, 0.0}};
    const Length3 b{Vec3{10.0, 20.0, 30.0}};

    EXPECT_QUANTITY_VEC_APPROX(lerp(a, b, 0.0), a);
    EXPECT_QUANTITY_VEC_APPROX(lerp(a, b, 1.0), b);
    EXPECT_VEC_APPROX(lerp(a, b, 0.25).value(), (Vec3{2.5, 5.0, 7.5}));

    EXPECT_VEC_APPROX(abs(Length3{Vec3{-1.0, 2.0, -3.0}}).value(),
                      (Vec3{1.0, 2.0, 3.0}));
}

TEST(UnitsVector, AllFourWidthsExist) {
    // Vector4 is what Spacetime will want for four-vectors, and Vector2 is
    // what a planar scenario wants. The machinery is generic, so supporting
    // them costs an alias rather than an implementation.
    const Quantity<dim::Length, Vec2> planar{Vec2{3.0, 4.0}};
    const Quantity<dim::Length, Vec4> fourVector{Vec4{1.0, 2.0, 2.0, 4.0}};

    EXPECT_QUANTITY_APPROX(length(planar), Length{5.0});
    EXPECT_QUANTITY_APPROX(length(fourVector), Length{5.0});

    static_assert(sizeof(Quantity<dim::Length, Vec2>) == sizeof(Vec2));
    static_assert(sizeof(Quantity<dim::Length, Vec4>) == sizeof(Vec4));
}

TEST(UnitsVector, AWholeForceLawTypeChecks) {
    // The shape every Physics module will have: a law written once, in
    // quantities, with nothing to convert and nothing to remember.
    const Mass earth = 1.0_Mearth;
    const Mass satellite{1000.0};
    const Length3 separation{Vec3{7.0e6, 0.0, 0.0}};
    constexpr Quantity<dim::Div<dim::Raise<dim::Length, 3>,
                                dim::Mul<dim::Mass, dim::Raise<dim::Time, 2>>>>
        gravitationalConstant{6.67430e-11};

    const Area rangeSquared = lengthSquared(separation);
    const Vec3 direction = normalized(separation);
    const Force magnitude = gravitationalConstant * earth * satellite / rangeSquared;
    const Force3 gravity = -(magnitude * direction);

    // The magnitude, checked against the arithmetic done by hand.
    const double expected = 6.67430e-11 * earth.value() * 1000.0 / (7.0e6 * 7.0e6);
    EXPECT_QUANTITY_NEAR(length(gravity), Force{expected}, Force{1.0e-6});

    // And it points back along the separation.
    EXPECT_VEC_APPROX(normalized(gravity), -direction);
}
