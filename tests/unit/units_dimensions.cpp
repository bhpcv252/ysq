/// The dimension algebra, and the operations that must not compile.
///
/// This file is the one that decides whether the module does its job. A
/// quantity that adds correctly is worth little if a length also adds to a
/// mass, so the negative cases carry as much weight here as the positive ones,
/// and each is paired with the positive form immediately below it.
///
/// **Every negative check goes through a named concept.** A bare
/// `static_assert(!requires(Length l, Mass m) { l + m; })` at namespace scope
/// is not portable: outside a template there is no substitution, so Clang
/// reports the invalid requirement as a hard error rather than evaluating the
/// requires-expression to false, and the file fails to compile. Wrapping the
/// requirement in a concept and then checking `!Addable<Length, Mass>`
/// instantiates a template, which is what makes it a substitution failure.
/// GCC is laxer, so this would otherwise have passed locally and failed in CI.

#include <Units/Acceleration.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Luminosity.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using namespace ysq;

// The vocabulary the negative checks are written in.

template <class A, class B>
concept Addable = requires(A a, B b) { a + b; };

template <class A, class B>
concept Subtractable = requires(A a, B b) { a - b; };

template <class A, class B>
concept Multipliable = requires(A a, B b) { a * b; };

template <class A, class B>
concept Divisible = requires(A a, B b) { a / b; };

template <class A, class B>
concept Ordered = requires(A a, B b) { a < b; };

template <class A, class B>
concept EqualityComparable = requires(A a, B b) { a == b; };

template <class A, class B>
concept Assignable = requires(A a, B b) { a = b; };

template <class A, class B>
concept ConvertibleFrom = requires(B b) { A{b}; };

template <class T>
concept SquareRootable = requires(T t) { sqrt(t); };

/// Whether naming the alias is well-formed at all, as opposed to whether the
/// call using it compiles. These are different questions and the difference is
/// what broke the Windows build once; see the test below.
template <class D, int N>
concept RootAliasWellFormed = requires { typename ysq::dim::Root<D, N>; };

/// Whether a type can be a Math container's element type, which is what the
/// whole vector-quantity design turns on.
template <class T>
concept UsableAsAVectorElement = requires { typename ysq::Vector3<T>; };

template <int N, class T>
concept Raisable = requires(T t) { ysq::raised<N>(t); };

template <class A, class B>
concept Dottable = requires(A a, B b) { dot(a, b); };

template <class T>
concept Lengthable = requires(T t) { length(t); };

}  // namespace

// ---------------------------------------------------------------------------
// The algebra
// ---------------------------------------------------------------------------

TEST(UnitsDimensions, BaseDimensionsAreIndependent) {
    static_assert(!std::is_same_v<dim::Length, dim::Mass>);
    static_assert(!std::is_same_v<dim::Mass, dim::Time>);
    static_assert(!std::is_same_v<dim::Current, dim::Temperature>);
    static_assert(!std::is_same_v<dim::Amount, dim::LuminousIntensity>);

    // All seven exponents are carried, not just the ones mechanics uses.
    static_assert(dim::LuminousIntensity::luminousIntensity == 1);
    static_assert(dim::Amount::amount == 1);
    static_assert(dim::Current::current == 1);
    SUCCEED();
}

TEST(UnitsDimensions, MultiplicationAddsExponents) {
    static_assert(std::is_same_v<dim::Mul<dim::Length, dim::Length>, dim::Area>);
    static_assert(std::is_same_v<dim::Mul<dim::Area, dim::Length>, dim::Volume>);
    static_assert(std::is_same_v<dim::Div<dim::Length, dim::Length>, dim::Dimensionless>);
    static_assert(std::is_same_v<dim::Inverse<dim::Time>, dim::Frequency>);
    SUCCEED();
}

TEST(UnitsDimensions, PowersAndRootsInvertEachOther) {
    static_assert(std::is_same_v<dim::Raise<dim::Length, 2>, dim::Area>);
    static_assert(std::is_same_v<dim::Root<dim::Area, 2>, dim::Length>);
    static_assert(std::is_same_v<dim::Root<dim::Volume, 3>, dim::Length>);
    static_assert(std::is_same_v<dim::Raise<dim::Length, 0>, dim::Dimensionless>);
    static_assert(std::is_same_v<dim::Raise<dim::Length, -1>, dim::WaveNumber>);
    SUCCEED();
}

TEST(UnitsDimensions, ARootExistsOnlyWhenEveryExponentDivides) {
    static_assert(dim::RootExists<dim::Area, 2>);
    static_assert(dim::RootExists<dim::Volume, 3>);
    static_assert(dim::RootExists<dim::Dimensionless, 2>);

    // A length has no square root in a system with integer exponents, and
    // neither does a volume: L^3 does not halve.
    static_assert(!dim::RootExists<dim::Length, 2>);
    static_assert(!dim::RootExists<dim::Volume, 2>);
    static_assert(!dim::RootExists<dim::Area, 3>);

    // The whole dimension has to divide, not just the first exponent.
    // An energy is L^2 M T^-2: the L halves, the M does not.
    static_assert(!dim::RootExists<dim::Energy, 2>);

    // A squared energy does, which is what makes the check meaningful rather
    // than a blanket refusal.
    static_assert(dim::RootExists<dim::Raise<dim::Energy, 2>, 2>);
    SUCCEED();
}

TEST(UnitsDimensions, AnImpossibleRootIsASubstitutionFailureNotAHardError) {
    // This is a regression test for a Windows-only build failure, pinned so
    // that it is a Windows-only failure no longer.
    //
    // `Root` used to carry a static_assert with a friendly message. MSVC
    // substitutes into a function template's declared return type *before* it
    // checks the constraint, so evaluating `SquareRootable<Length>` reached
    // sqrt's `-> Quantity<dim::Root<D, 2>, V>`, fired the assertion, and ended
    // the translation unit. Clang and GCC check the constraint first and never
    // saw it, so the suite was green locally and red in CI.
    //
    // Asking whether the alias can be *named* is what makes that reachable on
    // every compiler: if the assertion ever comes back, this file stops
    // compiling everywhere rather than only on the Windows runner.
    static_assert(RootAliasWellFormed<dim::Area, 2>);
    static_assert(RootAliasWellFormed<dim::Volume, 3>);
    static_assert(!RootAliasWellFormed<dim::Length, 2>);
    static_assert(!RootAliasWellFormed<dim::Energy, 2>);
    static_assert(!RootAliasWellFormed<dim::Area, 0>);
    static_assert(!RootAliasWellFormed<dim::Area, -2>);

    // And the same question asked of the call site, which is the one that was
    // failing. Both have to hold: the alias is what MSVC reaches early, the
    // call is what a user actually writes.
    static_assert(SquareRootable<Area>);
    static_assert(!SquareRootable<Length>);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// The laws, spelled as type identities
// ---------------------------------------------------------------------------

TEST(UnitsDimensions, DerivedQuantitiesAreTheProductsTheyAreDefinedAs) {
    static_assert(std::is_same_v<decltype(Mass{} * Acceleration{}), Force>);
    static_assert(std::is_same_v<decltype(Length{} / Time{}), Speed>);
    static_assert(std::is_same_v<decltype(Speed{} / Time{}), Acceleration>);
    static_assert(std::is_same_v<decltype(Mass{} * Speed{}), Momentum>);
    static_assert(std::is_same_v<decltype(Force{} * Length{}), Energy>);
    static_assert(std::is_same_v<decltype(Energy{} / Time{}), Power>);
    static_assert(std::is_same_v<decltype(Force{} / Area{}), Pressure>);
    static_assert(std::is_same_v<decltype(Energy{} * Time{}), Action>);
    static_assert(std::is_same_v<decltype(Mass{} / Volume{}), Density>);
    static_assert(std::is_same_v<decltype(Energy{} / Temperature{}), HeatCapacity>);
    static_assert(std::is_same_v<decltype(Momentum{} * Length{}), AngularMomentum>);

    // Reached by a different route: kinetic energy, and gravitational
    // potential energy, both land on the same type as force times distance.
    static_assert(std::is_same_v<decltype(Mass{} * Speed{} * Speed{}), Energy>);
    SUCCEED();
}

TEST(UnitsDimensions, DimensionallyIdenticalQuantitiesAreOneType) {
    // Documented limitations, asserted so that they are a known property of
    // the design rather than a surprise. Separating these needs quantity
    // kinds; see src/Units/README.md.
    static_assert(std::is_same_v<Torque, Energy>);
    static_assert(std::is_same_v<Frequency, AngularVelocity>);
    static_assert(std::is_same_v<Entropy, HeatCapacity>);
    static_assert(std::is_same_v<Radiance, Irradiance>);
    static_assert(std::is_same_v<LuminousFlux, LuminousIntensity>);
    static_assert(std::is_same_v<RadiantPower, Power>);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// What must not compile, each beside what must
// ---------------------------------------------------------------------------

TEST(UnitsDimensions, AdditionRequiresTheSameDimension) {
    static_assert(Addable<Length, Length>);
    static_assert(!Addable<Length, Mass>);
    static_assert(!Addable<Length, Time>);
    static_assert(!Addable<Energy, Power>);

    static_assert(Subtractable<Length, Length>);
    static_assert(!Subtractable<Length, Mass>);

    // Not even to a bare number. This is the one that would otherwise let a
    // raw literal leak into a dimensioned expression unnoticed.
    static_assert(!Addable<Length, double>);
    static_assert(!Addable<double, Length>);
    SUCCEED();
}

TEST(UnitsDimensions, ComparisonRequiresTheSameDimension) {
    static_assert(Ordered<Length, Length>);
    static_assert(!Ordered<Length, Mass>);
    static_assert(!Ordered<Length, double>);

    static_assert(EqualityComparable<Mass, Mass>);
    static_assert(!EqualityComparable<Mass, Length>);
    SUCCEED();
}

TEST(UnitsDimensions, ARawNumberIsNotAQuantity) {
    static_assert(!Assignable<Length, double>);
    static_assert(Assignable<Length, Length>);

    // Construction is explicit, which is what keeps a number out of a
    // dimensioned slot without a deliberate act.
    static_assert(!std::is_convertible_v<double, Length>);
    static_assert(ConvertibleFrom<Length, double>);

    // Dimensionless converts the other way, and only that way.
    static_assert(std::is_convertible_v<Dimensionless, double>);
    static_assert(!std::is_convertible_v<double, Dimensionless>);
    SUCCEED();
}

TEST(UnitsDimensions, RootsAreConstrainedAtTheCallSite) {
    static_assert(SquareRootable<Area>);
    static_assert(!SquareRootable<Length>);
    static_assert(!SquareRootable<Energy>);
    SUCCEED();
}

TEST(UnitsDimensions, ADimensionedQuantityCannotBeAMathElementType) {
    // This is the claim the entire vector-quantity design rests on, so it is
    // asserted rather than merely written down in the README. Vector, Matrix
    // and Tensor require Numeric of their element type, and Length cannot
    // satisfy it: multiplication is not closed over it, since Length * Length
    // is an Area and an Area is not convertible back to a Length.
    static_assert(!Numeric<Length>);
    static_assert(!Numeric<Mass>);
    static_assert(!Numeric<Force3>);
    static_assert(!UsableAsAVectorElement<Length>);

    // So the composition can only go the other way round, and does.
    static_assert(UsableAsAVectorElement<double>);
    static_assert(std::is_same_v<Length3::value_type, Vec3>);

    // The one exception, and it is harmless: a dimensionless quantity closes
    // under multiplication and does satisfy Numeric. Nothing depends on that
    // either way, but leaving it unstated would make the rule above look
    // broken the first time someone tests it.
    static_assert(Numeric<Dimensionless>);
    SUCCEED();
}

TEST(UnitsDimensions, VectorQuantitiesRefuseTheOperationsVectorsDoNotHave) {
    // There is no product of two vectors that is a vector or a scalar without
    // saying which is meant, so operator* does not exist and dot and cross do.
    static_assert(!Multipliable<Length3, Length3>);
    static_assert(Dottable<Length3, Force3>);

    // Multiplying a vector quantity by a scalar quantity is fine in either
    // order, and dividing by one is fine; dividing by a vector is not.
    static_assert(Multipliable<Velocity3, Time>);
    static_assert(Multipliable<Time, Velocity3>);
    static_assert(Divisible<Length3, Time>);
    static_assert(!Divisible<Length3, Length3>);

    // No ordering on a vector. A componentwise comparison wearing the name of
    // one is how a wrong answer gets past a review.
    static_assert(!Ordered<Length3, Length3>);
    static_assert(Addable<Length3, Length3>);
    static_assert(!Addable<Length3, Force3>);
    static_assert(EqualityComparable<Length3, Length3>);

    // length() is for vectors, and says nothing about a scalar.
    static_assert(Lengthable<Length3>);
    static_assert(!Lengthable<Length>);

    // Powers are scalar-only for the same reason as ordering: raising a vector
    // to a power is not an operation, and a componentwise one would be a
    // different thing wearing the name.
    static_assert(Raisable<2, Length>);
    static_assert(!Raisable<2, Length3>);
    SUCCEED();
}

TEST(UnitsDimensions, MixedPrecisionDoesNotSilentlyCombine) {
    using Lengthf = Quantity<dim::Length, float>;
    using Timef = Quantity<dim::Time, float>;

    static_assert(Divisible<Lengthf, Timef>);
    static_assert(Addable<Lengthf, Lengthf>);

    // A float quantity and a double one are different types and stay that
    // way. Combining them would pick a precision on the caller's behalf.
    static_assert(!Addable<Lengthf, Length>);
    static_assert(!Divisible<Lengthf, Time>);
    static_assert(!Multipliable<Lengthf, Mass>);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

TEST(UnitsDimensions, AQuantityIsItsValueInStorage) {
    // Compute takes arrays of these. A dimension exists at compile time and
    // must cost nothing at run time, so an array of quantities has to be
    // memcpy-able into a GPU buffer exactly as the underlying values are.
    static_assert(sizeof(Length) == sizeof(double));
    static_assert(sizeof(Length3) == sizeof(Vec3));
    static_assert(alignof(Length3) == alignof(Vec3));

    static_assert(std::is_standard_layout_v<Length>);
    static_assert(std::is_trivially_copyable_v<Length>);
    static_assert(std::is_standard_layout_v<Length3>);
    static_assert(std::is_trivially_copyable_v<Length3>);

    // Default construction zeroes rather than leaving garbage, matching Math.
    static_assert(Length{} == Length::zero());
    SUCCEED();
}
