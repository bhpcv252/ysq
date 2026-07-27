#pragma once

#include <Math/Scalar.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace ysq {

/// The dimensional structure of a physical quantity, and the machinery that
/// makes it arithmetic.
///
/// A Quantity is a value plus the kind of thing that value is. Addition needs
/// both operands to be the same kind; multiplication produces a new kind. That
/// is the whole idea, and it is enough to make a dimensionally impossible
/// equation a compile error instead of a wrong trajectory.
///
///     const Length d = 5.0 * units::kilometre;
///     const Time   t = 2.0 * units::second;
///     const Speed  v = d / t;      // Length / Time is a Speed, by construction
///     const Speed  w = d + t;      // does not compile, and cannot be made to
///
/// Values are stored in canonical SI and nowhere else. A unit is not part of
/// the type; it is a quantity of magnitude one unit, so conversion is ordinary
/// multiplication and division and needs no API of its own.

// ---------------------------------------------------------------------------
// Dimension
// ---------------------------------------------------------------------------

/// Exponents of the seven SI base quantities, in the order the SI lists them.
///
/// All seven are present from the start even though the mechanics that lands
/// first uses three. Adding a base dimension later would re-plumb every alias
/// in the module, and the six unused exponents cost exactly nothing: they are
/// template arguments that vanish at compile time.
///
/// Exponents are `int`, not rationals. Every SI-coherent unit of physical law
/// has integer exponents; the half-integer cases come from a choice of unit
/// system (Gaussian-unit charge is M^1/2 L^3/2 T^-1) or from an analysis
/// artifact (an amplitude spectral density is a square root taken at display
/// time, over a power spectral density that is integer-dimensioned). Rational
/// exponents would put std::ratio into every diagnostic this module emits,
/// which is a poor trade for the module whose job is legible diagnostics.
///
/// Nothing outside this header pattern-matches on Dimension. It is built only
/// through dim::Dim and combined only through dim::Mul, dim::Div, dim::Raise
/// and dim::Root, so if rational exponents ever do become necessary the change
/// is confined to this file and the symbol renderer.
template <int L, int M, int T, int I, int Th, int N, int J>
struct Dimension {
    static constexpr int length = L;
    static constexpr int mass = M;
    static constexpr int time = T;
    static constexpr int current = I;
    static constexpr int temperature = Th;
    static constexpr int amount = N;
    static constexpr int luminousIntensity = J;

    /// Same seven values in SI order, for code that iterates rather than names
    /// them. Format.hpp is the only caller.
    static constexpr int exponents[7] = {L, M, T, I, Th, N, J};
};

namespace dim {

/// Every exponent defaults to zero, so a dimension names only what it uses:
/// Dim<1> is a length, Dim<0, 1> a mass, Dim<> dimensionless.
template <int L = 0, int M = 0, int T = 0, int I = 0, int Th = 0, int N = 0, int J = 0>
using Dim = Dimension<L, M, T, I, Th, N, J>;

namespace detail {

template <class D>
struct IsDimensionImpl : std::false_type {};

template <int L, int M, int T, int I, int Th, int N, int J>
struct IsDimensionImpl<Dimension<L, M, T, I, Th, N, J>> : std::true_type {};

/// Every combined dimension is instantiated through here rather than by
/// writing the arithmetic directly in the alias.
///
/// This is purely about diagnostics, and it is worth a struct. Clang prints an
/// alias template's arguments as they were written, so composing dimensions
/// inline yields `Dimension<0 + 1, 1 + 0, 0 + -2, ...>` in every error message
/// that mentions a force. Passing the sums through a non-type template
/// parameter forces them to be evaluated before the type is named, and the
/// same message reads `Dimension<1, 1, -2, ...>`. It costs nothing at compile
/// time and it is the difference between a readable error and an arithmetic
/// puzzle, in the one module whose purpose is readable errors.
template <int L, int M, int T, int I, int Th, int N, int J>
struct Canonical {
    using type = Dimension<L, M, T, I, Th, N, J>;
};

template <class A, class B>
struct MulImpl;

template <int L1, int M1, int T1, int I1, int H1, int N1, int J1, int L2, int M2,
          int T2, int I2, int H2, int N2, int J2>
struct MulImpl<Dimension<L1, M1, T1, I1, H1, N1, J1>,
               Dimension<L2, M2, T2, I2, H2, N2, J2>> {
    using type = typename Canonical<L1 + L2, M1 + M2, T1 + T2, I1 + I2, H1 + H2,
                                    N1 + N2, J1 + J2>::type;
};

template <class A, class B>
struct DivImpl;

template <int L1, int M1, int T1, int I1, int H1, int N1, int J1, int L2, int M2,
          int T2, int I2, int H2, int N2, int J2>
struct DivImpl<Dimension<L1, M1, T1, I1, H1, N1, J1>,
               Dimension<L2, M2, T2, I2, H2, N2, J2>> {
    using type = typename Canonical<L1 - L2, M1 - M2, T1 - T2, I1 - I2, H1 - H2,
                                    N1 - N2, J1 - J2>::type;
};

template <class A, int N>
struct RaiseImpl;

template <int L, int M, int T, int I, int H, int Am, int J, int N>
struct RaiseImpl<Dimension<L, M, T, I, H, Am, J>, N> {
    using type = typename Canonical<L * N, M * N, T * N, I * N, H * N, Am * N,
                                    J * N>::type;
};

/// Whether a root of this degree exists, which is the single source of truth
/// for the question. Both RootImpl below and the RootExists concept are
/// defined in terms of it, so the divisibility rule is written once.
template <class A, int N>
struct RootExistsImpl : std::false_type {};

template <int L, int M, int T, int I, int H, int Am, int J, int N>
struct RootExistsImpl<Dimension<L, M, T, I, H, Am, J>, N>
    : std::bool_constant<(N > 0) && (L % N == 0) && (M % N == 0) && (T % N == 0) &&
                         (I % N == 0) && (H % N == 0) && (Am % N == 0) &&
                         (J % N == 0)> {};

/// Taking a root is the one dimension operation that can fail, and therefore
/// the one that has to fail by substitution rather than by assertion.
///
/// The primary template has no `type`, so `Root<D, N>` for exponents that do
/// not divide is a substitution failure in the immediate context. That is what
/// lets `requires { sqrt(q); }` answer false instead of ending the translation
/// unit.
///
/// A static_assert here carrying a friendly message is the obvious
/// alternative, and it is portably wrong. **MSVC substitutes into a function
/// template's declared return type before it checks the constraint**, so
/// `sqrt`'s `-> Quantity<dim::Root<D, 2>, V>` instantiates this even when
/// RootExists is false, and a hard assertion inside it takes the whole build
/// down. Clang and GCC check the constraint first and never reach it. The rule
/// that follows is general: nothing reachable from a function's return type
/// may hard-error, however good the message would have been.
///
/// units_dimensions.cpp pins this directly rather than relying on a Windows
/// runner to notice, by asserting that naming the alias is itself ill-formed.
template <class A, int N, bool = RootExistsImpl<A, N>::value>
struct RootImpl {};

template <int L, int M, int T, int I, int H, int Am, int J, int N>
struct RootImpl<Dimension<L, M, T, I, H, Am, J>, N, true> {
    using type = typename Canonical<L / N, M / N, T / N, I / N, H / N, Am / N,
                                    J / N>::type;
};

}  // namespace detail

template <class D>
inline constexpr bool isDimension = detail::IsDimensionImpl<D>::value;

template <class D>
concept DimensionType = isDimension<D>;

using Dimensionless = Dim<>;

// The seven base quantities. Derived dimensions are declared beside the
// quantity that names them, in Length.hpp, Force.hpp and the rest.
using Length = Dim<1>;
using Mass = Dim<0, 1>;
using Time = Dim<0, 0, 1>;
using Current = Dim<0, 0, 0, 1>;
using Temperature = Dim<0, 0, 0, 0, 1>;
using Amount = Dim<0, 0, 0, 0, 0, 1>;
using LuminousIntensity = Dim<0, 0, 0, 0, 0, 0, 1>;

template <DimensionType A, DimensionType B>
using Mul = typename detail::MulImpl<A, B>::type;

template <DimensionType A, DimensionType B>
using Div = typename detail::DivImpl<A, B>::type;

template <DimensionType A, int N>
using Raise = typename detail::RaiseImpl<A, N>::type;

template <DimensionType A, int N>
using Root = typename detail::RootImpl<A, N>::type;

template <DimensionType A>
using Inverse = Div<Dimensionless, A>;

template <class D>
inline constexpr bool isDimensionless = std::same_as<D, Dimensionless>;

/// Whether Root<D, N> exists, as a constraint for the call sites that need to
/// say so. Naming it buys the diagnostic: on a compiler that checks
/// constraints before substituting the return type, an odd-dimension square
/// root is reported as "constraints not satisfied: RootExists" against the
/// call rather than as a missing member of a detail template.
///
/// Redundant with the substitution failure that Root<D, N> produces on its
/// own, deliberately. The constraint is the readable half and the substitution
/// failure is the portable half, and neither alone does both jobs.
template <class D, int N>
concept RootExists = DimensionType<D> && detail::RootExistsImpl<D, N>::value;

}  // namespace dim

// ---------------------------------------------------------------------------
// What a quantity can hold
// ---------------------------------------------------------------------------

namespace detail {

template <class V>
struct ScalarOfImpl {
    using type = V;
};

template <class T>
struct ScalarOfImpl<Vector2<T>> {
    using type = T;
};

template <class T>
struct ScalarOfImpl<Vector3<T>> {
    using type = T;
};

template <class T>
struct ScalarOfImpl<Vector4<T>> {
    using type = T;
};

template <class V>
struct IsMathVectorImpl : std::false_type {};

template <class T>
struct IsMathVectorImpl<Vector2<T>> : std::true_type {};

template <class T>
struct IsMathVectorImpl<Vector3<T>> : std::true_type {};

template <class T>
struct IsMathVectorImpl<Vector4<T>> : std::true_type {};

}  // namespace detail

/// A quantity holds either a plain number or a Math vector of them.
///
/// Vector3<Length> is not the other spelling of the same thing and never will
/// be: Length cannot satisfy Numeric, because multiplication is not closed
/// over it (Length * Length is an Area), and Numeric is what Vector, Matrix
/// and Tensor require of their element type. So the composition goes the other
/// way round, a dimension wrapped around a vector. Complex is excluded from
/// Numeric for the same structural reason; see src/Math/README.md.
template <class V>
concept QuantityScalar = std::floating_point<V>;

template <class V>
concept QuantityVector = detail::IsMathVectorImpl<V>::value &&
                         std::floating_point<typename detail::ScalarOfImpl<V>::type>;

template <class V>
concept QuantityValue = QuantityScalar<V> || QuantityVector<V>;

/// The underlying number type: V itself for a scalar, the component type for a
/// vector. This is what a tolerance, a scale factor and a magnitude are all
/// expressed in.
template <QuantityValue V>
using ScalarOf = typename detail::ScalarOfImpl<V>::type;

// ---------------------------------------------------------------------------
// Quantity
// ---------------------------------------------------------------------------

/// A value in canonical SI, tagged with what kind of thing it is.
///
/// Standard layout and trivially copyable, like the Math types, so an array of
/// them uploads to a GPU buffer with no repacking: a Quantity<D, Vec3> is
/// exactly a Vec3 in storage. units_strict_warnings.cpp pins that.
///
/// The value is private and reached through value(). It is deliberately not
/// indexable and has no size(): exposing operator[] and size() would make a
/// quantity match the generic vector formatter and streaming operator in
/// tests/support/MathApprox.hpp by accident, and an accidental match is worse
/// than an explicit accessor.
///
/// Everything here is constexpr except sqrt() and root(), for the reason
/// recorded in Math/Scalar.hpp: std::sqrt is not constexpr before C++26.
template <dim::DimensionType D, QuantityValue Value = double>
class Quantity {
public:
    using dimension = D;
    using value_type = Value;
    using scalar_type = ScalarOf<Value>;

    constexpr Quantity() = default;

    /// Always explicit, so a raw number cannot become a Length by accident.
    explicit constexpr Quantity(const Value& v) : m_value(v) {}

    /// A dimensionless quantity converts back to its number, so a ratio drops
    /// straight into std::sin, a comparison against a literal, or anything
    /// else that takes a plain double.
    ///
    /// One direction only. Making the constructor implicit for dimensionless
    /// as well reads symmetrical and is a trap: `ratio > 0.5` then has two
    /// equally good readings, converting the literal up or the quantity down,
    /// and is ambiguous. Converting down is the direction with the use cases,
    /// so it is the one that exists.
    constexpr operator Value() const noexcept
        requires(dim::isDimensionless<D>)
    {
        return m_value;
    }

    /// The raw SI value. This is the boundary: everything that cannot be
    /// expressed in quantities, the Math integrators above all, crosses here
    /// and is expected to say so.
    [[nodiscard]] constexpr const Value& value() const noexcept { return m_value; }

    /// The magnitude in a given unit, which is division spelled so it reads
    /// left to right at a call site: `d.in(units::kilometre)`. Dimension
    /// checked, so a length cannot be read in seconds.
    [[nodiscard]] constexpr Value in(const Quantity<D, scalar_type>& unit) const
        noexcept {
        return m_value / unit.value();
    }

    [[nodiscard]] static constexpr Quantity zero() noexcept { return Quantity{}; }

    constexpr Quantity& operator+=(const Quantity& other) noexcept {
        m_value += other.m_value;
        return *this;
    }

    constexpr Quantity& operator-=(const Quantity& other) noexcept {
        m_value -= other.m_value;
        return *this;
    }

    /// Scaling by a plain number only. A compound assignment cannot change the
    /// type of its left-hand side, so there is no *= that multiplies by
    /// another quantity; that operation exists, and it returns something else.
    constexpr Quantity& operator*=(scalar_type factor) noexcept {
        m_value *= factor;
        return *this;
    }

    constexpr Quantity& operator/=(scalar_type divisor) noexcept {
        m_value /= divisor;
        return *this;
    }

    [[nodiscard]] friend constexpr Quantity operator+(const Quantity& q) noexcept {
        return q;
    }

    [[nodiscard]] friend constexpr Quantity operator-(const Quantity& q) noexcept {
        return Quantity{-q.m_value};
    }

    [[nodiscard]] friend constexpr Quantity operator+(const Quantity& a,
                                                      const Quantity& b) noexcept {
        return Quantity{a.m_value + b.m_value};
    }

    [[nodiscard]] friend constexpr Quantity operator-(const Quantity& a,
                                                      const Quantity& b) noexcept {
        return Quantity{a.m_value - b.m_value};
    }

    [[nodiscard]] friend constexpr Quantity operator*(const Quantity& q,
                                                      scalar_type factor) noexcept {
        return Quantity{q.m_value * factor};
    }

    [[nodiscard]] friend constexpr Quantity operator*(scalar_type factor,
                                                      const Quantity& q) noexcept {
        return Quantity{factor * q.m_value};
    }

    [[nodiscard]] friend constexpr Quantity operator/(const Quantity& q,
                                                      scalar_type divisor) noexcept {
        return Quantity{q.m_value / divisor};
    }

    [[nodiscard]] friend constexpr bool operator==(const Quantity&,
                                                   const Quantity&) = default;

    // Ordering is scalar-valued only. There is no ordering on vectors, and a
    // componentwise one masquerading as a comparison is how a wrong answer
    // gets past a review.

    [[nodiscard]] friend constexpr bool operator<(const Quantity& a,
                                                  const Quantity& b) noexcept
        requires QuantityScalar<Value>
    {
        return a.m_value < b.m_value;
    }

    [[nodiscard]] friend constexpr bool operator>(const Quantity& a,
                                                  const Quantity& b) noexcept
        requires QuantityScalar<Value>
    {
        return b.m_value < a.m_value;
    }

    [[nodiscard]] friend constexpr bool operator<=(const Quantity& a,
                                                   const Quantity& b) noexcept
        requires QuantityScalar<Value>
    {
        return !(b.m_value < a.m_value);
    }

    [[nodiscard]] friend constexpr bool operator>=(const Quantity& a,
                                                   const Quantity& b) noexcept
        requires QuantityScalar<Value>
    {
        return !(a.m_value < b.m_value);
    }

private:
    Value m_value{};
};

/// A pure number that has been through the dimension system: the result of
/// dividing two quantities of the same kind, and what a ratio, an efficiency
/// or a Lorentz factor is.
using Dimensionless = Quantity<dim::Dimensionless>;

// ---------------------------------------------------------------------------
// Products and quotients across dimensions
// ---------------------------------------------------------------------------

namespace detail {

/// The value type of a product: the vector one if either operand is a vector,
/// the scalar one if neither is. Both being vectors is excluded by the
/// constraint on operator* itself, since there is no product of two vectors
/// that is a vector-or-scalar without saying which one is meant. dot() and
/// cross() are how that is said.
template <class V1, class V2>
using ProductValue = std::conditional_t<QuantityScalar<V1>, V2, V1>;

}  // namespace detail

template <class D1, class V1, class D2, class V2>
    requires(QuantityScalar<V1> || QuantityScalar<V2>) &&
            std::same_as<ScalarOf<V1>, ScalarOf<V2>>
[[nodiscard]] constexpr auto operator*(const Quantity<D1, V1>& a,
                                       const Quantity<D2, V2>& b) noexcept
    -> Quantity<dim::Mul<D1, D2>, detail::ProductValue<V1, V2>> {
    return Quantity<dim::Mul<D1, D2>, detail::ProductValue<V1, V2>>{a.value() *
                                                                   b.value()};
}

/// Dividing by a vector is not an operation, so the divisor is scalar-valued
/// and the result keeps the shape of the numerator.
template <class D1, class V1, class D2, class V2>
    requires QuantityScalar<V2> && std::same_as<ScalarOf<V1>, ScalarOf<V2>>
[[nodiscard]] constexpr auto operator/(const Quantity<D1, V1>& a,
                                       const Quantity<D2, V2>& b) noexcept
    -> Quantity<dim::Div<D1, D2>, V1> {
    return Quantity<dim::Div<D1, D2>, V1>{a.value() / b.value()};
}

/// A magnitude times a direction.
///
/// The direction is a bare Math vector, because a unit vector has no
/// dimension: normalized() returns one, and this is what puts it back together
/// with a magnitude. Physics produces the two halves separately more often
/// than it produces the whole, so without this every force law would have to
/// wrap its direction in a dimensionless quantity first, which reads as
/// ceremony and is exactly the friction that sends people back to raw doubles.
template <class D, class V, QuantityVector Vec>
    requires QuantityScalar<V> && std::same_as<ScalarOf<Vec>, V>
[[nodiscard]] constexpr auto operator*(const Quantity<D, V>& magnitude,
                                       const Vec& direction) noexcept
    -> Quantity<D, Vec> {
    return Quantity<D, Vec>{direction * magnitude.value()};
}

template <class D, class V, QuantityVector Vec>
    requires QuantityScalar<V> && std::same_as<ScalarOf<Vec>, V>
[[nodiscard]] constexpr auto operator*(const Vec& direction,
                                       const Quantity<D, V>& magnitude) noexcept
    -> Quantity<D, Vec> {
    return Quantity<D, Vec>{direction * magnitude.value()};
}

/// A plain number over a quantity, which is the only way to reach an inverse
/// dimension without writing the dimension out.
template <class D, class V>
    requires QuantityScalar<V>
[[nodiscard]] constexpr auto operator/(ScalarOf<V> numerator,
                                       const Quantity<D, V>& q) noexcept
    -> Quantity<dim::Inverse<D>, V> {
    return Quantity<dim::Inverse<D>, V>{numerator / q.value()};
}

// ---------------------------------------------------------------------------
// Powers and roots
// ---------------------------------------------------------------------------

namespace detail {

/// Integer power by repeated multiplication rather than std::pow, so it is
/// constexpr and so raised<2>(q) is exactly q * q rather than exp(2 log q).
template <int N, class T>
[[nodiscard]] constexpr T powInt(const T& x) noexcept {
    if constexpr (N == 0) {
        return T{1};
    } else if constexpr (N < 0) {
        return T{1} / powInt<-N>(x);
    } else {
        return x * powInt<N - 1>(x);
    }
}

}  // namespace detail

/// q raised to an integer power, positive, negative or zero.
///
/// Named raised<N> rather than pow<N> because ysq::pow already has overloads
/// for Complex and Dual; while an explicit non-type argument does resolve
/// correctly against them, a near-miss produces a wall of unrelated candidates
/// in the one module whose purpose is legible errors.
template <int N, class D, class V>
    requires QuantityScalar<V>
[[nodiscard]] constexpr auto raised(const Quantity<D, V>& q) noexcept
    -> Quantity<dim::Raise<D, N>, V> {
    return Quantity<dim::Raise<D, N>, V>{detail::powInt<N>(q.value())};
}

/// Square root, defined only where the result has integer exponents. An area
/// has a square root; a length does not, in a system with integer exponents,
/// and the constraint says so at the call site rather than deep inside an
/// alias.
template <class D, class V>
    requires QuantityScalar<V> && dim::RootExists<D, 2>
[[nodiscard]] auto sqrt(const Quantity<D, V>& q) -> Quantity<dim::Root<D, 2>, V> {
    return Quantity<dim::Root<D, 2>, V>{detail::sqrtOf(q.value())};
}

/// The general root, on the same terms as sqrt.
template <int N, class D, class V>
    requires QuantityScalar<V> && dim::RootExists<D, N>
[[nodiscard]] auto root(const Quantity<D, V>& q) -> Quantity<dim::Root<D, N>, V> {
    using std::pow;
    return Quantity<dim::Root<D, N>, V>{pow(q.value(), V{1} / static_cast<V>(N))};
}

// ---------------------------------------------------------------------------
// Vector-valued quantities
// ---------------------------------------------------------------------------

template <class D1, class D2, QuantityVector V>
[[nodiscard]] constexpr auto dot(const Quantity<D1, V>& a,
                                 const Quantity<D2, V>& b) noexcept
    -> Quantity<dim::Mul<D1, D2>, ScalarOf<V>> {
    return Quantity<dim::Mul<D1, D2>, ScalarOf<V>>{dot(a.value(), b.value())};
}

template <class D1, class D2, class T>
[[nodiscard]] constexpr auto cross(const Quantity<D1, Vector3<T>>& a,
                                   const Quantity<D2, Vector3<T>>& b) noexcept
    -> Quantity<dim::Mul<D1, D2>, Vector3<T>> {
    return Quantity<dim::Mul<D1, D2>, Vector3<T>>{cross(a.value(), b.value())};
}

/// The two-dimensional cross product is a scalar, following Math.
template <class D1, class D2, class T>
[[nodiscard]] constexpr auto cross(const Quantity<D1, Vector2<T>>& a,
                                   const Quantity<D2, Vector2<T>>& b) noexcept
    -> Quantity<dim::Mul<D1, D2>, T> {
    return Quantity<dim::Mul<D1, D2>, T>{cross(a.value(), b.value())};
}

template <class D, QuantityVector V>
[[nodiscard]] constexpr auto lengthSquared(const Quantity<D, V>& q) noexcept
    -> Quantity<dim::Raise<D, 2>, ScalarOf<V>> {
    return Quantity<dim::Raise<D, 2>, ScalarOf<V>>{lengthSquared(q.value())};
}

template <class D, QuantityVector V>
[[nodiscard]] auto length(const Quantity<D, V>& q) -> Quantity<D, ScalarOf<V>> {
    return Quantity<D, ScalarOf<V>>{length(q.value())};
}

/// A direction is dimensionless, so this returns the bare Math vector rather
/// than a quantity. Undefined at zero length: every component comes back NaN,
/// which propagates instead of quietly becoming a wrong unit vector. Follows
/// Math exactly, including tryNormalized being the form to use where the input
/// can legitimately be zero.
template <class D, QuantityVector V>
[[nodiscard]] V normalized(const Quantity<D, V>& q) {
    return normalized(q.value());
}

template <class D, QuantityVector V>
[[nodiscard]] std::optional<V> tryNormalized(const Quantity<D, V>& q) {
    return tryNormalized(q.value());
}

template <class D, QuantityVector V>
[[nodiscard]] constexpr auto distanceSquared(const Quantity<D, V>& a,
                                             const Quantity<D, V>& b) noexcept
    -> Quantity<dim::Raise<D, 2>, ScalarOf<V>> {
    return lengthSquared(a - b);
}

template <class D, QuantityVector V>
[[nodiscard]] auto distance(const Quantity<D, V>& a, const Quantity<D, V>& b)
    -> Quantity<D, ScalarOf<V>> {
    return length(a - b);
}

// ---------------------------------------------------------------------------
// The Math free functions that do not reach quantities on their own
// ---------------------------------------------------------------------------

// lerp, clamp, min, max and abs are all constrained on Numeric in Math, which
// Quantity deliberately does not satisfy. Their absence is felt on the first
// interpolated position, so they are provided here rather than left to every
// caller to unwrap and rewrap by hand.

template <class D, QuantityValue V>
[[nodiscard]] constexpr Quantity<D, V> lerp(const Quantity<D, V>& a,
                                            const Quantity<D, V>& b,
                                            ScalarOf<V> t) noexcept {
    return a * (ScalarOf<V>{1} - t) + b * t;
}

template <class D, class V>
    requires QuantityScalar<V>
[[nodiscard]] constexpr Quantity<D, V> clamp(const Quantity<D, V>& value,
                                             const Quantity<D, V>& lo,
                                             const Quantity<D, V>& hi) noexcept {
    return Quantity<D, V>{clamp(value.value(), lo.value(), hi.value())};
}

template <class D, class V>
    requires QuantityScalar<V>
[[nodiscard]] constexpr Quantity<D, V> min(const Quantity<D, V>& a,
                                           const Quantity<D, V>& b) noexcept {
    return (b < a) ? b : a;
}

template <class D, class V>
    requires QuantityScalar<V>
[[nodiscard]] constexpr Quantity<D, V> max(const Quantity<D, V>& a,
                                           const Quantity<D, V>& b) noexcept {
    return (a < b) ? b : a;
}

/// Componentwise for a vector-valued quantity, matching Math's abs.
template <class D, QuantityValue V>
[[nodiscard]] Quantity<D, V> abs(const Quantity<D, V>& q) {
    if constexpr (QuantityScalar<V>) {
        return Quantity<D, V>{detail::absOf(q.value())};
    } else {
        return Quantity<D, V>{abs(q.value())};
    }
}

/// -1, 0 or +1 as a plain number, not as a quantity.
///
/// A sign is dimensionless. Returning a Quantity here would make
/// `sign(Length{-3})` mean minus one metre, which is not what anyone asking
/// for a sign wants and would then multiply back into an expression carrying a
/// length it should not have. Same reasoning as normalized() handing back a
/// bare Vec3: a direction has no dimension either.
template <class D, class V>
    requires QuantityScalar<V>
[[nodiscard]] constexpr V sign(const Quantity<D, V>& q) noexcept {
    return sign(q.value());
}

// ---------------------------------------------------------------------------
// Approximate comparison
// ---------------------------------------------------------------------------

/// Mixed relative and absolute comparison, delegating to Math's rule.
///
/// The absolute tolerance is itself a quantity rather than a bare number.
/// "Within a millimetre" is a length; writing it as 1e-3 would leave the
/// caller to remember that the number means metres, which is the exact class
/// of mistake this module exists to remove. The relative tolerance stays a
/// plain number because a ratio is dimensionless by construction.
template <class D, class V>
    requires QuantityScalar<V>
[[nodiscard]] constexpr bool approxEqual(
    const Quantity<D, V>& a, const Quantity<D, V>& b, V relTol = kDefaultRelTol<V>,
    const Quantity<D, V>& absTol = Quantity<D, V>{kDefaultAbsTol<V>}) noexcept {
    return approxEqual(a.value(), b.value(), relTol, absTol.value());
}

template <class D, class V>
    requires QuantityScalar<V>
[[nodiscard]] constexpr bool isNearZero(
    const Quantity<D, V>& q,
    const Quantity<D, V>& absTol = Quantity<D, V>{kDefaultAbsTol<V>}) noexcept {
    return isNearZero(q.value(), absTol.value());
}

}  // namespace ysq
