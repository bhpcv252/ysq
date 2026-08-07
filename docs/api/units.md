# Units API reference

Every public type, dimension alias, unit constant and literal in `Units`.
Start with [docs/units.md](../units.md) for the idea;
[src/Units/README.md](../../src/Units/README.md) is the authoritative source
for conversion-factor exactness, sourcing, and the "what a dimension cannot
tell you" caveats; read it before trusting a constant transcribed here.
`Units` depends on `Math`, not `Core`.

## `Units/Unit.hpp`

The dimension system and the `Quantity` template everything else is built
from.

```cpp
template <int L, int M, int T, int I, int Th, int N, int J> struct Dimension { /* exponents */ };

namespace dim {
    template <int L=0, int M=0, int T=0, int I=0, int Th=0, int N=0, int J=0>
    using Dim = Dimension<L, M, T, I, Th, N, J>;

    using Dimensionless = Dim<>;
    using Length = Dim<1>; using Mass = Dim<0,1>; using Time = Dim<0,0,1>;
    using Current = Dim<0,0,0,1>; using Temperature = Dim<0,0,0,0,1>;
    using Amount = Dim<0,0,0,0,0,1>; using LuminousIntensity = Dim<0,0,0,0,0,0,1>;

    template <DimensionType A, DimensionType B> using Mul = /* exponents add */;
    template <DimensionType A, DimensionType B> using Div = /* exponents subtract */;
    template <DimensionType A, int N> using Raise = /* exponents * N */;
    template <DimensionType A, int N> using Root = /* exponents / N, absent if it doesn't divide */;
    template <DimensionType A> using Inverse = Div<Dimensionless, A>;

    template <class D, int N> concept RootExists = /* Root<D, N> is well-formed */;
}
```

Exponents are `int`, not rationals: every SI-coherent physical law has
integer exponents, and rational ones would put `std::ratio` into every
diagnostic. `dim::Root<D, N>` has no `type` when the exponents don't divide
by `N`: a substitution failure, so `requires { sqrt(q); }` answers `false`
instead of hard-erroring.

```cpp
template <dim::DimensionType D, QuantityValue Value = double>
class Quantity {
public:
    using dimension = D;
    using value_type = Value;
    using scalar_type = ScalarOf<Value>;   // Value itself, or a vector's component type

    explicit constexpr Quantity(const Value& v);           // always explicit
    constexpr operator Value() const noexcept requires(dim::isDimensionless<D>);

    constexpr const Value& value() const noexcept;          // the boundary: raw SI value
    constexpr Value in(const Quantity<D, scalar_type>& unit) const noexcept;  // d.in(units::kilometre)

    static constexpr Quantity zero() noexcept;
    // += -= (same-dimension), *= /= (by a plain scalar_type)
    // + - (same dimension), * / (see the free functions below), ==
    // < > <= >= only when Value is scalar (no ordering on a vector quantity)
};

using Dimensionless = Quantity<dim::Dimensionless>;
```

| Member | Description |
| --- | --- |
| Constructor | Always `explicit`: a raw `double` never becomes a `Length` by accident. |
| `operator Value()` | Implicit, **one direction only**, and only for a dimensionless quantity: a ratio converts straight to its number so it drops into `std::sin` or a comparison against a literal. |
| `value()` | The boundary. Everything that can't be expressed in quantities (the `Math` integrators, above all) crosses here. |
| `in(unit)` | The magnitude in a given unit, dimension-checked: `d.in(units::kilometre)` reads left to right; a length can't be read `.in(units::second)`. |

Free functions (all in namespace `ysq`, argument-dependent lookup finds
them):

| Function | Result dimension | Notes |
| --- | --- | --- |
| `a * b` (two quantities) | `dim::Mul<D1, D2>` | At least one operand must be scalar-valued; two vector quantities have no product that's unambiguously vector-or-scalar, so use `dot`/`cross` instead. |
| `a / b` | `dim::Div<D1, D2>` | Divisor must be scalar-valued. |
| `magnitude * direction` | same as `magnitude`, vector-valued | `direction` is a bare `Vec3` (dimensionless): puts a magnitude and a direction back together. |
| `number / quantity` | `dim::Inverse<D>` | The way to reach an inverse dimension without spelling it out. |
| `raised<N>(q)` | `dim::Raise<D, N>` | Integer power, positive/negative/zero, by repeated multiplication (`constexpr`, exact); named `raised` rather than `pow` to avoid colliding with `Math`'s `Complex`/`Dual` overloads. |
| `sqrt(q)` | `dim::Root<D, 2>` | Requires `dim::RootExists<D, 2>`: an `Area` has a square root, a `Length` does not. |
| `root<N>(q)` | `dim::Root<D, N>` | The general form; requires `dim::RootExists<D, N>`. |
| `dot(a, b)` | `dim::Mul<D1, D2>`, scalar | |
| `cross(a, b)` | `dim::Mul<D1, D2>`, vector (`Vector3`) or scalar (`Vector2`, matching `Math`) | |
| `lengthSquared(q)` / `length(q)` | squared / unchanged, scalar | |
| `normalized(q)` / `tryNormalized(q)` | none, returns a bare `Vec2/3/4` | A direction is dimensionless. `tryNormalized` is `nullopt` for zero/non-finite/overflowing; `normalized` yields NaN. |
| `distanceSquared(a, b)` / `distance(a, b)` | as `lengthSquared`/`length` of `a - b` | |
| `lerp(a, b, t)` | unchanged | `t` is a plain `scalar_type`. |
| `clamp`/`min`/`max`/`abs`/`sign(q)` | unchanged, except `sign` which returns a plain number | `sign` returns a bare number, not a quantity: `sign(Length{-3})` is not "minus one metre." |
| `approxEqual(a, b, relTol, absTol)` | `bool` | `absTol` is itself a **quantity** ("within a millimetre" is a length), `relTol` a plain number. |
| `isNearZero(q, absTol)` | `bool` | |

```cpp
const Length d = 5.0 * units::kilometre;
const Time t = 2.0 * units::second;
const Speed v = d / t;              // Length / Time is a Speed, by construction
// const Speed w = d + t;            // does not compile

const Force magnitude = gravitationalConstant * m1 * m2 / lengthSquared(r);
const Vec3 direction = normalized(r);
const Force3 gravity = -(magnitude * direction);

const double inKm = d.in(units::kilometre);
```

**Vector quantities wrap the vector, not the other way round.**
`Vector3<Length>` cannot exist: `Length` doesn't satisfy `Math`'s `Numeric`
concept, since `Length * Length` is an `Area`, not a `Length` (multiplication
isn't closed over it). So the composition goes the other way:
`Quantity<dim::Length, Vec3>`, aliased `Length3`.

**Crossing to the `Math` integrators is exact but manual.** A dimensioned
state satisfies `OdeState` (only needs a vector space) but not `OdeSystem`
(the derivative of a position is a velocity, a different type), so
`Physics` sets up and verifies in quantities and crosses to raw values at
the integrator boundary:

```cpp
const PhaseState<Vec3> initial{position.value(), velocity.value()};
const PhaseState<Vec3> result = integrate(stepper, acceleration, initial, 0.0,
                                          duration.value(), step);
const Length3 final{result.position};   // exact: a change of static type, nothing else
```

## Per-quantity headers

Each header below declares a `dim::` alias, the `Quantity` alias(es), unit
constants in `units::`, and (where natural) `ysq::literals` suffixes. Unit
constants are always `double`-valued regardless of the `Quantity`'s own
value type: a conversion factor is a definition, and a `float` shouldn't
spell it with a degraded copy of an exact number.

| Header | Dimension(s) | Quantity aliases | Representative units | Literals |
| --- | --- | --- | --- | --- |
| `Length.hpp` | `Length`, `Area`, `Volume`, `WaveNumber` | `Length`, `Length2/3/4`, `Area`, `Volume`, `WaveNumber` | `metre`, `kilometre`, `centimetre`, `millimetre`, `micrometre`, `nanometre`, `angstrom`, `astronomicalUnit`, `parsec`, `lightYear`, `solarRadius`, `squareMetre`, `cubicMetre` | `_m _km _cm _mm _au _pc _ly` |
| `Mass.hpp` | `Mass`, `Density`, `SurfaceDensity`, `LinearDensity` | `Mass`, `Density`, `SurfaceDensity`, `LinearDensity` | `kilogram`, `gram`, `tonne`, `atomicMassUnit`, `solarMass`, `earthMass`, `kilogramPerCubicMetre` | `_kg _g _Msun _Mearth` |
| `Time.hpp` | `Time`, `Frequency` (= `AngularVelocity`) | `Time`, `Frequency`, `AngularVelocity` | `second`, `millisecond`, `microsecond`, `nanosecond`, `minute`, `hour`, `day`, `week`, `month` (a twelfth of the Julian year, not a calendar month), `year` (Julian), `megayear`, `gigayear`, `hertz` | `_s _ms _min _h _day _yr _Myr _Gyr _Hz` (collide with `std::chrono_literals`, deliberately; see below) |
| `Velocity.hpp` | `Velocity` | `Speed` (scalar), `Velocity2/3/4` (vector) | `metrePerSecond`, `kilometrePerSecond`, `kilometrePerHour`, `speedOfLight` | `_mps _kmps` |
| `Acceleration.hpp` | `Acceleration`, `Jerk` | `Acceleration`, `Acceleration2/3/4`, `Jerk`, `Jerk3` | `metrePerSecondSquared`, `standardGravity` | `_mps2 _g0` |
| `Force.hpp` | `Force`, `Momentum`, `AngularMomentum`, `Pressure`, `Torque` | `Force`, `Force2/3/4`, `Momentum`, `Momentum2/3/4`, `AngularMomentum`, `AngularMomentum3`, `Torque`, `Torque3`, `Pressure` | `newton`, `dyne`, `pascal`, `bar`, `atmosphere`, `kilogramMetrePerSecond` | `_N _Pa _bar` |
| `Energy.hpp` | `Energy` (= `Torque`), `Power`, `Action`, `SpecificEnergy` | `Energy`, `Power`, `Action`, `SpecificEnergy` | `joule`, `erg`, `electronvolt`/`kilo`/`mega`/`gigaelectronvolt`, `watt` | `_J _eV _MeV _W` |
| `Temperature.hpp` | `Temperature`, `HeatCapacity` (= `Entropy`) | `Temperature`, `HeatCapacity`, `Entropy` | `kelvin`, `joulePerKelvin` | `_K` (Celsius/Fahrenheit are functions, not constants; see below) |
| `Luminosity.hpp` | `RadiantPower` (= `Power`), `Irradiance` (= `Radiance`), `LuminousFlux` (= `LuminousIntensity`), `Illuminance` | `RadiantPower`, `Irradiance`, `Radiance`, `LuminousIntensity`, `LuminousFlux`, `Illuminance` | `solarLuminosity`, `candela`, `lumen`, `lux` | `_Lsun _cd` |
| `Electromagnetism.hpp` | `ElectricField`, `MagneticFluxDensity` | `ElectricField`, `ElectricField3`, `MagneticFluxDensity`, `MagneticFluxDensity3` | `voltPerMetre`, `tesla`, `gauss` | none |
| `Constants.hpp` | `ElectricCharge`, `GravitationalParameter`, `InverseAmount`, `LuminousEfficacy` | `ElectricCharge`, `GravitationalParameter`, `LuminousEfficacy` | see `constants::` below | none |

```cpp
using namespace ysq::literals;
const Length distance = 5.0_km;
const Time flightTime = 3.5_h;
const Mass probe = 1200.0_kg;
```

`Time.hpp`'s `_s`, `_min`, `_h`, `_day`, `_yr` collide on purpose with
`std::chrono_literals`'s `_s`/`_min`/`_h`/`_d`/`_y`. Opening both namespaces
unqualified in the same scope is an ambiguity **compile error**, loud and
quick to fix, which was judged the better trade over disfiguring the
spelling. Engine headers never open `ysq::literals` themselves.

**Temperature has no unit constants for Celsius/Fahrenheit**, because they're
affine (zero Celsius isn't zero kelvin), not scaled, so there is no
multiplicative factor to make a constant out of:

```cpp
constexpr Temperature fromCelsius(double degrees) noexcept;
constexpr double toCelsius(Temperature) noexcept;
constexpr Temperature fromFahrenheit(double degrees) noexcept;
constexpr double toFahrenheit(Temperature) noexcept;
```

### `constants::`: the seven constants that define the SI

From `Constants.hpp`. These are definitional, not just fundamental: the
Newtonian gravitational constant `G` is deliberately **not** here (it's
measured, with ~2e-5 relative uncertainty, and belongs with the gravity law
that uses it in `Physics/Gravity`).

```cpp
namespace constants {
    inline constexpr Frequency caesiumHyperfineFrequency{9192631770.0};   // fixes the second
    inline constexpr Speed speedOfLight{299792458.0};                     // fixes the metre
    inline constexpr Action planckConstant{6.62607015e-34};               // fixes the kilogram
    inline constexpr ElectricCharge elementaryCharge{1.602176634e-19};    // fixes the ampere
    inline constexpr HeatCapacity boltzmannConstant{1.380649e-23};        // fixes the kelvin
    inline constexpr Quantity<dim::InverseAmount> avogadroConstant{6.02214076e23};  // fixes the mole
    inline constexpr LuminousEfficacy luminousEfficacy{683.0};            // fixes the candela

    inline constexpr Action reducedPlanckConstant = /* planckConstant / (2*pi), computed */;

    // IAU 2015 B3 nominal mass parameters: exact by convention, unlike the masses in kilograms
    inline constexpr GravitationalParameter nominalSolarMassParameter{1.32712440e20};
    inline constexpr GravitationalParameter nominalEarthMassParameter{3.986004e14};
}
```

**Integrate orbits with `constants::nominalSolarMassParameter` (GM), not
with `units::solarMass` converted back to kilograms.** The IAU fixes the GM
*parameter* exactly; recovering a mass in kilograms means dividing by a
measured `G` (relative uncertainty ~2.2e-5), which throws away four
significant figures for nothing. `units::solarMass` exists because
applications sometimes need kilograms, and it is the lossy form.

### What a dimension cannot tell you

Several distinct physical quantities share a dimension and are therefore
the *same type* here: `Torque`/`Energy`, `Frequency`/`AngularVelocity`,
`Entropy`/`HeatCapacity`, `Radiance`/`Irradiance`,
`LuminousFlux`/`LuminousIntensity`. Nothing stops handing one to code that
expects the other; both aliases exist so code can at least say which one is
meant. The formatter (below) reflects this honestly: it never guesses a
named symbol like `J`, only base-unit powers.

## `Units/Format.hpp`

`std::formatter` for every `Quantity`, printing **base-unit powers, never
named derived symbols**: an energy prints `m^2 kg s^-2`, not `J`, because
`Energy` and `Torque` share that dimension and a formatter that guessed
would be wrong half the time.

```cpp
#include <Units/Format.hpp>

ysq::logging::info("v = {:.3f}", speed);      // v = 7800.000 m s^-1
ysq::logging::info("r = {:.2e}", position);   // r = (1.50e+11, 0.00e+00, 0.00e+00) m
```

Factors appear in SI order (`m kg s A K mol cd`) with signed exponents; the
format spec forwards to the underlying value exactly as `Math/Format.hpp`
does for a vector.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/units)
and let us know.
