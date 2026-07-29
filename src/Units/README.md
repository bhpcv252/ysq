# Units

The dimensional structure of physical quantities, and the SI as the coordinate
system on it. This module knows what kind of thing a number is. It knows
nothing about how that thing behaves.

Every law in `Physics` is stated in these types, so a dimensionally impossible
equation is a compile error rather than a wrong trajectory.

**Target:** `ysq::Units` (INTERFACE, header-only)
**Depends on:** `Math`. Not `Core`: a dimension needs no logger.

## Contents

| Header                     | Purpose                                                       |
| -------------------------- | ------------------------------------------------------------- |
| `Units/Unit.hpp`           | `Dimension`, `Quantity`, and the whole algebra                 |
| `Units/Constants.hpp`      | The seven constants that define the SI, and the IAU parameters |
| `Units/Length.hpp`         | Length, area, volume, wave number                              |
| `Units/Mass.hpp`           | Mass and the three densities                                   |
| `Units/Time.hpp`           | Time, frequency, angular velocity                              |
| `Units/Velocity.hpp`       | Speed and velocity                                             |
| `Units/Acceleration.hpp`   | Acceleration and jerk                                          |
| `Units/Force.hpp`          | Force, momentum, angular momentum, torque, pressure            |
| `Units/Energy.hpp`         | Energy, power, action, specific energy                         |
| `Units/Temperature.hpp`    | Temperature, heat capacity, entropy                            |
| `Units/Luminosity.hpp`     | Radiometry and photometry                                      |
| `Units/Electromagnetism.hpp` | Electric field, magnetic flux density                        |
| `Units/Format.hpp`         | `std::formatter` for every quantity                            |

Conversion factors, their sources and their exactness are in
[Conversion factors](#conversion-factors) below.

## The shape of it

```cpp
const Length d = 5.0 * units::kilometre;
const Time   t = 2.0 * units::second;
const Speed  v = d / t;      // Length / Time is a Speed, by construction
const Speed  w = d + t;      // does not compile, and cannot be made to
```

A dimension is `Dimension<L, M, T, I, Th, N, J>`, the exponents of the seven SI
base quantities. A quantity is a value tagged with one. Addition needs both
operands to be the same kind; multiplication produces a new kind. That is the
whole idea, and it is enough.

## What earns a name

`Units` names a quantity when a law stated in `Physics` will be written in it.
Not a catalogue of everything nameable.

That is why `Pressure` and `Entropy` are here before `Thermodynamics` is, and
why the electromagnetic family waited for `Electromagnetism` to land before
`ElectricField` and `MagneticFluxDensity` were added, in `Electromagnetism.hpp`:
the ampere was already in the algebra (`ElectricCharge` lives in
`Constants.hpp`, needed to state the SI itself), so the addition was purely
additive once the law that needed them existed, rather than guessing at an
interface ahead of one.

## Vector quantities wrap the vector, not the other way round

`Vector3<Length>` cannot exist, and this is structural rather than an
oversight. `Numeric` is what `Vector`, `Matrix` and `Tensor` require of their
element type, and `Length` cannot satisfy it, because multiplication is not
closed over it: `Length * Length` is an `Area`. `Complex` is excluded from
`Numeric` for the same kind of reason, and `src/Math/README.md` records that
one.

So the dimension sits outside and the vector inside, `Quantity<dim::Length,
Vec3>`, aliased `Length3`. Every vector operation is then given a dimensional
meaning:

| Operation           | Dimension of the result             |
| ------------------- | ----------------------------------- |
| `dot(a, b)`         | product, scalar-valued              |
| `cross(a, b)`       | product, vector-valued              |
| `length(v)`         | unchanged, scalar-valued            |
| `lengthSquared(v)`  | squared, scalar-valued              |
| `normalized(v)`     | none: a plain `Vec3`                |

A direction is dimensionless, so `normalized` returns the bare Math vector. The
operation that puts a magnitude and a direction back together is
`magnitude * direction`, which is how a force law is actually written:

```cpp
const Force  magnitude = gravitationalConstant * m1 * m2 / lengthSquared(r);
const Vec3   direction = normalized(r);
const Force3 gravity   = -(magnitude * direction);
```

There is no `operator*` between two vector quantities. No product of two
vectors is a vector or a scalar without saying which is meant, and `dot` and
`cross` are how that is said.

## Storage is SI, and a unit is just a quantity

A unit is not part of a type. `units::kilometre` is a `Length` whose value is
1000, so converting in is multiplication and converting out is division:

```cpp
const Length d    = 5.0 * units::kilometre;
const double inKm = d / units::kilometre;   // dimensionless, converts to double
const double same = d.in(units::kilometre); // the same division, read left to right
```

There is no second mechanism, so there is no second mechanism to get wrong. A
`Length` built from astronomical units and one built from metres are the same
type and the same bits.

Unit constants are `double`-valued even though `Quantity` is templated on its
value type. A conversion factor is a definition rather than a measurement, and
a definition rendered at `float` precision has been damaged before it is used.
Float quantities work; they just do not get to spell their magnitude with a
degraded copy of an exact number.

Literals live in `ysq::literals` and use the natural SI suffixes, which means
`_s`, `_min`, `_h` and `_y` collide with `std::chrono_literals`. Opening both
namespaces unqualified in one scope is an ambiguity error, which is loud and
takes seconds to fix. That is a better trade than disfiguring the spelling, and
it is safe to make because the failure is a compile error rather than a wrong
number. Engine headers never open the namespace.

## Integer exponents

Every SI-coherent unit of physical law has integer exponents. The half-integer
cases come from a choice of unit system (Gaussian-unit charge is
`M^1/2 L^3/2 T^-1`) or from an analysis artifact (an amplitude spectral density
is a square root taken at display time, over a power spectral density that is
integer-dimensioned). Rational exponents would put `std::ratio` into every
diagnostic this module emits, which is a poor trade for the module whose job is
legible diagnostics.

`sqrt` is therefore constrained to dimensions whose exponents are all even, and
`root<N>` to those divisible by N.

Taking a root is the only dimension operation that can fail, and it fails
**twice over, deliberately**. `dim::Root<D, N>` has no `type` when the
exponents do not divide, so naming it is a substitution failure; and `sqrt` and
`root` additionally carry a `dim::RootExists` constraint. The constraint is the
readable half, giving "constraints not satisfied" against the call site. The
substitution failure is the portable half, and it is not optional:

> **MSVC substitutes into a function template's declared return type before it
> checks the constraint.** Clang and GCC check the constraint first.

`sqrt` returns `Quantity<dim::Root<D, 2>, V>`, so MSVC instantiates `Root` even
when the constraint is unsatisfied. `Root` originally carried a `static_assert`
with a friendly message, which meant that merely *asking* whether
`sqrt(Length)` compiles ended the translation unit. The suite was green on two
compilers and red on the third. The general rule, worth more than the message
was: **nothing reachable from a function's return type may hard-error.**

`units_dimensions.cpp` pins it on every compiler by asserting that naming
`dim::Root<dim::Length, 2>` is itself ill-formed, so a reintroduced assertion
breaks the build everywhere rather than only on a Windows runner.

Nothing outside `Unit.hpp` pattern-matches on `Dimension`. It is built only
through `dim::Dim` and combined only through `dim::Mul`, `dim::Div`,
`dim::Raise` and `dim::Root`. If rational exponents ever do become necessary,
the change is confined to that file and the symbol renderer, and no alias or
call site moves. Adopting Gaussian-unit electromagnetism is the one thing that
would force it.

## Definitional constants are here; measured ones are not

Since the 2019 redefinition the SI is not a set of artefacts but a set of seven
exactly fixed constants of nature, and every base unit falls out of them. That
is why they live in `Units`: they are not facts the simulation discovers, they
are the definition of the vocabulary it speaks. `Constants.hpp` has all seven,
plus the reduced Planck constant, which is exact for the same reason.

The line is drawn at definitional, not at fundamental. The Newtonian constant
of gravitation is as fundamental as anything in that file and is deliberately
absent: it is measured, it has a relative uncertainty around 2e-5, and it
parameterizes a specific interaction. It belongs with the gravity that uses it.

That has one visible consequence. What the IAU fixes exactly is the mass
*parameter* GM, not the mass, so `constants::nominalSolarMassParameter` is
exact and `units::solarMass` is not: recovering kilograms means dividing by a
measured G, which throws away four significant figures for nothing. Integrate
orbits with GM. `units::solarMass` exists because applications ask for
kilograms, and it is the lossy form; see [Conversion factors](#conversion-factors)
below for the G used.

## Failure is reported, not invented

Following `Math` exactly: `tryX` returns `std::optional`, plain `X` propagates
NaN.

- `tryNormalized` is nullopt for a zero or non-finite vector quantity;
  `normalized` yields NaN.
- A NaN in a quantity comes back as a NaN, and comparisons against it are
  false, including `approxEqual` against itself.

`approxEqual` takes its absolute tolerance as a quantity rather than a bare
number. "Within a millimetre" is a length, and writing it as `1e-3` would leave
the caller to remember which unit the number is in, which is the exact class of
mistake this module exists to remove. The relative tolerance stays a plain
number, because a ratio is dimensionless by construction.

## Nothing costs anything at run time

A `Quantity` is its value in storage. `sizeof(Length) == sizeof(double)` and
`sizeof(Length3) == sizeof(Vec3)`, both standard layout and trivially copyable,
so an array of quantities uploads to a GPU buffer with no repacking, exactly as
an array of the underlying Math values does. `units_strict_warnings.cpp` pins
that, including for a dimension with all seven exponents non-zero.

Everything is `constexpr` except `sqrt` and `root`, for the reason recorded in
`Math/Scalar.hpp`: `std::sqrt` is not `constexpr` before C++26. An initial
condition, a unit conversion and a derived constant can therefore all be
constant expressions, and nothing in this module runs before `main`.

## What a dimension cannot tell you

These are properties of the design, not defects to be fixed later. Each is
asserted in `units_dimensions.cpp` so it stays a known property rather than
becoming a surprise.

| These share a dimension           | and are            |
| --------------------------------- | ------------------ |
| Torque and energy                 | `M L^2 T^-2`       |
| Frequency and angular velocity    | `T^-1`             |
| Entropy and heat capacity         | `M L^2 T^-2 Th^-1` |
| Radiance and irradiance           | steradian is dimensionless |
| Luminous flux and luminous intensity | likewise        |

Separating them needs **quantity kinds**: a tag alongside the dimension. That
is deliberately not in this module. A kind does not compose under
multiplication without an explicit lattice saying what `Mass × Acceleration`
yields, and a lattice that is subtly wrong is worse than no lattice at all. If
it ever lands it is a layer on top, not a change to `Quantity`.

The formatter follows the same rule and prints base-unit powers only. An energy
renders as `m^2 kg s^-2` rather than `J`, and so does a torque, because that is
the truth. Printing `J` would assert a distinction the type system does not
carry and would be wrong half the time it mattered.

**Temperature is kelvin only.** Celsius and Fahrenheit are affine rather than
scaled, so they cannot be unit constants: a unit constant works because
conversion is multiplication, and zero Celsius is not zero kelvin. They are
named functions instead. Absolute temperature and temperature interval are not
distinguished either, for the same reason as kinds.

## Quantities do not go into the Math integrators

`Rk4Stepper` declares `State m_k1{}`, so the derivative is required to be the
same type as the state. For a position that derivative is a velocity, which is
a different type by construction. The step size has the same problem: `Scalar`
is `StateScalarT<State>`, taken from the state's own value type, so it is a
bare `double` where the physics says it is a `Time`.

This is structural, not a missing overload. A dimensioned state does satisfy
`OdeState`, since that only asks for a vector space; what fails is
`OdeSystem`, and it fails in the right place.

So `Physics` sets up and verifies in quantities and crosses to raw values at
the integrator:

```cpp
const PhaseState<Vec3> initial{position.value(), velocity.value()};
const PhaseState<Vec3> final = integrate(stepper, acceleration, initial, 0.0,
                                         duration.value(), step);
const Length3 result{final.position};
```

The crossing is exact: `Length3{q.value()} == q` bit for bit, since it is a
change of static type and nothing else. `units_kinematics.cpp` asserts all of
this, including the two concept checks, so the boundary is documented by
something that fails when it moves.

A dimension-aware stepper is a later decision to be made deliberately. Nothing
here forecloses it.

## Warnings

`ysq::Units` links no warning flags, for the reason in
`src/Units/CMakeLists.txt` and in `src/Math/README.md`: on an INTERFACE target
they propagate to every consumer's own sources, and the root `README.md`'s
Warnings section rules that out for `Renderer`, `UI` and `Applications`.

The strict set is applied instead in `tests/smoke/units_strict_warnings.cpp`,
which includes every header here, instantiates `Quantity` for `float` and
`double` across all three vector widths, and calls every free function. `float`
is not optional coverage: `-Wdouble-promotion` only has anything to say below
double precision, so what the `float` half catches is a plain `2.0` written
inside a template where `T{2}` was meant. Injecting one into `Quantity`'s
`operator*` does make that file fail to compile, which is how the claim was
checked rather than assumed.

## Tests

Five unit files, one integration file, the smoke check above, and six
compile-failure targets.

`units_dimensions.cpp` is the one that decides whether the module is correct,
and its negative cases carry as much weight as its positive ones. **Every
negative check goes through a named concept.** A bare
`static_assert(!requires(Length l, Mass m) { l + m; })` at namespace scope is
not portable: outside a template there is no substitution, so Clang reports the
invalid requirement as a hard error rather than evaluating it to false. GCC is
laxer, so writing it the obvious way passes locally and fails in CI.

`tests/compile_fail/` builds six targets that must not compile, as CTest tests
marked `WILL_FAIL`. They are the second line rather than the first, because
`WILL_FAIL` only checks for a nonzero exit and a source with a typo in it also
"passes". Each is paired with its positive form in `units_dimensions.cpp`, and
that pairing is the actual guarantee. They can be turned off with
`-DYSQ_BUILD_COMPILE_FAIL_TESTS=OFF`.

## Conversion factors

Their sources, and which of them are exact.

### Why exactness is tracked

A factor that is exact should compare equal after a round trip, and
`units_conversions.cpp` asserts equality wherever equality holds rather than
reaching for a tolerance everywhere. That is what would catch a factor quietly
rounded when someone retypes it: a tolerance-based test passes happily on a
value that has lost three digits.

Three separate things are called "exact" below and they are not the same:

- **Defined.** The number is a definition and cannot be measured. The metre per
  second of light, the second per caesium transition, the astronomical unit.
- **Exactly representable.** The defined value also happens to fit a `double`
  with no rounding, so the C++ literal is the number itself.
- **Conventional.** Fixed by agreement so that published results do not shift
  when nature is remeasured, but not a definition of a unit. The IAU nominal
  solar values are these.

A value can be defined without being exactly representable. The electronvolt is
defined exactly and its decimal expansion is not a binary fraction, so the
stored `double` is the nearest one to it.

### The seven defining constants of the SI

Since the 2019 redefinition these are fixed by fiat and every base unit is
derived from them. All are defined; all are stored as the nearest `double`.
Source: BIPM, *The International System of Units*, 9th edition, 2019.

| Constant | Symbol | Value | Fixes |
| --- | --- | --- | --- |
| Caesium hyperfine frequency | Δν_Cs | 9 192 631 770 Hz | the second |
| Speed of light in vacuum | c | 299 792 458 m/s | the metre |
| Planck constant | h | 6.626 070 15e-34 J s | the kilogram |
| Elementary charge | e | 1.602 176 634e-19 C | the ampere |
| Boltzmann constant | k | 1.380 649e-23 J/K | the kelvin |
| Avogadro constant | N_A | 6.022 140 76e23 mol⁻¹ | the mole |
| Luminous efficacy at 540 THz | K_cd | 683 lm/W | the candela |

Δν_Cs, c, e and N_A are integers small enough to be exactly representable. h, k
and the others are not, and are stored as the nearest `double`.

The reduced Planck constant ħ = h / 2π is exact by the same argument, being a
defined value divided by a mathematical constant, and is computed rather than
typed so the two cannot drift apart.

### Length

| Unit | Value in metres | Status |
| --- | --- | --- |
| kilometre, centimetre, millimetre, micrometre, nanometre | powers of ten | defined |
| ångström | 1e-10 | defined, not SI |
| astronomical unit | 149 597 870 700 | **defined and exactly representable** |
| light-year | 9 460 730 472 580 800 | **defined and exactly representable** |
| parsec | 3.085 677 581 491 367 3e16 | defined, irrational |
| nominal solar radius R☉ | 6.957e8 | conventional |

The astronomical unit has been a defined number of metres since **IAU 2012
Resolution B2**, which replaced the old definition derived from the Gaussian
gravitational constant. 149 597 870 700 is an integer well below 2⁵³, so it
survives a round trip bit for bit.

The light-year is c times the Julian year, both exact integers. Their product,
9 460 730 472 580 800, is larger than 2⁵³, but it has enough factors of two
that it is still representable to the last bit; the test asserts the equality
rather than assuming it.

The parsec is exactly (648 000 / π) au, the distance at which one au subtends
one arcsecond. It carries π, so it is irrational and cannot round-trip
exactly. It is the one length in the table compared with a tolerance.

R☉ is **IAU 2015 Resolution B3**, a convention rather than a measurement of the
actual Sun, which is neither spherical nor constant.

### Time

| Unit | Value in seconds | Status |
| --- | --- | --- |
| minute, hour, day | 60, 3600, 86 400 | defined |
| Julian year | 31 557 600 | defined, exactly 365.25 days |
| megayear, gigayear | 1e6 and 1e9 Julian years | defined |

The year here is the **Julian** year. It is not the tropical year and not a
calendar year, neither of which is a fixed number of seconds, and it is the one
the light-year is defined against. Astronomy quotes intervals in Julian years
for exactly this reason.

### Mass

| Unit | Value in kilograms | Status |
| --- | --- | --- |
| gram, tonne | 1e-3, 1e3 | defined |
| atomic mass constant u | 1.660 539 068 92e-27 | **measured**, CODATA 2022 |
| nominal solar mass M☉ | 1.988 409 870 698 051e30 | **derived from a measurement** |
| nominal Earth mass M⊕ | 5.972 167 867 791 379e24 | **derived from a measurement** |

The two astronomical masses are the only entries in this module that depend on
a measured constant, and it is worth being precise about why.

What the IAU fixes exactly is the **mass parameter** GM, not the mass:

| Parameter | Value | Status |
| --- | --- | --- |
| nominal GM☉ | 1.327 124 40e20 m³/s² | conventional, IAU 2015 B3 |
| nominal GM⊕ | 3.986 004e14 m³/s² | conventional, IAU 2015 B3 |

GM is what orbits actually measure, and they measure it to far more digits than
G is known to. Splitting it into a mass means dividing by a measured G:

> **G = 6.674 30e-11 m³ kg⁻¹ s⁻²** (CODATA 2018 and 2022, relative standard
> uncertainty 2.2e-5)

which throws away four significant figures. So `units::solarMass` carries G's
uncertainty and `constants::nominalSolarMassParameter` does not. **Integrate
orbits with GM.** The mass in kilograms exists because applications ask for
kilograms.

`units_conversions.cpp` divides the stored GM by the G recorded above and
asserts it reproduces the stored mass, so the two cannot drift apart if either
is retyped.

### Force, pressure and energy

| Unit | Value in SI | Status |
| --- | --- | --- |
| dyne | 1e-5 N | defined, CGS |
| erg | 1e-7 J | defined, CGS |
| bar | 1e5 Pa | defined |
| standard atmosphere | 101 325 Pa | defined, CGPM 1954 |
| electronvolt | 1.602 176 634e-19 J | **defined**, being e times one volt |
| standard gravity g₀ | 9.806 65 m/s² | defined, CGPM 1901 |

The electronvolt is exact because the elementary charge is one of the constants
that defines the SI. Before 2019 it was a measured quantity; it is not any
more.

The CGS units are here because older astrophysics literature is written in
them, and reading a published number is where a conversion error is most
likely.

Standard gravity is a convention used to define the kilogram-force and to quote
g-loads. It is not the acceleration anywhere in particular on Earth.

### Temperature

Kelvin only. Celsius and Fahrenheit are affine rather than scaled, so they are
conversion functions rather than unit constants; see "What a dimension cannot
tell you" above for why that distinction has to be visible in the API.

| Conversion | Relation |
| --- | --- |
| Celsius | K = °C + 273.15 |
| Fahrenheit | K = (°F − 32) × 5/9 + 273.15 |

The two scales cross at −40, which is the single value that catches a slope and
an offset that are both wrong in compensating ways, so that is what the test
checks.

### Luminosity

| Unit | Value in SI | Status |
| --- | --- | --- |
| nominal solar luminosity L☉ | 3.828e26 W | conventional, IAU 2015 B3 |
| candela, lumen, lux | 1 | base and derived SI |

L☉ is fixed to a round number by convention, chosen so that published stellar
luminosities do not shift every time the Sun is remeasured.

Radiometry and photometry are kept apart because they are different physics:
the candela is a base quantity of the SI, not something derivable from watts,
since the weighting is a statement about human observers rather than about
light. No amount of dimensional algebra recovers it.

### Sources

- BIPM, *The International System of Units (SI)*, 9th edition, 2019.
- IAU 2012 Resolution B2, on the re-definition of the astronomical unit.
- IAU 2015 Resolution B3, on nominal conversion constants for selected solar
  and planetary properties.
- CODATA internationally recommended values of the fundamental physical
  constants, 2022 adjustment.
- CGPM Resolution 2 (1954) for the standard atmosphere; CGPM 3rd Conference
  (1901) for standard gravity.
