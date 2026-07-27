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
| `Units/Format.hpp`         | `std::formatter` for every quantity                            |

Conversion factors, their sources and their exactness are in
[docs/units.md](../../docs/units.md).

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
why the electromagnetic family is not: the ampere is already in the algebra, so
`Charge.hpp` is purely additive when `Electromagnetism` lands, and adding it
early would be guessing at an interface instead of following one.

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
kilograms, and it is the lossy form; `docs/units.md` records the G used.

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
they propagate to every consumer's own sources, and `docs/architecture.md`
rules that out for `Renderer`, `UI` and `Applications`.

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
