# Units

Conversion factors, their sources, and which of them are exact.

Design rationale lives in [src/Units/README.md](../src/Units/README.md). This
document is the table of numbers and where they came from, and the record of
what is definition and what is measurement.

## Why exactness is tracked

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

## The seven defining constants of the SI

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

## Length

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

## Time

| Unit | Value in seconds | Status |
| --- | --- | --- |
| minute, hour, day | 60, 3600, 86 400 | defined |
| Julian year | 31 557 600 | defined, exactly 365.25 days |
| megayear, gigayear | 1e6 and 1e9 Julian years | defined |

The year here is the **Julian** year. It is not the tropical year and not a
calendar year, neither of which is a fixed number of seconds, and it is the one
the light-year is defined against. Astronomy quotes intervals in Julian years
for exactly this reason.

## Mass

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

## Force, pressure and energy

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

## Temperature

Kelvin only. Celsius and Fahrenheit are affine rather than scaled, so they are
conversion functions rather than unit constants; `src/Units/README.md` explains
why that distinction has to be visible in the API.

| Conversion | Relation |
| --- | --- |
| Celsius | K = °C + 273.15 |
| Fahrenheit | K = (°F − 32) × 5/9 + 273.15 |

The two scales cross at −40, which is the single value that catches a slope and
an offset that are both wrong in compensating ways, so that is what the test
checks.

## Luminosity

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

## Sources

- BIPM, *The International System of Units (SI)*, 9th edition, 2019.
- IAU 2012 Resolution B2, on the re-definition of the astronomical unit.
- IAU 2015 Resolution B3, on nominal conversion constants for selected solar
  and planetary properties.
- CODATA internationally recommended values of the fundamental physical
  constants, 2022 adjustment.
- CGPM Resolution 2 (1954) for the standard atmosphere; CGPM 3rd Conference
  (1901) for standard gravity.
