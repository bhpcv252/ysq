# Units and dimensional analysis

How YSQ makes an entire class of physics bug impossible to compile.

## The idea

Every physical quantity has a *kind*, not just a number: 5 is meaningless on
its own, but 5 metres, 5 kilograms, and 5 seconds are three completely
different things, and adding any two of them together is nonsense. This
sounds obvious, and it is, right up until a large codebase passes plain
numbers around and nothing enforces it. That's not a hypothetical: NASA's
Mars Climate Orbiter was lost in 1999 because one piece of ground software
produced thruster data in pound-force-seconds and another consumed it as
newton-seconds, and nothing in between caught the mismatch until the
spacecraft burned up in the Martian atmosphere.

YSQ's answer is to make the kind of a quantity part of its C++ type. A
`Length` and a `Mass` are different types, so `length + mass` is a compile
error, not a wrong answer discovered later. This costs nothing at run time;
the type system checks it once, at compile time, and the compiled code is
identical to using plain numbers.

## What YSQ gives you

A `Quantity` is a value tagged with a `Dimension`, the exponents of the
seven SI base quantities (length, mass, time, current, temperature, amount,
luminous intensity) that describe what kind of thing it is. Addition
requires both sides to carry the same dimension; multiplication and
division combine dimensions arithmetically, so `Length / Time` is
automatically a `Speed`, a new type, without anyone having declared it by
hand.

| Header | Gives you |
| --- | --- |
| `Units/Unit.hpp` | `Dimension`, `Quantity`, and the whole algebra |
| `Units/Constants.hpp` | The seven constants that define the SI, and the IAU nominal solar/terrestrial values |
| `Units/Length.hpp`, `Mass.hpp`, `Time.hpp`, `Velocity.hpp`, `Acceleration.hpp` | The basics |
| `Units/Force.hpp` | Force, momentum, angular momentum, torque, pressure |
| `Units/Energy.hpp` | Energy, power, action |
| `Units/Temperature.hpp` | Temperature, heat capacity, entropy |
| `Units/Electromagnetism.hpp` | Electric field, magnetic flux density |
| `Units/Luminosity.hpp` | Radiometry and photometry |
| `Units/Format.hpp` | `std::formatter` for every `Quantity`, printing honest base-unit powers rather than a guessed symbol |

A `Length` is not tied to any one unit: `units::kilometre` is just a
`Length` whose value is 1000, so converting into or out of a unit is
ordinary multiplication or division, and a length built from kilometres and
one built from metres are the same type and the same bits.

## Using it

```cpp
#include <Units/Length.hpp>
#include <Units/Time.hpp>
#include <Units/Velocity.hpp>

const ysq::Length d = 5.0 * ysq::units::kilometre;
const ysq::Time   t = 2.0 * ysq::units::second;
const ysq::Speed  v = d / t;       // Length / Time is a Speed, by construction
// const ysq::Speed w = d + t;     // does not compile, and cannot be made to
```

Vector quantities work the same way, and this is what a real force law
looks like written in `Units` rather than in bare numbers:

```cpp
#include <Units/Force.hpp>

const ysq::Force  magnitude = gravitationalConstant * m1 * m2 / lengthSquared(r);
const ysq::Vec3   direction = normalized(r);   // a direction is dimensionless
const ysq::Force3 gravity   = -(magnitude * direction);
```

`normalized(r)` returns a plain `Vec3`, not a `Length3`, because a direction
genuinely has no dimension; `magnitude * direction` is how a size and a
direction go back together into a vector quantity. There's no `operator*`
between two vector quantities directly, on purpose: nothing about "a length
vector times a length vector" says whether you mean `dot` or `cross`, so
`Units` makes you say which.

## Go deeper

[docs/api/units.md](api/units.md) has every dimension alias, `Quantity`
free function, unit constant, and literal suffix, per header.

[src/Units/README.md](../src/Units/README.md) has the full type catalogue,
the design rationale (why `Vector3<Length>` can't exist, why storage is
always SI, why `G` deliberately isn't one of the seven constants), and a
**Conversion factors** section recording exactly which numbers are defined,
which are measured, and which are conventions fixed by agreement, since a
test that treats all three the same way would hide a quietly rounded
constant.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+units)
and let us know.
