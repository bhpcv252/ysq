# Applications/Helper

Scenario-setup code shared between applications, not engine content.

**Target:** `ysq::ApplicationsHelper` (static)
**Depends on:** `ysq::Math`, `ysq::Core` (`BodyCatalog.hpp`'s `Csv`), `ysq::Physics` (`BodyCatalog.hpp`'s `Body` and gravitational constant, and `Physics/Gravity/Kepler.hpp`'s orbital-element machinery every header here builds on).
**Used by:** `SolarSystem`, `LunarEclipse`, and `KeplerSolarSystem`.

## Why this isn't in the engine

The root `CLAUDE.md`'s engine-vs-phenomena rule, and `src/Math/README.md`'s
own charter ("everything the engine computes with, and nothing that knows
what it is computing about"), draw the line between general laws/geometry
and scenario-specific setup. Converting classical orbital elements to a
Cartesian state vector is a general law -- Kepler's own -- so that machinery
lives in `Physics/Gravity/Kepler.hpp` instead, not here.

What's left here is genuinely scenario-setup, not engine content: parsing a
specific CSV column schema, resolving bodies by name and parent, choosing a
render color, and interpreting a published right-ascension/declination pole
for one real catalog's own data. Real orbital mechanics runs underneath all
of it, but this layer is the part that knows what it's loading and how it
will be drawn, which is what keeps it out of `Physics`.

## Contents

| Header | Purpose |
| --- | --- |
| `Applications/Helper/Pole.hpp` | A published pole direction (right ascension/declination) to a frame rotation |
| `Applications/Helper/BodyCatalog.hpp` | A `Csv` table of real bodies to either a resolved hierarchy of `Body` objects (`loadBodyCatalog`, for a real integrator to take over from) or a hierarchy of live orbital elements (`loadKeplerBodyCatalog`, for repeated evaluation) |
| `Applications/Helper/KeplerPopulation.hpp` | A procedural population of non-interacting Kepler orbits (an asteroid belt, a planet's ring) within a real astronomical range |

Classical orbital elements, Kepler's-equation solving, and state-vector
conversion (`OrbitalElements`, `OrbitalElementsAtEpoch`,
`stateVectorFromElements`, `stateVectorAtTime`, `trueAnomalyFromMeanAnomaly`,
`keplerMeanMotion`, `keplerOrbitalPeriod`) live in
`Physics/Gravity/Kepler.hpp`; see `src/Physics/README.md`. Everything below
builds on that.

## Pole

Real satellite orbital elements (JPL's included) are given relative to each
moon's own local Laplace plane, not one shared reference plane, and the data
gives that plane's pole as a right ascension/declination -- in JPL's case,
in the ICRF/J2000 mean equatorial frame. Planetary orbital elements, by
contrast, are conventionally given relative to the J2000 *ecliptic*. Both
kinds of data are real, but they are not expressed in the same frame, and a
simulation needs exactly one.

```cpp
#include <Applications/Helper/Pole.hpp>

// A moon's own Laplace-plane pole, in the equatorial frame JPL publishes it in.
const ysq::Quat laplacePlaneToEquatorial =
    ysq::applications::poleRotation(radians(rightAscensionDeg), radians(declinationDeg));

// Build the orbit in the moon's own local Laplace-plane frame first...
const ysq::KeplerStateVector local =
    ysq::stateVectorFromElements(moonElements, gmParent);
// ...then rotate it into the shared equatorial frame the rest of the
// scenario is built in.
const ysq::Vec3 position = rotate(laplacePlaneToEquatorial, local.position);
const ysq::Vec3 velocity = rotate(laplacePlaneToEquatorial, local.velocity);
```

A planet's own ecliptic-referenced elements need the complementary fixed
rotation -- the J2000 obliquity, 23.4392911 degrees, about the shared
vernal-equinox axis (`Quat::fromAxisAngle(Vec3::unitX(), radians(23.4392911))`)
-- applied once to carry them into the same equatorial frame every moon's
data already targets. `BodyCatalog.hpp` is where that composition actually
happens, per row, based on whether a row's own `pole_ra_deg`/`pole_dec_deg`
are present.

## BodyCatalog

Loads a whole hierarchy -- a star, its planets, their moons -- from a `Csv`
table in one pass: parent resolution, the Kepler/Pole composition above
applied per row, real absolute positions and velocities out. See
`BodyCatalog.hpp`'s own doc comment for the full column schema;
`SolarSystem/data/solar_system_bodies.csv` is the real worked example (Sun,
all 8 planets, and every moon JPL SSD publishes orbital elements for).

```cpp
#include <Applications/Helper/BodyCatalog.hpp>
#include <Core/Csv.hpp>

const std::optional<ysq::Csv> table = ysq::Csv::load("bodies.csv", &csvError);
const ysq::Quat eclipticToEquatorial =
    ysq::Quat::fromAxisAngle(ysq::Vec3::unitX(), radians(23.4392911));

std::string error;
const std::optional<std::vector<ysq::applications::CatalogBody>> bodies =
    ysq::applications::loadBodyCatalog(*table, eclipticToEquatorial,
                                       ysq::applications::kJ2000JulianDate, &error);
```

**Real data mixes reference epochs.** JPL's own satellite tables are not
all published at J2000 -- some (Uranus's and Neptune's irregular moons,
Uranus's inner regular moons) use later epochs, since that is simply when
each table was last fit to observations. A row's optional `epoch_jd`
column names its own epoch; `loadBodyCatalog` propagates that row's mean
anomaly forward to `targetEpochJulianDate` at the two-body mean motion
(`Physics/Gravity/Kepler.hpp`'s `keplerMeanMotion`) before converting it to
a true anomaly, so every body
in the result is consistent at the same instant regardless of what epoch
its source table happened to use. A row with no `epoch_jd` is assumed
already at the target epoch.

## loadKeplerBodyCatalog

`loadBodyCatalog`'s own row parsing, parent resolution and epoch
propagation, kept as live `OrbitalElementsAtEpoch` instead of collapsed to
one fixed `Body`: for a caller (`KeplerSolarSystem`) that re-evaluates
every body's position at simulation time directly via
`Physics/Gravity/Kepler.hpp`'s `stateVectorAtTime`, rather than handing one
initial condition to a real
n-body integrator. Same column schema, same validation, same errors as
`loadBodyCatalog` -- the two loaders parse the same file identically, they
just build a different result from it.

```cpp
const std::optional<std::vector<ysq::applications::KeplerCatalogBody>> bodies =
    ysq::applications::loadKeplerBodyCatalog(*table, eclipticToEquatorial,
                                             ysq::applications::kJ2000JulianDate, &error);
```

Each `KeplerCatalogBody` carries its own `parentIndex` (into this same
result vector, `-1` for the one root), that parent's own gravitational
parameter, its own frame rotation (the same pole-or-`referenceFrameRotation`
composition `loadBodyCatalog` performs), and its `elements` (`std::nullopt`
for the root). A caller walks the chain itself: a body's absolute position
at time `t` is its parent's own (recursively resolved the same way) plus
`rotate(frameRotation, stateVectorAtTime(*elements, parentGm, t).position)`.
`KeplerSolarSystem/main.cpp`'s own `BodyPositions` is the worked example,
memoized per frame rather than assuming the catalog happens to list every
parent before its own children.

## KeplerPopulation

A procedural population of particles on independent, non-interacting
Kepler orbits -- what an asteroid belt or a planet's ring actually is, at
the scale an application like `KeplerSolarSystem` renders at. General
population generation, not specific to any one belt or ring; a caller
supplies the real astronomical range.

```cpp
#include <Applications/Helper/KeplerPopulation.hpp>

const std::vector<ysq::applications::KeplerParticle> asteroidBelt =
    ysq::applications::generateKeplerPopulation(
        /* parentIndex */ 0, gmSun,
        /* minSemiMajorAxis */ 2.1 * ysq::units::astronomicalUnit.value(),
        /* maxSemiMajorAxis */ 3.3 * ysq::units::astronomicalUnit.value(),
        /* maxEccentricity */ 0.3, /* maxInclination */ radians(20.0),
        /* count */ 4000, /* seed */ 1, /* realRadiusMeters */ 2000.0,
        /* renderSize */ 0.015f, ysq::Vec3f{0.55f, 0.5f, 0.42f});
```

Each `KeplerParticle` carries its own `parentIndex`/`parentGm` (the same
fields a `KeplerCatalogBody` has, so both feed the same
`stateVectorAtTime` call at render time) and elements sampled uniformly
within the given ranges. `seed` makes the result deterministic: the same
seed always produces the same population, for both a stable picture across
runs and a testable generator. Both `realRadiusMeters` and `renderSize`
are real render sizes, not a real-vs-fake choice: the first is what these
particles actually are at this render scale (usually far too small to
see individually), the second an artistic size chosen so the population
is visible at all -- `KeplerSolarSystem/main.cpp`'s own "true-to-scale"
toggle picks between the two per frame.

**What this does not model.** A real belt's own structure -- Kirkwood
gaps, resonant clumping -- is a real gravitational effect between the
belt's own bodies and a nearby giant planet. Every particle here is an
independent, non-interacting two-body orbit around its one parent, so none
of that emerges; this populates a belt's real *shape* (where it is), not
its internal dynamics. See
`src/Applications/README.md`'s "Closed-form propagation vs. real N-body"
section for the same tradeoff at the whole-application level.
