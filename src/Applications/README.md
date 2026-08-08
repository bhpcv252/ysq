# Applications

Runnable programs built on the engine. Each one sets up a scenario — what
bodies exist, their initial state, which rungs of the physics ladder apply —
and runs it against the shared description of reality `src/` provides.
Presentation layer's top; see the root `README.md`'s Project structure
section for why nothing lower may depend on this.

**Target:** none of its own. Each subdirectory is its own executable,
`add_subdirectory`'d here, linking whichever engine libraries it needs.

This directory itself is added unconditionally, not only under
`YSQ_BUILD_GRAPHICS`. An application's *executable* is windowed and so is
graphics-only, but its scenario-construction code is pure physics and builds
either way; see the convention below and `tests/e2e/`, which links exactly
that code headless.

## Convention

One subdirectory per application, named for what it is
(`SolarSystem/`, not `App1/`), holding:

- `CMakeLists.txt` — an `add_executable` for `main.cpp`, plus a small
  library target for any scenario-construction code an `e2e` test also
  needs to link without pulling in graphics.
- `main.cpp` — window creation, the fixed-step simulation loop, rendering
  and UI. Owns its own list of objects; nothing here is a scene graph, the
  same rule `Renderer` follows.
- `Scenario.hpp`/`.cpp` (where applicable) — pure physics: what bodies
  exist and their initial state. No `Platform`/`Renderer`/`UI` dependency,
  so `tests/e2e/` can link it directly and assert the same physical
  invariants the running application would show.

## Link what a PUBLIC dependency doesn't already give you

An application's `main.cpp` typically ends up `#include`-ing headers from
several engine modules directly. The instinct to link every one of those
explicitly, in case a transitive `PUBLIC` dependency ever changes, is
reasonable on its own — but every module here already links `PUBLIC`
exactly what its own headers hand back (see each module's own `README.md`
for why), so re-listing something a dependency you've already listed
already exposes `PUBLIC` doesn't add safety, it just puts that library on
the final link line more than once. Harmless — the linker ignores the
repeats — but it recurs for every application that reasons the same way,
so it's worth doing once, correctly: before adding an explicit
`target_link_libraries` entry, check whether a dependency you're already
listing already exposes it `PUBLIC`. Only list what nothing else already
gives you.

`SolarSystem/CMakeLists.txt` is the worked example: `Math`, `Units` and
`Physics` all arrive via `SolarSystemScenario`'s own `PUBLIC ysq::Physics`;
`Platform` arrives via both `Renderer` and `UI`. `Core` and `Compute` are
used directly and nothing else exposes either `PUBLIC`, so both are listed.

## Contents

| Application | Scenario |
| --- | --- |
| `SolarSystem` | The real Sun, all 8 planets, and every moon JPL SSD publishes an orbit for (~175 bodies), true to scale, real Newtonian N-body gravity, energy/momentum tracked live |
| `LunarEclipse` | Sun, Earth, Moon, Jupiter under real n-body gravity with Earth's J2 and rotation active; the Moon's eclipse illumination is computed from `Optics/Illumination.hpp`, not scripted |
| `KeplerSolarSystem` | The same real bodies as `SolarSystem` plus the 5 IAU dwarf planets, procedural asteroid-belt/Kuiper-belt populations and real planetary rings; every position a direct closed-form Kepler evaluation at the current simulated time, not an integration step, so requested speed (1 year/sec, or far beyond) costs nothing extra. See "Closed-form propagation vs. real N-body" below for what that trades away. |

`Helper/` is not an application: it has no executable, just scenario-setup
code more than one application needs (`Pole.hpp`'s
published-pole-to-frame-rotation conversion, `BodyCatalog.hpp`'s
whole-hierarchy CSV loader, and `KeplerPopulation.hpp`'s procedural
population generator, all built for loading and rendering real, downloaded
orbital data). It is not engine content -- it knows about named bodies, a
specific CSV column schema, and render colors, the same distinction the
root `CLAUDE.md`'s "Engine vs. phenomena" note draws for `Applications/`
generally -- so it lives here rather than in `Math` or `Physics`, alongside
the applications that actually use it. The orbital-element and
Kepler's-equation machinery it builds on top of is a general two-body law,
not a scenario, so that part lives in `Physics/Gravity/Kepler.hpp` instead.
See `Helper/README.md` for the reasoning in full.

Binaries land in `build/bin/`; see the root `README.md`'s Running section.

## Compose engine laws here, not phenomena in the engine

`LunarEclipse` is the convention to follow for any future application whose
scenario is a named real-world phenomenon (an eclipse, a tide, a rainbow):
the engine provides only the general laws underneath it (occlusion,
refraction, scattering, gravity, rotation), and the phenomenon itself,
whatever composition of those laws produces the effect a scenario is built
to show, is assembled entirely here. `LunarEclipse/Scenario.cpp` sets up
real bodies with real physical properties; nothing about an "eclipse" is
computed until `main.cpp` calls the engine's general `illuminate()` with
this scenario's own Sun/Earth/Moon geometry and interprets the result. See
the root `CLAUDE.md`'s "Engine vs. phenomena" note.

## Closed-form propagation vs. real N-body

`KeplerSolarSystem` exists as its own application, deliberately not a mode
of `SolarSystem`, because the two answer different questions and trade off
against each other. `SolarSystem` integrates real Newtonian (optionally
1PN-corrected) N-body gravity forward in time: bodies genuinely pull on
each other, which is what makes its energy/momentum tracking, and Mercury's
visible perihelion precession, mean something. That real interaction is
also exactly why it cannot cheaply jump simulated time by a year per
second: `Physics/Mechanics/Hermite.hpp`'s individual-timestep scheduler has
to actually step every body forward, however small a step the fastest one
needs, to get there.

`KeplerSolarSystem` evaluates every body's position directly from its own
fixed (Sun-fixed, per-body) two-body orbital elements at whatever
simulated time is asked for --
`Physics/Gravity/Kepler.hpp`'s `stateVectorAtTime` -- so a jump of any
size costs the same handful of Newton-Raphson iterations a jump of one
second would. The real cost: no body's gravity affects any other's. A
moon's ellipse around its planet never perturbs another moon; the
asteroid belt shows no real Kirkwood gaps, because those come from actual
resonant interaction with Jupiter that a non-interacting Kepler orbit
cannot produce. What it does still show as real, not scripted, physics:
`Physics/Gravity/PostNewtonian.hpp`'s own closed-form GR perihelion
precession rate, applied as a real rotation of each Sun-parented body's
own argument of periapsis over time (`Scenario.cpp`'s own precession
loop) -- Mercury's ellipse still visibly precesses at the real rate,
without needing a real integrator to get it.

Its data file (`KeplerSolarSystem/data/solar_system_bodies.csv`) is its
own copy of `SolarSystem`'s, not shared, extended with the 5 IAU dwarf
planets; see that file's own header comment for sourcing. Rings, the
asteroid belt and the Kuiper belt are all the same kind of thing: a real
population of independently orbiting particles
(`Applications/Helper/KeplerPopulation.hpp`), a ring's own particles
parented to its planet (not the Sun), so a real ring shows the same real
differential rotation Kepler's third law implies -- an inner particle
genuinely completes a revolution faster than an outer one. One real,
stated simplification remains, in `Scenario.hpp`'s own `RingPopulation`
doc comment: a ring's particles are sampled in the shared frame the rest
of the scenario's elements are expressed in, not tilted to its real
planet's own spin-axis plane (no per-planet pole is part of this
catalog's schema, only per-moon ones).
