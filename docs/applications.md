# Applications: building your own simulation

Everything before this page taught you a piece of the engine. This page is
where they come together: how a runnable simulation is actually put
together, and how to start a new one of your own.

## The idea

An `Application` is a scenario: which bodies exist, what their starting
state is, which rung of the gravity ladder (or which metric, or which
fluid model) applies, and what to draw and measure while it runs. Nothing
here is new physics; that all lives in the engine modules covered on the
earlier pages. An `Application` is where you *choose*, not where you
*invent*.

The one convention worth understanding before you write your own: **scenario
setup is kept separate from windowing and rendering.** A scenario's
"what bodies exist, with what state" code has no dependency on `Platform`,
`Renderer`, or `UI` at all, which means it's pure physics that can be
constructed and run headless, with no window, no display, nothing graphical
at all required. That's what lets a test assert the same physical
invariants (energy conservation, a stable orbit) that the windowed program
would show, without a display, in CI, on every push.

## What YSQ gives you

One subdirectory per application, named for what it is (`SolarSystem/`, not
`App1/`), holding:

| File | Purpose |
| --- | --- |
| `CMakeLists.txt` | An executable for `main.cpp`, plus a small library target for the scenario code an `e2e` test also links |
| `main.cpp` | Window creation, the fixed-step simulation loop, rendering and UI |
| `Scenario.hpp`/`.cpp` | Pure physics: what bodies exist and their initial state, no `Platform`/`Renderer`/`UI` dependency |

`main.cpp` is where [docs/core.md](core.md)'s `Clock` loop,
[docs/platform.md](platform.md)'s window, [docs/renderer.md](renderer.md)'s
draw calls, and [docs/ui.md](ui.md)'s panels all actually meet. It owns its
own list of what to draw; there's no scene graph here either, the same rule
`Renderer` follows.

## Using it: starting a new application

1. **Create the directory.** `src/Applications/YourScenario/`, and add
   `add_subdirectory(YourScenario)` to `src/Applications/CMakeLists.txt`.
2. **Write `Scenario.hpp`/`.cpp`.** Construct whatever `Body`s (or
   `SPHParticle`s, or a `Schwarzschild` metric, or whatever the scenario
   needs) with their initial state. No `Platform`, `Renderer`, or `UI`
   `#include` anywhere in this file.
3. **Write `CMakeLists.txt`.** An `add_executable` for `main.cpp`, and a
   small library target wrapping `Scenario.cpp` that both `main.cpp` and,
   later, an `e2e` test can link. Link only what a `PUBLIC` dependency
   doesn't already give you: if your scenario library already exposes
   `Physics` `PUBLIC`, `main.cpp` doesn't need to list `Math` or `Units`
   again just because it happens to `#include` their headers directly.
4. **Write `main.cpp`.** Initialize `Platform`, create a `Window`, build a
   `Renderer` and `ImGuiLayer`, construct your scenario, then the fixed-step
   loop from [docs/core.md](core.md): `clock.advance(...)`, `while
   (clock.consumeStep()) { ... }`, draw, repeat.
5. **Add an `e2e` test**, if there's a physical invariant worth pinning
   (energy conservation is the usual one): link the scenario library
   directly, run it for a fixed number of steps with no window at all, and
   assert on the result.
6. **Build and run:**

```sh
cmake --build build
./build/bin/your-scenario
```

`SolarSystem/`, `LunarEclipse/` and `KeplerSolarSystem/` are the three
worked examples that exist today; reading any of their `CMakeLists.txt`
alongside this page is worth doing before writing your own, since it's the
concrete answer to "what does 'only link what a `PUBLIC` dependency
doesn't already give you' actually look like."

**Building a real orbit from published orbital elements?** Real data (JPL's,
or any astronomy source) comes as classical orbital elements -- semi-major
axis, eccentricity, inclination, and so on -- not as the position and
velocity an n-body simulation actually starts from.
`Physics/Gravity/Kepler.hpp`'s `stateVectorFromElements` does that
conversion; it's a general two-body law, so it lives in the engine rather
than in `Applications/`, and any scenario needing it just calls it directly.
`LunarEclipse/Scenario.cpp` is the worked example for one body at a time:
real elements in, a real Cartesian state out, one call per orbiting body.

For a whole hierarchy at once -- a star, its planets, their moons, all from
one downloaded table -- `Applications/Helper/BodyCatalog.hpp`'s
`loadBodyCatalog` does the parent resolution and per-row real-vs-published
reference-frame handling (`Pole.hpp`'s job) for you; see its own doc
comment for the CSV column schema and
`SolarSystem/data/solar_system_bodies.csv` for the real, working example.

**Not running a real integrator at all?** `loadBodyCatalog` resolves each
body to one fixed initial `Body`, for a real n-body integrator to take
over from -- the right shape if your scenario has bodies that actually
pull on each other. If it doesn't (your bodies orbit independently, no
real gravitational interaction between them), `BodyCatalog.hpp`'s sibling
`loadKeplerBodyCatalog` keeps every body's own orbital elements live
instead, so you evaluate its position directly at whatever simulated time
you want (`Physics/Gravity/Kepler.hpp`'s `stateVectorAtTime`) rather than
stepping there.
`KeplerSolarSystem/Scenario.cpp` and `main.cpp` are the worked example,
including `Applications/Helper/KeplerPopulation.hpp` for a procedural,
non-interacting population (an asteroid belt, a ring) within a real
astronomical range. See `src/Applications/README.md`'s "Closed-form
propagation vs. real N-body" section for what you trade away by not
running a real integrator.

**`LunarEclipse` is the sharper example of "nothing here is new physics."**
An eclipse isn't a law, it's what a scene of real bodies with real physical
properties looks like when the engine's general occlusion, refraction and
scattering laws (`Optics/Illumination.hpp`) are pointed at it. Its
`Scenario.cpp` sets up Sun, Earth, Moon and Jupiter with real masses, radii,
orbital elements, and Earth's real J2, rotation and atmosphere; nothing
about what an eclipse *is* appears until `main.cpp` calls `illuminate()`
with that geometry and reads back a color. If you're building a scenario
around a named real-world phenomenon rather than an open-ended system like
a solar system, this is the shape to follow, not a special case.

## Go deeper

[src/Applications/README.md](../src/Applications/README.md) has the full
convention, including the `SolarSystem/CMakeLists.txt` worked example this
page's step 3 refers to. The [tutorials](tutorials/01-your-first-simulation.md)
walk through building a scenario from nothing, one piece at a time, which
is worth doing once before improvising your own.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+applications)
and let us know.
