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
| `SolarSystem` | Sun and five planets, Newtonian N-body gravity, energy/momentum tracked live |
| `LunarEclipse` | Sun, Earth, Moon, Jupiter under real n-body gravity with Earth's J2 and rotation active; the Moon's eclipse illumination is computed from `Optics/Illumination.hpp`, not scripted |

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
