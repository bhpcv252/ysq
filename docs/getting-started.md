# Getting started

Build YSQ, run the one example that exists today, and change something in it.

YSQ is an engine for simulating physical systems and rendering them with
OpenGL. The engine encodes the math and physics; a program under
`Applications/` sets up a specific scenario (what bodies exist, with what
starting positions and masses) and runs it against that shared description
of reality. This page gets you from a fresh clone to seeing that happen.

## Build it

```sh
git clone --recurse-submodules <repo-url> ysq
cd ysq
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

If you already cloned without `--recurse-submodules`, run
`git submodule update --init --recursive` first. No GPU is required: the CPU
is the reference implementation the whole engine and test suite run against,
so this builds and runs on any machine.

`-DCMAKE_BUILD_TYPE=Release` matters here specifically: CMake's own default
is unoptimized, and `solar-system`'s ~175-body gravity simulation is a hot
numerical loop that an unoptimized build does nothing to speed up. Drop it
if you're working on the engine itself and want assertions active while you
develop.

## Run it

```sh
./build/bin/solar-system
```

This is one example application: the real Sun, all 8 planets, and every
moon JPL publishes an orbit for (~175 bodies in all, true to scale), moving
under Newtonian gravity, with the total energy and momentum of the system
plotted live in a chart alongside the 3D view. Watching that chart is the
point: energy and momentum are supposed to stay flat, and if a change you
make later breaks the physics, that's usually the first place it'll show up
as a visible drift instead of a flat line.

## Change something

The scenario lives in `src/Applications/SolarSystem/`, separately from the
window and rendering code, specifically so it's easy to find and easy to
change. Real orbital data lives in `data/solar_system_bodies.csv`, loaded
at startup rather than hardcoded (see
`src/Applications/Helper/README.md`'s BodyCatalog section for the column
format); edit a body's `eccentricity` or `mass_kg` there, or change the
simulation's time scale in `main.cpp`, rebuild, and rerun:

```sh
cmake --build build
./build/bin/solar-system
```

Push a planet's eccentricity up in the CSV and watch its orbit stretch
into a wider ellipse; push its mass down enough elsewhere in the system
and watch a moon's orbit destabilize, the energy plot telling you the
difference between a bound and an unbound orbit before you even see it
happen on screen.

## Go deeper

[docs/applications.md](applications.md) covers the `Applications/`
convention in full: how `solar-system` is put together, and how to start a
new one of your own. The [tutorials](tutorials/01-your-first-simulation.md)
build a simulation up from nothing, one piece at a time, if you'd rather
start there than by editing an existing one.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+getting-started)
and let us know.
