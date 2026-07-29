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
cmake -B build
cmake --build build
```

If you already cloned without `--recurse-submodules`, run
`git submodule update --init --recursive` first. No GPU is required: the CPU
is the reference implementation the whole engine and test suite run against,
so this builds and runs on any machine.

## Run it

```sh
./build/bin/solar-system
```

This is the one example application that exists so far: the Sun and five
planets, moving under Newtonian gravity, with the total energy and momentum
of the system plotted live in a chart alongside the 3D view. Watching that
chart is the point: energy and momentum are supposed to stay flat, and if a
change you make later breaks the physics, that's usually the first place
it'll show up as a visible drift instead of a flat line.

## Change something

The scenario lives in `src/Applications/SolarSystem/`, separately from the
window and rendering code, specifically so it's easy to find and easy to
change. Open it, change a planet's initial velocity or the simulation's time
scale, rebuild, and rerun:

```sh
cmake --build build
./build/bin/solar-system
```

Push a planet's velocity up and watch its orbit stretch into a wider
ellipse; push it up enough and watch it escape the system entirely, the
energy plot telling you the difference between a bound and an unbound orbit
before you even see it happen on screen.

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
