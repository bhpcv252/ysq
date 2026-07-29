# Documentation

Consumer-facing documentation: what YSQ teaches about, what the engine gives
you for it, and how to use that to build a simulation in `Applications/`.
Written for anyone from a hobbyist with no physics or math background to a
working scientist: read top to bottom for the first, or skip straight to
"Using it" and "Go deeper" on each page for the second.

This documentation has three tiers, each with a different job:

1. **The pages below** are concept-level: the underlying idea, a survey of
   what YSQ provides, and a worked example. Read these first.
2. **[docs/api/](api/README.md)** is pure reference: every public class and
   function, organized by module, with signatures and edge-case behavior
   but no explanation of why. Look something up here while writing
   `Applications/` code.
3. **Each module's own `src/<Module>/README.md`** is the authoritative
   design record: full derivations, invariants, and the reasoning behind a
   choice that a signature alone doesn't explain.

Each page below is self-contained and links to both of the other two tiers
under "Go deeper."

## Start here

- [Getting started](getting-started.md): build YSQ, run the one example
  application that exists, change something in it
- [API reference](api/README.md): every public class and function, by
  module, if you already know roughly what you want

## Engine modules

- [Core](core.md): logging, timing, identity, events, configuration
- [Vectors, matrices, and exact derivatives](math/algebra.md): the
  computational vocabulary, and how automatic differentiation works
- [Numerical integration](math/integrators.md): ODEs, steppers, and why
  symplectic methods matter for a long-running simulation
- [Units and dimensional analysis](units.md): how a dimension mismatch
  becomes a compile error instead of a lost spacecraft
- [Platform](platform.md): the window, the graphics context, and input
- [Compute](compute.md): CPU and GPU backends, and why CPU is the reference,
  not the fallback

## Physics

- [Physics overview](physics/index.md): the gravity ladder, and why
  spacetime is the one abstraction underneath light, lensing, and redshift
- [Mechanics](physics/mechanics.md): `Body`, momentum over velocity, proper
  time, and the units boundary
- [Gravity](physics/gravity.md): Newtonian gravity, softening, Barnes-Hut,
  the 1PN correction
- [Spacetime](physics/spacetime.md): metrics, geodesics, and the four
  spacetimes YSQ can put something in
- [Electromagnetism](physics/electromagnetism.md): Coulomb and Biot-Savart
  fields, the Lorentz force, a Maxwell solver that actually propagates
- [Fluids](physics/fluids.md): particles that carry the fluid (SPH) versus
  a fixed grid it flows through (Eulerian)
- [Thermodynamics](physics/thermodynamics.md): the ideal gas law, black-body
  radiation, and the heat equation
- [Optics](physics/optics.md): light propagation, lensing, and frequency
  shift as one computation

## Rendering and UI

- [Renderer](renderer.md): the camera, meshes, debug drawing, and the ray
  tracer
- [UI](ui.md): ImGui panels bound to your own variables, ImPlot charts as
  diagnostics

## Building a simulation

- [Applications](applications.md): the convention every simulation follows,
  and how to start a new one

## Tutorials

Four tutorials, in order, building one simulation up from nothing:

1. [Your first simulation](tutorials/01-your-first-simulation.md): a
   two-body orbit, no rendering
2. [Adding visualization](tutorials/02-adding-visualization.md): the same
   orbit, with a window, a camera, and a live energy chart
3. [Choosing a gravity model](tutorials/03-choosing-a-gravity-model.md):
   scaling to many bodies, and adding relativistic precession to one
4. [Spacetime and light](tutorials/04-spacetime-and-light.md): leaving the
   gravity ladder for a full geodesic through curved spacetime

## Fast lookup

- [Cookbook](cookbook.md): short answers to specific questions, once you
  already know roughly what you want

---
Notice something missing or wrong in these docs?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+missing)
and let us know.
