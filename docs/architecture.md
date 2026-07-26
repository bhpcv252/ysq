# Architecture

## The one rule

Dependencies flow one way. Nothing lower depends on anything higher.

```
Applications
     |
Renderer, UI          presentation
     |
  Physics             organized by theory
     |
  Compute             CPU reference + GPU backends
     |
Core, Math, Units, Platform      base
```

`Units` builds on `Math`. `Platform` (window, GL context, input) sits in the base
layer because `Compute`'s OpenGL backend needs a context from it. `Core` depends
on nothing in the engine.

Each module is its own CMake target, static or `INTERFACE` where it is
template-only, and its own `README.md`. A module that needs something from a
higher layer is a design error, not a linking problem: the abstraction is in the
wrong place. Move the concept down, do not add an edge upward.

## Keep the engine domain-neutral

The engine encodes math and physics: what things are and how they behave.
Applications encode scenarios: which things exist, with what initial conditions,
and what to measure.

A planet's mass belongs to an application. The gravitational force law belongs to
`Physics`. If a scenario needs new physics, the physics goes in the engine and
the scenario stays in `Applications/`. This is what makes the engine reusable
across the solar system, a binary pair, a galaxy collision and a black hole
without special-casing any of them.

## Physics is organized by theory, not by application

`Mechanics`, `Spacetime`, `Gravity`, `Electromagnetism`, `Fluids`,
`Thermodynamics`, `Optics`.

Spacetime is the core abstraction. Because a metric defines how anything moves
through it, light propagation, gravitational lensing, and Doppler, gravitational
and cosmological frequency shift are all the same computation: a null geodesic
through a metric, evaluated in different metrics. They are not separate features.

Gravity is deliberately a ladder of approximations rather than one model, because
full dynamical numerical relativity is intractable for general many-body systems:

| Model            | Regime                                    |
| ---------------- | ----------------------------------------- |
| Newtonian        | Weak field, slow motion. Dynamical many-body gravity. |
| Post-Newtonian   | Adds relativistic corrections             |
| Fixed background | Schwarzschild and Kerr near compact objects, FLRW at cosmological scales. Light and test particles. |

An application picks the rung that matches its regime and its budget.

## Compute backends and fallback

`Physics` never talks to a GPU. It dispatches through the `ComputeBackend`
interface, and a backend is selected at runtime.

**The CPU backend is the reference implementation, not a degraded mode.** It
defines what correct means. Every GPU backend is an optimization that must
reproduce the CPU result within tolerance, and the test suite enforces that. This
is why the whole test suite runs CPU-only and needs no GPU, no window and no
display.

Selection probes candidates in priority order and takes the first that reports
available, with a manual override for debugging and benchmarking:

```
CUDA         -> NVIDIA GPU and toolkit present
Vulkan       -> Vulkan 1.1+ loader with a compute queue
OpenGL 4.3+  -> context reports 4.3 or higher (compute shaders)
CPU          -> always succeeds
```

The bottom rung has no hardware requirement, so there is no machine where the
engine fails to start. What varies is throughput, not capability.

| Platform                    | Best compute available          | Rendering    |
| --------------------------- | ------------------------------- | ------------ |
| macOS                       | CPU today; Vulkan via MoltenVK, or a native Metal backend | OpenGL 4.1 |
| Linux / Windows, NVIDIA     | CUDA, then Vulkan, then GL compute | OpenGL 4.6 |
| Linux / Windows, AMD/Intel  | Vulkan, then GL compute         | OpenGL 4.5+  |
| Headless server, CI         | CPU                             | none needed  |

macOS is the constrained case and it is worth being precise about why. Apple
deprecated OpenGL at 4.1 and never shipped compute shaders, which are a 4.3
feature, so the OpenGL compute backend cannot run there. NVIDIA drivers have not
been available for years, so CUDA is permanently out. Vulkan is not native either,
but MoltenVK translates it to Metal and does support compute. So macOS has a GPU
compute path; it just runs through Vulkan or Metal rather than OpenGL or CUDA.

Rendering is a separate axis and is not constrained the same way. OpenGL 4.1 is
sufficient for the real-time rasterized visualizer with ImGui panels, so macOS is
a first-class rendering target.

### Precision

Consumer GPUs are poor at `float64`, often between 1/32 and 1/64 of `float32`
throughput. GPU backends will generally run `float32` while the CPU reference runs
`float64`. Two consequences:

- "GPU matches CPU" is a tolerance comparison, never equality.
- Some scenarios, long-baseline orbital integration in particular, may need to
  stay on CPU for accuracy regardless of available hardware. Backend selection
  therefore eventually considers the scenario, not only the hardware.

### Ray tracing

The ray tracer has a portability fork worth deciding deliberately when `Renderer`
lands. Written as an OpenGL compute shader it will not run on macOS. Written as a
fragment shader doing the same raymarching it runs on 4.1 and is portable
everywhere. Same math, different shader stage.

The light physics itself belongs in `Physics/Optics` either way. `Renderer` only
decides where it executes.

## Build layout

| Path           | Contents                                              |
| -------------- | ----------------------------------------------------- |
| `cmake/`       | Shared CMake modules (currently the warning sets)      |
| `third_party/` | Vendored dependencies, all built from source           |
| `src/`         | Engine modules and applications, one target each       |
| `tests/`       | Smoke, unit, integration and e2e, outside the library graph |

Two build options shape what gets configured:

- `YSQ_BUILD_GRAPHICS=OFF` drops GLFW, GLAD and ImGui entirely. The CI matrix
  builds this configuration on every push, so the claim that the core is
  graphics-free stays true rather than becoming aspirational.
- `YSQ_BUILD_TESTS=OFF` drops GoogleTest.

### CI

Six jobs: Linux, macOS and Windows, each with graphics on and off, all with
warnings as errors. Windows is not optional coverage. The project claims MSVC
support and `cmake/YsqWarnings.cmake` carries an MSVC-specific warning set, so
without a Windows job both the claim and the code behind it would be untested.

Everything CI runs is CPU-only and needs no display, so no runner needs a GPU or
a virtual framebuffer.

### Warnings

Two INTERFACE targets in `cmake/YsqWarnings.cmake`, linked `PRIVATE` so they
apply to a module's own sources without leaking to consumers:

- `ysq::warnings` for everything
- `ysq::warnings_strict` adds `-Wconversion -Wsign-conversion -Wdouble-promotion`,
  and goes on the engine core: `Math`, `Units`, `Physics`, `Compute`

The split is deliberate. In the engine core a silent `double` to `float`
narrowing is a physics bug: an energy accumulator or an integrator tolerance
quietly loses precision and a conservation invariant drifts, which is painful to
find by reading. In the presentation layer the same conversions are constant and
intentional, because OpenGL and ImGui are `float`/`int` APIs, so the strict set
there would only train people to reach for `static_cast`, which silences the
warning without fixing anything.

`-Wsign-conversion` is listed explicitly because Clang's `-Wconversion` implies it
for C++ and GCC's does not. Without it the two compilers disagree and a build that
is clean locally fails in CI.

A well-built `Units` module prevents much of this structurally, since a `Length`
and a `Time` will not implicitly convert to each other or to a raw number at all.
Strong types catch at the design level what warnings catch at the compile level.
