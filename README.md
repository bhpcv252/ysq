# YSQ — Yotta Spacetime Quantities

A C++ engine for simulating and visualizing physical systems, rendered with OpenGL.

The engine encodes the math and physics that define how things behave. Applications
under `Applications/` set up specific scenarios and run them against that shared
description of reality.

## Features

- Relativistic spacetime core: Minkowski, Schwarzschild, Kerr, and FLRW metrics
  with a geodesic solver
- Gravity as a hierarchy of models: Newtonian, post-Newtonian, and GR on fixed
  background metrics
- Electromagnetism, fluid dynamics, thermodynamics
- Optics on curved spacetime: light propagation, gravitational lensing, and
  frequency shift (Doppler, gravitational, cosmological)
- Custom math library: vectors, matrices, quaternions, tensors, ODE integrators
  (including symplectic)
- Strongly-typed, dimensioned quantities (scalar or vector)
- Real-time OpenGL rendering: instanced meshes, multi-light Blinn-Phong,
  immediate-mode debug drawing with billboard and fixed-orientation text
  labels, skyboxes, plus a fragment-shader ray tracer with shadows and
  reflections
- ImGui controls and ImPlot charts (live time-series and scatter) in the same
  window and frame as the 3D view
- GPU compute backends: OpenGL compute shaders, CUDA, Vulkan
- Example application: solar system (Newtonian N-body, energy/momentum tracked
  live)

## Physical models

Gravity is modeled as a ladder of approximations to general relativity, since full
dynamical numerical relativity is intractable for general many-body systems.
Newtonian gravity (GR's weak-field limit) drives dynamical self-gravitating
systems; post-Newtonian adds relativistic corrections; and fixed background metrics
handle light and test particles — Schwarzschild and Kerr near compact objects, FLRW
at cosmological scales. Applications select the model that fits their regime and
budget. Because spacetime is the core abstraction, light propagation, lensing, and
all frequency shifts are the same computation — a null geodesic through a metric —
evaluated in different metrics.

## Dependencies

The core (math, physics, integrators, units) is implemented in-house.
External libraries, vendored under `third_party/`:

| Library    | Role                                          |
| ---------- | --------------------------------------------- |
| OpenGL     | Graphics API                                  |
| GLFW       | Window, input, OpenGL context                 |
| GLAD       | OpenGL function loader (generated, committed) |
| Dear ImGui | Control and debug panels                      |
| Dear ImPlot | Charting on top of Dear ImGui                |
| stb_image  | Image loading for `Renderer::Texture`         |
| spdlog     | Logging (configured to use `std::format`)     |
| GoogleTest | Test framework (tests only)                   |

General string formatting uses C++20 `std::format`, so there's no separate
formatting dependency.

## Requirements

- C++20 compiler with `std::format` (GCC 13+, Clang 17+, or MSVC 19.32+)
- CMake 3.20+

`Core/Logger.hpp` takes its format strings as `std::format_string`, so a bad
format string is a compile error rather than a runtime throw. That name is
`P2508`, which MSVC exposes from 19.32 (VS 2022 17.2); `<format>` alone arrived
earlier, at 19.29.

**No GPU is required.** The CPU compute backend is the reference implementation,
so the engine and its full test suite build and run on any machine. Everything
below is optional acceleration, selected at runtime with a fallback to CPU:

| Optional            | Enables                                                 |
| ------------------- | ------------------------------------------------------- |
| OpenGL 4.1+ driver  | The real-time visualizer                                 |
| OpenGL 4.3+ driver  | The OpenGL compute backend (compute shaders are 4.3)     |
| CUDA Toolkit        | The CUDA compute backend (NVIDIA only)                   |
| Vulkan SDK          | The Vulkan compute backend (via MoltenVK on macOS)       |

macOS caps OpenGL at 4.1, so the visualizer works there but the OpenGL compute
backend does not. See [src/Compute/README.md](src/Compute/README.md) for how
backend selection and fallback work per platform.

### Linux build dependencies

GLFW builds both the X11 and Wayland backends, and needs their development
packages present at build time. On Debian and Ubuntu:

```sh
sudo apt install xorg-dev libwayland-dev libwayland-bin libxkbcommon-dev
```

Windows and macOS need nothing beyond a compiler and CMake. A headless build
(`-DYSQ_BUILD_GRAPHICS=OFF`) needs none of these on any platform.

Optionally, `libosmesa6` provides an OpenGL context on a machine with no display
at all, in software. It is loaded at run time rather than linked, so it is not
needed to build; without it, `Platform`'s headless backend has no context to
give and the one test that wants one skips.

## Building

```sh
git clone --recurse-submodules <repo-url> ysq
cd ysq
cmake -B build
cmake --build build
```

If you already cloned without submodules:

```sh
git submodule update --init --recursive
```

### Options

| Option                   | Default | Effect                                          |
| ------------------------ | ------- | ----------------------------------------------- |
| `YSQ_BUILD_TESTS`        | `ON`    | Build the test suite                             |
| `YSQ_BUILD_COMPILE_FAIL_TESTS` | `ON` | Build the tests that assert something does *not* compile. Each costs a nested compiler invocation. |
| `YSQ_BUILD_GRAPHICS`     | `ON`    | Build against GLFW, GLAD and Dear ImGui. `OFF` drops them entirely, for headless and CI builds. |
| `YSQ_WARNINGS_AS_ERRORS` | `OFF`   | Treat warnings as errors. CI builds with this on. |
| `YSQ_REQUIRE_HEADLESS_GL` | `OFF`  | Fail rather than skip when no headless OpenGL context can be created. For machines known to have OSMesa; CI sets it on the one job that does. |
| `YSQ_BUILD_COMPUTE_CUDA` | `ON`    | Build the CUDA compute backend if the toolkit is found. Detected, not required: an absent toolkit is skipped, not a build failure. |
| `YSQ_BUILD_COMPUTE_VULKAN` | `ON`  | Build the Vulkan compute backend if the SDK is found. Same "use it if found" behaviour as above. |

```sh
cmake -B build -DYSQ_BUILD_GRAPHICS=OFF   # headless: no graphics dependencies
```

## Running

Each program under `Applications/` builds to its own executable in `build/bin/`:

```sh
./build/bin/solar-system
```

Available: `solar-system`.

## Testing

Tests live under `tests/` and run through CTest (built by default; disable with
`-DYSQ_BUILD_TESTS=OFF`):

```sh
ctest --test-dir build
```

Unit tests cover modules in isolation — a vector rotation, an integrator's
observed order of accuracy, a units dimension check. Integration tests cover combinations — a
symplectic integrator with Newtonian gravity holding a stable orbit, or spacetime
with optics reproducing a known deflection angle. End-to-end tests run whole
applications headless and assert physical invariants such as energy and momentum
conservation. Smoke tests sit below all of that and cover the build itself: that
each vendored dependency actually links and that build options took effect.

Compile-failure tests are the one category that asserts an absence. `Units`
guarantees that adding a distance to a mass will not build, and a test suite made
only of programs that compile cannot check a guarantee like that. Those live in
`tests/compile_fail/` as targets marked `WILL_FAIL`, each paired with the
positive form of the same construct in an ordinary test, since a failing build
proves nothing on its own about *why* it failed.

Everything runs CPU-only and needs no GPU, no window and no display. The one
exception is the OpenGL context test, which still needs no display: it uses a
software context where there is no display server, and skips where even that is
unavailable. Configure with `-DYSQ_REQUIRE_HEADLESS_GL=ON` to make those skips
failures. See [tests/README.md](tests/README.md).

## Continuous integration

Six jobs: Linux, macOS and Windows, each with graphics on and off, all built
with warnings as errors. Windows is not optional coverage: the project
claims MSVC support and `cmake/YsqWarnings.cmake` carries an MSVC-specific
warning set, so without a Windows job neither the claim nor the code behind
it would be tested. Every job is CPU-only and needs no display, so none needs
a GPU or a virtual framebuffer.

The one test whose outcome depends on the machine is the OpenGL context test,
which skips where no context can exist. A test that skips on all six jobs
tests nothing, so the Linux graphics-on job installs OSMesa and configures
with `YSQ_REQUIRE_HEADLESS_GL=ON`, turning that skip into a failure on the one
runner where a context is guaranteed to be available.

## Warnings

`cmake/YsqWarnings.cmake` defines two `INTERFACE` targets, linked `PRIVATE` so
they apply to a module's own sources without leaking to its consumers:
`ysq::warnings` for everything, and `ysq::warnings_strict`, which adds
`-Wconversion -Wsign-conversion -Wdouble-promotion` on top and goes on the
engine core (`Math`, `Units`, `Physics`, `Compute`).

The split is deliberate. In the engine core a silent `double`-to-`float`
narrowing is a physics bug: an energy accumulator or an integrator tolerance
quietly loses precision and a conservation invariant drifts. In the
presentation layer (`Renderer`, `UI`, `Applications`) the same conversions are
constant and intentional, since OpenGL and ImGui are `float`/`int` APIs, and
the strict set there would only train people to reach for `static_cast`
without fixing anything. `-Wsign-conversion` is listed explicitly because
Clang's `-Wconversion` implies it for C++ and GCC's does not; without it a
build clean on one compiler fails on the other.

`Math` and `Units` are header-only `INTERFACE` targets, so they cannot carry
the strict set on their own target: on an `INTERFACE` target the flags would
reach every consumer's own sources instead. Both apply it in a smoke test
that includes every header and explicitly instantiates every template, since
an uninstantiated template is barely checked at all.

## Formatting

`.clang-format` is the whole style. CI checks it on pull requests and on pushes
to `main`, over the same file list this formats:

```sh
git ls-files '*.hpp' '*.cpp' '*.h' '*.c' | grep -v '^third_party/' \
    | tr '\n' '\0' | xargs -0 clang-format -i
```

clang-format's output shifts between releases, so CI pins one version
(**22.1.8**, from PyPI). A different local version can disagree with it on
untouched code. `third_party/` is upstream code and is never reformatted.

## Project structure

Each module under `src/` is its own library (static, or header-only `INTERFACE`
where it's template-only), defined by its own `CMakeLists.txt`. Applications link
against those libraries and build to executables. Every library module carries a
`README.md` describing its interface and dependencies. Tests under `tests/` link
the modules they exercise plus GoogleTest, outside the library dependency graph.

Dependencies flow one way. At the base is the system layer: `Core`, `Math`,
`Units`, and `Platform` (window, GL context, input), with `Units` built on `Math`.
`Compute` builds on that — its OpenGL backend uses Platform's context, while the
CPU, CUDA, and Vulkan backends are standalone. `Physics` builds on `Compute` and
falls back to the CPU backend when no GPU is present. `Renderer` and `UI` form the
presentation layer, drawing on `Platform` and `Math`. `Applications` sit on top.
Nothing lower depends on anything higher. A headless visual run uses an offscreen
context; the simulation core and tests need no graphics context at all.

Directory layout; each library module under `src/` carries its own `README.md`
listing its headers in full, so the tree below stops at the module boundary
rather than duplicating that per file. That duplication had drifted out of
sync with the code before this line was written, which is the reason it isn't
done that way anymore.

```
ysq/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── .clang-format
├── .github/workflows/ci.yml
├── cmake/                           Shared CMake modules (warning sets, shader embedding)
├── docs/                            Consumer-facing documentation
│
├── third_party/                    Vendored dependencies
│   ├── README.md
│   ├── glfw/                        submodule
│   ├── glad/                        generated loader, committed
│   ├── imgui/                       submodule
│   ├── implot/                      submodule
│   ├── stb/                         submodule (stb_image)
│   ├── spdlog/                      submodule
│   └── googletest/                  submodule (tests)
│
├── src/
│   ├── CMakeLists.txt
│   │
│   ├── Core/                        Logging, timing, identity, events, configuration
│   ├── Math/                        Vectors, matrices, quaternions, tensors, calculus, ODE integrators
│   ├── Units/                       Dimensioned quantities over the SI
│   ├── Platform/                    Window, GL context, input (graphics builds only)
│   │
│   ├── Compute/                     Backend Physics dispatches to
│   │   ├── CPU/                      Reference implementation, always available
│   │   ├── OpenGL/                   4.3+ compute shaders, graphics builds only
│   │   ├── CUDA/                     Built only when the CUDA Toolkit is found
│   │   └── Vulkan/                   Built only when the Vulkan SDK is found
│   │
│   ├── Physics/                     Mechanics, Spacetime, Gravity, Electromagnetism,
│   │   │                            Fluids, Thermodynamics, Optics: organized by theory
│   │   ├── Mechanics/
│   │   ├── Spacetime/
│   │   ├── Gravity/
│   │   ├── Electromagnetism/
│   │   ├── Fluids/
│   │   ├── Thermodynamics/
│   │   └── Optics/
│   │
│   ├── Renderer/                    Camera, shaders, meshes, textures, rasterizer, ray tracer
│   │   └── shaders/                  *.vert, *.frag, embedded at configure time
│   │
│   ├── UI/                          Dear ImGui panels, Dear ImPlot charts
│   │
│   └── Applications/                Runnable simulation programs
│       └── SolarSystem/
│
└── tests/
    ├── support/                    Test-only helpers, outside the engine
    ├── smoke/                      Build wiring: dependencies link, options took effect
    ├── unit/                       One module in isolation
    ├── integration/                Modules in combination
    ├── compile_fail/               Constructs that must not compile (CTest WILL_FAIL)
    └── e2e/                        Full application runs, headless
```

## Modules

| Module         | Contents                                                                                                                                                                                                                                             |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Core`         | Logging (spdlog behind a facade), timing (simulation and wall-clock), UUIDs, events, configuration                                                                                                                                                                                   |
| `Math`         | Vectors, matrices, quaternions, complex/dual numbers, tensors, statistics, interpolation, calculus, ODE interface and integrators (Euler, RK4, adaptive, symplectic)                                                                                 |
| `Units`        | Dimensioned quantities (scalar or vector) built from the SI's seven base dimensions: length, mass, time, velocity, acceleration, force, energy, temperature, electromagnetism, luminosity, and the constants that define the SI. Built on `Math` |
| `Platform`     | Window, GL context, and input, wrapping GLFW                                                                                                                                                                                                         |
| `Compute`      | Backend `Physics` dispatches to: a CPU reference implementation plus GPU acceleration (OpenGL compute shaders, CUDA, Vulkan)                                                                                                                         |
| `Physics`      | Mechanics; relativistic spacetime (Minkowski, Schwarzschild, Kerr, FLRW) with a geodesic solver; gravity (Newtonian, post-Newtonian, Barnes-Hut summation); electromagnetism; fluids; thermodynamics; optics (propagation, lensing, frequency shift) |
| `Renderer`     | Camera and controllers, shaders, instanced meshes, textures, immediate-mode debug drawing and text labels, and both a forward rasterizer and a fragment-shader ray tracer                                                                          |
| `UI`           | Dear ImGui panels bound to plain references, Dear ImPlot charts, a stats overlay                                                                                                                                                                    |
| `Applications` | Runnable simulation programs built on the engine                                                                                                                                                                                                     |

## Documentation

Each library module under `src/` has its own `README.md` describing its
interface, dependencies, and the derivations behind it: the authoritative
reference for how the engine works. `docs/` holds consumer-facing
documentation: tutorials and conceptual explanations for building a
simulation in `Applications/`, cross-linking down into the module READMEs
rather than restating them.

## License

MIT. See [LICENSE](LICENSE).
