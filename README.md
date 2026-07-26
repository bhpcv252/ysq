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
- Real-time OpenGL rendering with ImGui controls, plus ray tracing
- GPU compute backends: OpenGL compute shaders, CUDA, Vulkan
- Example applications: solar system, binary stars, galaxy collision, black hole,
  light deflection

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
backend does not. See [docs/architecture.md](docs/architecture.md) for how
backend selection and fallback work per platform.

### Linux build dependencies

GLFW builds both the X11 and Wayland backends, and needs their development
packages present at build time. On Debian and Ubuntu:

```sh
sudo apt install xorg-dev libwayland-dev libwayland-bin libxkbcommon-dev
```

Windows and macOS need nothing beyond a compiler and CMake. A headless build
(`-DYSQ_BUILD_GRAPHICS=OFF`) needs none of these on any platform.

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
| `YSQ_BUILD_GRAPHICS`     | `ON`    | Build against GLFW, GLAD and Dear ImGui. `OFF` drops them entirely, for headless and CI builds. |
| `YSQ_WARNINGS_AS_ERRORS` | `OFF`   | Treat warnings as errors. CI builds with this on. |

```sh
cmake -B build -DYSQ_BUILD_GRAPHICS=OFF   # headless: no graphics dependencies
```

## Running

Each program under `Applications/` builds to its own executable in `build/bin/`:

```sh
./build/bin/solar-system
```

Available: `solar-system`, `binary-stars`, `galaxy-collision`, `black-hole`,
`light-deflection`.

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

Everything runs CPU-only and needs no GPU, no window and no display.

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

The tree below is the target layout. Modules appear as they are implemented, so
not all of it exists yet.

```
ysq/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── .clang-format
├── .github/workflows/ci.yml
├── cmake/
│   └── YsqWarnings.cmake       Shared warning sets
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── math.md
│   ├── physics.md
│   └── rendering.md
│
├── third_party/                    Vendored dependencies
│   ├── README.md
│   ├── glfw/                        submodule
│   ├── glad/                        generated loader, committed
│   ├── imgui/                       submodule
│   ├── spdlog/                      submodule
│   └── googletest/                  submodule (tests)
│
├── src/
│   ├── CMakeLists.txt
│   │
│   ├── Core/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
│   │   ├── Version.hpp.in          Version, generated from the CMake project version
│   │   ├── Logger.hpp              Facade over spdlog; spdlog stays out of the header
│   │   ├── Timer.hpp               Wall-clock stopwatch
│   │   ├── Clock.hpp               Simulation time: fixed steps, time scale, pause
│   │   ├── UUID.hpp
│   │   ├── Event.hpp               Type-keyed event bus
│   │   └── Config.hpp
│   │
│   ├── Math/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
│   │   ├── Scalar.hpp              The Numeric concept, constants, tolerances
│   │   ├── Vector2.hpp
│   │   ├── Vector3.hpp
│   │   ├── Vector4.hpp
│   │   ├── Matrix2.hpp
│   │   ├── Matrix3.hpp
│   │   ├── Matrix4.hpp
│   │   ├── Quaternion.hpp
│   │   ├── Complex.hpp
│   │   ├── Dual.hpp                Dual numbers (automatic differentiation)
│   │   ├── Tensor.hpp
│   │   ├── Statistics.hpp
│   │   ├── Interpolation.hpp
│   │   ├── Calculus.hpp            Numerical differentiation and integration
│   │   ├── ODE.hpp                 ODE system interface: dy/dt = f(t, y)
│   │   ├── CoordinateSystems.hpp
│   │   ├── Format.hpp              std::formatter for every Math type
│   │   └── Integrators/            Methods that advance an ODE
│   │       ├── Euler.hpp
│   │       ├── RK4.hpp
│   │       ├── Adaptive.hpp        Adaptive step size
│   │       └── Symplectic.hpp      Verlet / leapfrog (energy-conserving)
│   │
│   ├── Units/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
│   │   ├── Unit.hpp                Dimensioned-quantity machinery (scalar or vector)
│   │   ├── Length.hpp
│   │   ├── Mass.hpp
│   │   ├── Time.hpp
│   │   ├── Velocity.hpp
│   │   ├── Acceleration.hpp
│   │   ├── Force.hpp
│   │   ├── Energy.hpp
│   │   ├── Temperature.hpp
│   │   └── Luminosity.hpp
│   │
│   ├── Platform/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
│   │   ├── Window.hpp              Window and GL context (GLFW)
│   │   └── Input.hpp
│   │
│   ├── Compute/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
│   │   ├── ComputeBackend.hpp      Backend interface
│   │   ├── CPU/                    Reference backend (no GPU required)
│   │   │   └── CpuBackend.hpp
│   │   ├── OpenGL/
│   │   │   ├── ComputeShader.hpp
│   │   │   └── shaders/            *.comp
│   │   ├── CUDA/
│   │   │   └── kernels/            *.cu
│   │   └── Vulkan/
│   │       ├── VulkanCompute.hpp
│   │       └── shaders/            *.comp (SPIR-V)
│   │
│   ├── Physics/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
│   │   ├── Body.hpp                Matter state: mass, charge, position, momentum
│   │   │
│   │   ├── Mechanics/
│   │   │   ├── Frame.hpp           Reference frames
│   │   │   ├── Kinematics.hpp      Worldlines, proper time (Newtonian = low-v limit)
│   │   │   └── Dynamics.hpp        Equations of motion
│   │   │
│   │   ├── Spacetime/
│   │   │   ├── Metric.hpp          Abstract metric
│   │   │   ├── Minkowski.hpp       Flat spacetime (special relativity)
│   │   │   ├── Schwarzschild.hpp   Non-rotating mass
│   │   │   ├── Kerr.hpp            Rotating mass
│   │   │   ├── FLRW.hpp            Expanding universe (cosmological)
│   │   │   └── Geodesic.hpp        Timelike and null geodesic solver
│   │   │
│   │   ├── Gravity/
│   │   │   ├── Newtonian.hpp       Weak-field limit; dynamical many-body gravity
│   │   │   ├── PostNewtonian.hpp   Relativistic corrections
│   │   │   └── BarnesHut.hpp       Tree-based force summation
│   │   │
│   │   ├── Electromagnetism/
│   │   │   ├── Field.hpp           E and B fields
│   │   │   ├── Maxwell.hpp         Field evolution
│   │   │   └── Lorentz.hpp         Force on charges
│   │   │
│   │   ├── Fluids/
│   │   │   └── FluidDynamics.hpp
│   │   │
│   │   ├── Thermodynamics/
│   │   │   └── Thermodynamics.hpp
│   │   │
│   │   └── Optics/
│   │       ├── Propagation.hpp     Light as null geodesics (uses Spacetime)
│   │       ├── Lensing.hpp         Gravitational lensing
│   │       └── FrequencyShift.hpp  Doppler, gravitational, and cosmological shift
│   │
│   ├── Renderer/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
│   │   ├── Renderer.hpp
│   │   ├── Camera.hpp
│   │   ├── Shader.hpp
│   │   ├── Texture.hpp
│   │   ├── Mesh.hpp
│   │   ├── RayTracer.hpp           Ray-traced rendering; light physics in Physics/Optics
│   │   └── shaders/                *.vert, *.frag
│   │
│   ├── UI/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
│   │   └── ImGuiLayer.hpp
│   │
│   └── Applications/
│       ├── README.md
│       ├── CMakeLists.txt
│       ├── SolarSystem/
│       │   ├── CMakeLists.txt
│       │   └── main.cpp
│       ├── BinaryStars/
│       │   ├── CMakeLists.txt
│       │   └── main.cpp
│       ├── GalaxyCollision/
│       │   ├── CMakeLists.txt
│       │   └── main.cpp
│       ├── BlackHole/
│       │   ├── CMakeLists.txt
│       │   └── main.cpp
│       └── LightDeflection/
│           ├── CMakeLists.txt
│           └── main.cpp
└── tests/
    ├── README.md
    ├── CMakeLists.txt
    ├── support/                    Test-only helpers, outside the engine
    │   └── MathApprox.hpp          Approximate comparison and printing for Math values
    ├── smoke/                      Build wiring: dependencies link, options took effect
    │   ├── CMakeLists.txt
    │   ├── spdlog_format.cpp
    │   ├── graphics_link.cpp
    │   └── math_strict_warnings.cpp  Math's templates under the strict warning set
    ├── unit/                       Isolated module tests
    │   ├── CMakeLists.txt
    │   ├── core_version.cpp
    │   ├── core_logger.cpp
    │   ├── core_timer.cpp
    │   ├── core_clock.cpp
    │   ├── core_uuid.cpp
    │   ├── core_event.cpp
    │   ├── core_config.cpp
    │   ├── math_vector.cpp
    │   ├── math_matrix.cpp
    │   ├── math_quaternion.cpp
    │   ├── math_complex.cpp
    │   ├── math_dual.cpp
    │   ├── math_tensor.cpp
    │   ├── math_statistics.cpp
    │   ├── math_interpolation.cpp
    │   ├── math_calculus.cpp
    │   ├── math_coordinates.cpp
    │   ├── math_ode.cpp
    │   ├── math_integrators.cpp    Observed order of every method
    │   ├── units_dimensions.cpp
    │   ├── physics_gravity.cpp
    │   ├── spacetime_geodesic.cpp
    │   └── optics_frequencyshift.cpp
    ├── integration/                Cross-module behavior
    │   ├── CMakeLists.txt
    │   ├── core_runtime.cpp        Config + Logger + Clock + Timer + Event + UUID
    │   ├── math_kepler.cpp         Integrators + vectors + coordinates + statistics
    │   ├── orbit_stability.cpp     Gravity + symplectic integrator
    │   ├── lensing_deflection.cpp  Spacetime + optics
    │   └── nbody_energy.cpp        Barnes-Hut + conservation
    └── e2e/                        Full application runs (headless)
        ├── CMakeLists.txt
        ├── solar_system.cpp        Energy & momentum conservation
        └── black_hole.cpp          Photon paths vs. analytic
```

## Modules

| Module         | Contents                                                                                                                                                                                                                                             |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Core`         | Logging (spdlog behind a facade), timing (simulation and wall-clock), UUIDs, events, configuration                                                                                                                                                                                   |
| `Math`         | Vectors, matrices, quaternions, complex/dual numbers, tensors, statistics, interpolation, calculus, ODE interface and integrators (Euler, RK4, adaptive, symplectic)                                                                                 |
| `Units`        | Length, mass, time, velocity, acceleration, force, energy, temperature, luminosity as dimensioned quantities (scalar or vector), built on `Math`                                                                                                     |
| `Platform`     | Window, GL context, and input, wrapping GLFW                                                                                                                                                                                                         |
| `Compute`      | Backend `Physics` dispatches to: a CPU reference implementation plus GPU acceleration (OpenGL compute shaders, CUDA, Vulkan)                                                                                                                         |
| `Physics`      | Mechanics; relativistic spacetime (Minkowski, Schwarzschild, Kerr, FLRW) with a geodesic solver; gravity (Newtonian, post-Newtonian, Barnes-Hut summation); electromagnetism; fluids; thermodynamics; optics (propagation, lensing, frequency shift) |
| `Renderer`     | OpenGL rendering (camera, shaders, textures, meshes) and ray tracing                                                                                                                                                                                 |
| `UI`           | Dear ImGui panels                                                                                                                                                                                                                                    |
| `Applications` | Runnable simulation programs built on the engine                                                                                                                                                                                                     |

## Documentation

Each library module under `src/` has its own `README.md` describing its interface
and dependencies. Longer notes and derivations are in `docs/`.

## License

MIT. See [LICENSE](LICENSE).
