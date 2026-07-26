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

- C++20 compiler with `std::format` (GCC 13+, Clang 17+, or MSVC 19.29+)
- CMake 3.20+
- GPU and driver with OpenGL 4.3+ — for the visualizer and the OpenGL compute
  backend; the simulation core and its tests run CPU-only
- CUDA Toolkit and/or Vulkan SDK (optional, for those GPU compute backends)

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

Unit tests cover modules in isolation — a vector rotation, an integrator's order
of accuracy, a units dimension check. Integration tests cover combinations — a
symplectic integrator with Newtonian gravity holding a stable orbit, or spacetime
with optics reproducing a known deflection angle. End-to-end tests run whole
applications headless and assert physical invariants such as energy and momentum
conservation.

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

```
ysq/
├── CMakeLists.txt
├── README.md
├── LICENSE
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
│   │   ├── Logger.hpp              Thin wrapper over spdlog
│   │   ├── Timer.hpp
│   │   ├── Clock.hpp               Simulation time vs. wall-clock
│   │   ├── UUID.hpp
│   │   ├── Event.hpp
│   │   └── Config.hpp
│   │
│   ├── Math/
│   │   ├── README.md
│   │   ├── CMakeLists.txt
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
    ├── unit/                       Isolated module tests
    │   ├── CMakeLists.txt
    │   ├── math_vector.cpp
    │   ├── math_integrators.cpp
    │   ├── units_dimensions.cpp
    │   ├── physics_gravity.cpp
    │   ├── spacetime_geodesic.cpp
    │   └── optics_frequencyshift.cpp
    ├── integration/                Cross-module behavior
    │   ├── CMakeLists.txt
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
| `Core`         | Logging, timing (sim and wall-clock), UUIDs, events, configuration                                                                                                                                                                                   |
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
