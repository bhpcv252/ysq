# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

**YSQ (Yotta Spacetime Quantities)** — a C++ scientific simulation engine, rendered
with OpenGL. The engine encodes the math and physics that define how things behave;
applications under `Applications/` set up specific scenarios and run them against
that shared description of reality.

- Language: C++20 (requires `std::format`)
- Build: CMake 3.20+
- Tests: CTest + GoogleTest
- Logging: spdlog, behind the `Core/Logger` facade (configured for `std::format`)
- Graphics: OpenGL, with GLFW (window/context), GLAD (loader), Dear ImGui (UI)
- Dependencies vendored under `third_party/` (GLAD is generated and committed; the
  rest are submodules)

Full detail is in `README.md`. Longer design notes and derivations are in `docs/`.
Read them before working on a module.

## Architecture

Dependencies flow one way. Nothing lower depends on anything higher.

- **Base / system:** `Core`, `Math`, `Units`, `Platform` (window, GL context, input)
- **`Compute`** builds on the base — CPU reference backend plus GPU backends
  (OpenGL compute, CUDA, Vulkan)
- **`Physics`** builds on `Compute`, falling back to the CPU backend with no GPU
- **`Renderer`, `UI`** — presentation layer
- **`Applications`** — on top, consuming the engine

The simulation core and tests run CPU-only; no GPU is required to build or test.

`Physics` is organized by theory: `Mechanics`, `Spacetime` (Minkowski,
Schwarzschild, Kerr, FLRW + geodesic solver), `Gravity` (Newtonian, post-Newtonian,
Barnes-Hut), `Electromagnetism`, `Fluids`, `Thermodynamics`, `Optics`. Gravity is a
ladder of approximations to general relativity; spacetime is the core abstraction,
so light propagation, lensing, and frequency shift are all null geodesics through a
metric.

Keep the engine domain-neutral. Math and physics concepts belong in the engine;
scenario setup and results belong in `Applications/`. Do not let a lower layer
depend on a higher one.

## Workflow

**Plan first, every task. No exceptions.**

1. Inspect the relevant parts of the project.
2. Present an implementation plan and wait for the human to confirm it before
   writing any code.
3. The plan must cover: the change itself, tests to add or update, and docs to
   update.

Human confirmation is required at the planning step for every task. Do not start
implementing until the plan is approved.

**Never assume.** Do not guess requirements, intent, naming, file locations, or
design decisions. If anything is unspecified or unclear, ask the human. Propose;
do not presume.

**Every task includes** implementation, tests (unit / integration / e2e as
appropriate), and updating any affected docs (`README.md`, module `README.md`,
`docs/`). After implementing, run the build and the full test suite yourself,
confirm both pass, and fix anything that fails before reporting back. Run the build
and tests yourself — don't hand the commands back to the human.

## Build and test

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

Tests build by default; disable with `-DYSQ_BUILD_TESTS=OFF`.

Run an application:

```sh
./build/bin/solar-system
```

## Code style

- Comments: short and on point. Say why, not what. No narration.
- Match the conventions of the module you're editing.
- Headers are `.hpp` with a matching `.cpp`; header-only where template-only.
- Preserve the one-way dependency flow. New physics/math goes in the right engine
  module, never in `Applications/`.
