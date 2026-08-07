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

Full detail is in `README.md`. Each module's derivations and design notes live
in its own `src/<Module>/README.md`; read the relevant one before working on a
module. `docs/` holds consumer-facing documentation for building simulations
in `Applications/`, not internal design notes.

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

**Engine vs. phenomena.** The engine encodes only general laws: conservation
laws, force laws, wave/ray propagation, scattering, metric geodesics, and so
on. It must never encode a named real-world phenomenon (an eclipse, a
rainbow, a tide) as its own type or function. Every such phenomenon is an
emergent result of applying the engine's general laws to a specific
scenario's bodies, geometry, and initial conditions, and that composition
belongs entirely in `Applications/`. Before adding anything under `src/`
outside `Applications/`, state whether it is general outside this one
scenario, or only makes sense because of what this scenario is trying to
show. If the latter, it belongs in `Applications/`.

**Engine completeness.** The goal is a complete simulation engine other
people build arbitrary simulations on top of, not the minimum one
application in front of you needs. Judge whether something belongs in the
engine by whether it's a general law, method, or piece of infrastructure any
consumer could need — never by whether the current task's application
happens to use it. If the engine is missing a general capability, that's a
reason to build it there, not a reason to scope it down to what the
immediate app requires.

## Choosing between Math, Physics, and Applications

### Mathematics module

**Mission**

The Mathematics module exists to provide universal computational tools.

It does not describe the real world.

Everything in Math should remain correct even if the universe had
completely different laws of physics. For example, vectors would still be
vectors, matrices would still multiply, and numerical integration would
still approximate differential equations.

The Mathematics module answers one question: "How do we compute?"

It never answers:
- What is gravity?
- Why do planets move?
- Why does light bend?

**What belongs in Mathematics**

1. **Abstract mathematical structures.** Objects defined by mathematics
   itself. Examples: scalar, complex number, vector, matrix, tensor,
   quaternion, polynomial, function, graph, set. These have no physical
   meaning.
2. **Geometry.** Pure geometric operations. Examples: distance, angle,
   rotation, projection, plane, sphere, triangle, polygon, bounding box,
   convex hull. Notice that geometry says nothing about matter.
3. **Numerical methods.** General computational algorithms. Examples:
   Runge-Kutta, Euler integration, Newton-Raphson, gradient descent, binary
   search, root finding, optimization, least squares, numerical
   differentiation, numerical integration. These algorithms could be used
   in economics just as easily as physics.
4. **Probability & statistics.** Everything that is mathematical rather
   than experimental. Examples: random variables, probability
   distributions, Monte Carlo, statistical estimators, covariance, PCA.
5. **Mathematical algorithms.** Examples: FFT, Delaunay triangulation,
   KD-tree, spline interpolation, Bezier curves, Voronoi diagrams.

**What NEVER belongs in Mathematics**

Anything with physical meaning. Bad examples: gravity, force, mass,
velocity, temperature, energy, electric field, photon, wave, planet,
spring, pressure. Because these only exist after we've decided we're
describing the physical universe.

### Physics module

**Mission**

Physics exists to describe how the universe behaves.

Physics converts mathematics into models of reality.

Physics answers "How does nature work?"

It does not answer "Which planet are we simulating?"

**What belongs in Physics**

1. **Physical quantities.** Anything that has physical units. Examples:
   length, mass, time, charge, energy, momentum, force, power, pressure,
   density, temperature, angular velocity, acceleration, electric field,
   magnetic field, stress, strain. Notice these are physical concepts, not
   objects.
2. **Physical laws.** Universal relationships. Examples: Newton's laws,
   the law of gravitation, Maxwell's equations, Snell's law, Hooke's law,
   the ideal gas law, the Bernoulli equation, the Schrodinger equation,
   Navier-Stokes, Kepler's laws. These should work regardless of the
   application.
3. **Physical models.** Reusable implementations of physics. Examples:
   gravity model, collision model, spring model, rigid body model,
   particle model, fluid model, thermal model, orbital mechanics, optical
   ray model, atmospheric model. Each is reusable.
4. **Physical solvers.** These combine mathematical algorithms with
   physical laws. Examples: RigidBodySolver, OrbitalSolver, FluidSolver,
   ConstraintSolver, CollisionSolver. The solver knows physics. The RK4
   integrator it uses belongs in Math.

**What NEVER belongs in Physics**

Specific entities. Not: Earth, Moon, Mars, a rocket, a building, a bridge,
the pendulum (the specific pendulum in your demo), a solar system, a
player, a vehicle. These are application concepts.

### Exception: named-body units of measurement in Units

`Units` has constants like `units::solarMass`, `units::earthMass`,
`units::solarRadius`, `units::solarLuminosity`, and
`constants::nominalSolarMassParameter`/`nominalEarthMassParameter`, plus
literal suffixes built on them (`_Msun`, `_Mearth`, `_Lsun`). These name the
Sun and Earth, which looks at first glance like the "no named real-world
entities in the engine" rule, but it is not a violation. Do not move these
to Applications and do not flag new ones like them as violations without
applying the test below first.

**The test: is this a unit, or is this a scenario object?**

A unit answers "how much" and stays a single fixed scalar forever. A solar
mass is exactly as much a unit as an astronomical unit, a parsec, or a
light-year, already accepted in this same module: "this exoplanet is 0.3
Earth radii" and "this star is 5 solar masses" are standard, professional
usage, not a description of the Sun or Earth as objects. That's why these
sit next to `astronomicalUnit`/`parsec`/`lightYear` rather than being an
inconsistency with them.

A scenario object answers "what is this thing and what does it do":
position, velocity, orbit, mass as a simulated body, render color, a row
in a catalog. That's Earth or the Sun as an actual thing in a running
simulation, which is squarely Applications content, exactly like
`LunarEclipse/Scenario.cpp`'s `Body earth` or `BodyCatalog`'s CSV rows.

**Before adding a new named-body constant to Units, both must hold:**

1. It is a real, standardized scientific unit of measurement, fixed by a
   named authority (an IAU resolution, a CODATA value, or equivalent), not
   something you're defining for one scenario's convenience.
2. It functions as a conversion factor or scale used to describe *other*
   things, not as the state of that body itself.

If either fails, it's Applications content, not Units. A one-off
convenience constant for a specific demo (say, a made-up "Jupiter radius"
you invented for a single scenario, or an actual simulated body's live
mass/radius/position) never qualifies, no matter how real the underlying
data is.

### Applications module

**Mission**

Applications answer "What exactly are we trying to simulate?"

Applications choose:
- which physical models to use
- which objects exist
- initial conditions
- scenario logic

Applications do not invent mathematics.

Applications do not invent physics.

They assemble them.

**What belongs in Applications**

1. **Real entities.** Examples: Earth, Moon, Sun, ISS, a rocket, Mars, a
   black hole, a satellite, a hydrogen atom, a bridge, a wind turbine.
   Everything with an identity belongs here.
2. **Scenario definitions.** Examples: lunar eclipse, solar eclipse,
   Earth-Moon system, three-body problem, projectile demo, double-pendulum
   demo, wave tank. Every scenario is an application.
3. **Initial conditions.** Examples: Earth's mass, the Moon's position,
   launch velocity, rocket fuel, temperature, initial pressure. These are
   not laws. They are inputs.
4. **Simulation orchestration.** Example: create Earth, create Moon,
   create Sun, attach gravity, run the orbital solver, advance time. This
   belongs only here.

### The ultimate decision rules

These are the rules to actually use while coding, in order.

- **Rule 1.** Ask: could this exist without the physical universe?
  Yes → Mathematics. No → continue.
- **Rule 2.** Ask: is this a universal law of nature?
  Yes → Physics. No → continue.
- **Rule 3.** Ask: is this a specific thing, object, or scenario?
  Yes → Application.
- **Rule 4.** Ask: could every physics simulation reuse this?
  Yes → Physics.
- **Rule 5.** Ask: could software outside physics reuse this?
  Yes → Mathematics.
- **Rule 6.** Ask: does this describe "how to compute"?
  Yes → Mathematics.
- **Rule 7.** Ask: does this describe "how nature behaves"?
  Yes → Physics.
- **Rule 8.** Ask: does this describe "what exists in my simulation"?
  Yes → Application.

### A quick classification table

| If it represents... | Module |
|---|---|
| Abstract structures (vectors, matrices, tensors) | Math |
| Pure algorithms (sorting, integration, interpolation, optimization) | Math |
| Geometry without physical meaning | Math |
| Physical quantities (mass, force, energy, pressure) | Physics |
| Universal laws (gravity, electromagnetism, thermodynamics) | Physics |
| Reusable physical models and solvers | Physics |
| Named entities (Earth, Moon, Sun, Rocket, Bridge) | Application |
| Initial conditions and parameters for a scenario | Application |
| Simulation-specific orchestration | Application |

### Editing inside the engine is a blocking action

Consumers should never need to edit Math, Physics, or any other engine
module to build a simulation in Applications. If a task seems to require
it, that's a signal the engine is missing a feature it should already have,
not a shortcut to take. Before adding anything to an engine module:

1. Check whether the feature already exists elsewhere in the engine under a
   different name.
2. Check whether the need can be met by composing existing Math/Physics
   primitives inside Applications instead.
3. Only after both of the above come up empty, add the feature to the
   correct module using the rules above.

That is three explicit passes, not one glance. Do not skip straight to
editing Physics or Math because an application hit a wall.

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

**Never defer or mark anything "out of scope" unilaterally.** If part of a
task seems better left for later (too large, too risky, needs infrastructure
that doesn't exist yet, whatever the reason), do not just decide that and
move on. Say so explicitly, explain why, and ask the human whether it's
actually okay to defer. The human decides what's in and out of scope, not
the plan.

## Committing

**Never commit, amend, tag, or push. The human does that.** This holds even after
a task is finished and verified, and even when asked to "finish up" or "wrap up".

When the work is done, output exactly three things and nothing else:

1. A `cd` command to the repository root.
2. The commit message.
3. The PR description, filled in from `.github/pull_request_template.md`.

Always use that template. Keep every section filled in; if a section does not
apply, say so in one line rather than deleting the heading. No filler, no
restating the diff, no unnecessary blank lines.

Commit messages follow conventional commits with a scope, e.g.
`feat(physics): add Schwarzschild metric`, `chore(root): wire up CTest`.

**Never write Claude, Anthropic, or any AI attribution anywhere.** Not in commit
messages, not in `Co-Authored-By` trailers, not in PR descriptions, not in code
comments, not in docs.

## Build and test

```sh
cmake -B build
cmake --build build
```

Builds Release by default, and only the engine and applications, not tests.
Enable tests with `-DYSQ_BUILD_TESTS=ON`:

```sh
cmake -B build -DYSQ_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Run an application:

```sh
./build/bin/solar-system
```

## Code style

- **Comments.** A comment describes the code as it exists right now. It
  never describes a change, a diff, a fix, or the history of how the code
  got here. No "changed to...", "now does...", "previously...", "fixes
  issue with...". If you'd only understand the comment by knowing what the
  code used to look like, delete it.
- Keep comments short and on point. No paragraphs, no restating what the
  code already says line by line, no narration of what you're about to do.
- Before writing a comment, check whether the code needs it at all. If
  well-named identifiers and clear structure already make the behavior
  obvious, don't add one. Only comment where the code cannot explain
  itself: a non-obvious constraint, a subtle invariant, a workaround for a
  specific edge case.
- When a task changes code that already has a comment, treat the comment
  as needing the same scrutiny as new code: if it's still needed, rewrite
  it to describe the code as it now stands, not why it changed. If it's no
  longer needed, delete it.
- Treat this as a blocking check, not an afterthought: think about whether
  a comment is warranted, and what it should say about the current code,
  before writing it.
- Match the conventions of the module you're editing.
- Headers are `.hpp` with a matching `.cpp`; header-only where template-only.
- Preserve the one-way dependency flow. New physics/math goes in the right engine
  module, never in `Applications/`.
