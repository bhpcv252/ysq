# Cookbook

Short answers to specific questions. If you're learning a concept for the
first time, the per-module pages linked from [docs/README.md](README.md)
are the better start; come back here once you know roughly what you want
and just need the shape of it.

**How do I switch which integrator my simulation uses?**
Change the stepper type; nothing else moves, as long as you're switching
within one family (see the table in
[docs/math/integrators.md](math/integrators.md)). `VelocityVerletStepper<S>`
to `PefrlStepper<S>` for a more accurate symplectic method, or
`Rk4Stepper<S>` to `DormandPrince54Stepper<S>` run through
`integrateAdaptive` if you want accuracy rather than step size as the
knob.

**How do I run without a window, for a test or a script?**
Skip `Platform`/`Renderer`/`UI` entirely: your scenario code (see
[docs/applications.md](applications.md)) has no dependency on any of them,
so constructing bodies and stepping them forward works with no window, no
context, nothing graphical. That's exactly what `tests/e2e/` does.

**How do I bind a UI control to a simulation parameter?**
`Panel::slider`/`checkbox`/etc. take a plain reference and read/write it
directly, no separate UI state to keep in sync:
```cpp
controls.slider("Time scale", timeScale, 0.0f, 10.0f);
```
See [docs/ui.md](ui.md).

**How do I force a specific compute backend, or check what's available?**
`ysq::selectComputeBackend(kind)` skips probing and returns exactly that
backend, or `nullptr` if it's not available; `ysq::computeBackendAvailable(kind)`
answers without selecting one. See [docs/compute.md](compute.md).

**How do I pick a gravity model?**
See the decision table in
[Tutorial 3](tutorials/03-choosing-a-gravity-model.md#which-rung-when):
roughly, `NewtonianField` for a handful of exact bodies,
`BarnesHutTree` for many approximate ones, the 1PN correction for one
relativistic orbit, and a `Schwarzschild`/`Kerr` metric with the geodesic
solver for light or a test particle near a compact object.

**How do I know my simulation is actually correct, not just running?**
Watch a conserved quantity, usually total energy (kinetic plus potential;
see [Tutorial 1](tutorials/01-your-first-simulation.md)). It should stay
flat, or for a symplectic stepper, oscillate in a narrow band rather than
drift; see
[Order isn't the whole story](math/integrators.md#order-isnt-the-whole-story).
A steadily growing or shrinking energy plot means something's wrong with
the setup, the step size, or the integrator choice, not with the physics.

**How do I start a brand new `Application`?**
See [docs/applications.md](applications.md#using-it-starting-a-new-application)
for the full checklist: new directory, `Scenario.hpp`/`.cpp` with no
`Platform`/`Renderer`/`UI` dependency, `CMakeLists.txt`, `main.cpp`.

**How do I add a debug line, arrow, or an orbit trail?**
`renderer.debugDraw().line(a, b)`, `.arrow(origin, direction)`, or
`.axes()`; see [docs/renderer.md](renderer.md).

**How do I add a text label to something in the scene?**
`renderer.debugDraw().text(position, "label")` for one that always faces
the camera, `.textFixed(position, right, up, "label")` for one with a
fixed orientation that skews with perspective like real geometry. See
[docs/renderer.md](renderer.md).

**How do I add a new test?**
```cmake
ysq_add_test(test_my_thing SOURCES my_thing.cpp LIBS ysq::Physics ysq::TestSupport)
```
`ysq_add_test` links `GTest::gtest_main` and the standard warning set for
you. See `tests/README.md` in the repository for the full convention,
including `EXPECT_QUANTITY_NEAR` for comparing dimensioned values.

**How do I format my code before committing?**
```sh
git ls-files '*.hpp' '*.cpp' '*.h' '*.c' | grep -v '^third_party/' \
    | tr '\n' '\0' | xargs -0 clang-format -i
```
CI checks this with the same file list; see the root `README.md`'s
Formatting section for the pinned `clang-format` version.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+cookbook)
and let us know.
