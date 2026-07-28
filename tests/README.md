# Tests

CTest + GoogleTest. Built by default; disable with `-DYSQ_BUILD_TESTS=OFF`.

```sh
ctest --test-dir build
ctest --test-dir build --output-on-failure -R CoreClock
```

One executable per test file, registered with `gtest_discover_tests`, so each
`TEST()` case appears in CTest under its own name. Discovery runs in
`PRE_TEST` mode, listing the cases when CTest starts rather than as each binary
finishes linking: the POST_BUILD default runs those listings concurrently with
the rest of a parallel build, where they contend for the machine and, on CMake
4.2 through 4.4.0, race each other for a shared test-list file.

## Layout

| Directory      | Scope                                                            |
| -------------- | ---------------------------------------------------------------- |
| `smoke/`       | The build itself: dependencies link, options took effect          |
| `unit/`        | One module in isolation                                           |
| `integration/` | Modules in combination                                            |
| `compile_fail/`| Constructs that must **not** compile                              |
| `e2e/`         | Whole applications, headless, asserting physical invariants       |
| `support/`     | Test-only helpers, header-only, outside the engine                |

`e2e/` arrives with the applications it exercises. `integration/` holds
`core_runtime.cpp`, which drives `Config`, `Logger`, `Clock`, `Timer`, `Event`
and `UUID` through one fixed-step run; `math_kepler.cpp`, which puts six `Math`
headers on a two-body orbit and checks the conserved quantities and Kepler's
third law come out; `units_kinematics.cpp`, which takes a dimensioned scenario
across the boundary into a `Math` integrator and back; and
`platform_context.cpp`, which opens a real OpenGL context and uses it.

`support/MathApprox.hpp` supplies `EXPECT_VEC_NEAR`, `EXPECT_MAT_APPROX` and
friends. `EXPECT_NEAR` is the wrong tool for a `Math` value twice over: it
compares one scalar, so a matrix has to be checked element by element with a
failure message that names neither matrix, and its tolerance is absolute, which
is meaningless once magnitudes leave the neighbourhood of 1. Link it with
`LIBS ysq::TestSupport`.

`support/UnitsApprox.hpp` does the same for dimensioned quantities, with the
tolerance itself a quantity, so `EXPECT_QUANTITY_NEAR(orbit, analytic, 1.0_km)`
reads as it should and a length tolerance on a mass comparison does not compile.
It is a separate header rather than part of `MathApprox.hpp` so that a test
exercising only `Math` does not acquire a dependency on `Units`; `TestSupport`
supplies the `tests/` include root, and the test links `ysq::Units` itself.

Everything here runs CPU-only and needs no GPU, no window and no display. The
CPU compute backend is the reference implementation, so correctness is testable
on any machine; GPU backends are validated against it within tolerance rather
than for exact equality, since they generally run `float32` where the reference
runs `float64`.

The one test that wants a real OpenGL context is `integration/platform_context.cpp`,
and it does not need a display either. See the platform tests below.

## Smoke tests

These exist because a dependency that is checked out, configured and compiled
can still be wired up wrong, and the failure would otherwise surface much later
in something unrelated.

- `spdlog_format.cpp` asserts spdlog was built with `SPDLOG_USE_STD_FORMAT`, so
  the project keeps one formatting implementation and no bundled fmt.
- `graphics_link.cpp` links GLAD, GLFW and Dear ImGui including its backends,
  and calls only entry points that need no context or display. Built only when
  `YSQ_BUILD_GRAPHICS=ON`.
- `math_strict_warnings.cpp` compiles every `Math` header under
  `ysq::warnings_strict` and explicitly instantiates every template for both
  `float` and `double`. `Math` is an INTERFACE library and cannot carry those
  flags itself without pushing them onto `Renderer`, `UI` and `Applications`,
  which `docs/architecture.md` rules out, so the check lives here. The explicit
  instantiations are the point: an uninstantiated template is barely checked,
  and `-Wdouble-promotion` has nothing to say above single precision.
- `units_strict_warnings.cpp` is the same check for `Units`, which is INTERFACE
  for the same reason and so cannot carry the flags either.

## Platform tests

`Platform` is the only module whose subject matter is partly the machine, so its
tests are split by what they need.

`unit/platform_input.cpp` needs nothing at all. `InputState` takes events and
answers questions and knows nothing about GLFW, so the test plays the part of
the window: key mapping round-trips, the press/hold/release edges, auto-repeat,
focus loss, cursor deltas and scroll accumulation are all exercised with no
window in existence. It links `glfw` alongside the module, deliberately: the
mapping is only worth asserting against the codes it actually has to match.

`unit/platform_window.cpp` runs on the `Null` backend, which is compiled in
everywhere and needs no display, and stops before creating a context. It covers
the reference counting and the settings a window refuses, which are checked
before the windowing system is asked anything and so fail identically on every
machine.

`integration/platform_context.cpp` opens a real context, and whether one can
exist is a property of the machine rather than of the code. Each test tries the
native backend, falls back to `Null` (whose contexts come from OSMesa in
software), and skips if neither yields one.

**A file of tests that always skip is a file that tests nothing**, so
`-DYSQ_REQUIRE_HEADLESS_GL=ON` turns every skip in it into a failure:

```sh
sudo apt install libosmesa6
cmake -B build -DYSQ_REQUIRE_HEADLESS_GL=ON
cmake --build build && ctest --test-dir build -R PlatformContext
```

CI sets it on the Linux `graphics=ON` job, the one runner where OSMesa is
installed and a context is therefore guaranteed. Everywhere else the skip stands,
because there is nothing the test could do about a machine with no rasteriser.

It is a build option rather than an environment variable read at run time
because MSVC deprecates `std::getenv` and the project builds with warnings as
errors, so reading one would cost a suppression or a per-compiler wrapper for a
value that cannot change during a run anyway.

Two things there are deliberately not covered. Nothing delivers a synthetic key
event, because no backend available here can send one, so the hop from the GLFW
callback into `InputState` is untested; `EveryInputCallbackIsRegistered` covers
the failure that would otherwise be invisible, which is a callback never being
installed. And `setCursorMode` and `contentScale` are called but not asserted on,
since the `Null` backend has neither a cursor nor a monitor to answer for them.

## Compile-failure tests

`Units` guarantees that adding a distance to a mass will not build. A suite made
only of programs that compile cannot check a guarantee like that.

The primary check is still an ordinary test: `unit/units_dimensions.cpp`
evaluates named concepts with `static_assert`, which is free and portable.
**Write them as concepts, never as a bare `requires` at namespace scope.**
Outside a template there is no substitution, so Clang reports an invalid
requirement as a hard error instead of evaluating the requires-expression to
false, and the file fails to compile. GCC is laxer, so the obvious spelling
passes locally and fails in CI.

`compile_fail/` is the second line: real targets, `EXCLUDE_FROM_ALL`, built by a
nested `cmake --build` inside a CTest test marked `WILL_FAIL`. Two properties of
that mechanism matter when adding a case.

`WILL_FAIL` only checks for a nonzero exit code, so a source with a typo in it
also "passes". Nothing there is trustworthy alone. Every case is paired with its
positive form in `units_dimensions.cpp`, and that pairing is the guarantee: the
positive proves the construct is otherwise valid, the negative proves the
dimension is what rejects it. Matching the diagnostic text instead was
considered and rejected, since GCC, Clang and MSVC word these differently.

Each test starts a build in the shared tree, so they carry `RESOURCE_LOCK` to
keep `ctest -j` from running several at once. Turn the category off with
`-DYSQ_BUILD_COMPILE_FAIL_TESTS=OFF`.

## Adding a test

```cmake
ysq_add_test(test_math_vector SOURCES math_vector.cpp
    LIBS ysq::Math ysq::TestSupport)
```

`ysq_add_test` is defined in `tests/CMakeLists.txt` and links `GTest::gtest_main`
and `ysq::warnings` for you. Pass `STRICT` to swap in `ysq::warnings_strict`
instead; only a target that exists to compile the engine core's templates under
the conversion warnings should ask for it.

## Choosing a tolerance

Two rules, both learned the hard way.

**A tolerance has to sit above the accuracy the inputs can carry.** Recovering
a quantity of size `q` from operands of size 1 leaves about `epsilon / q` of
relative accuracy, because that is how many bits of the difference survive.
Asking for better cannot be met by any arrangement of the arithmetic, and a
test that asks for it passes or fails on rounding luck.

**If a tolerance moves when floating-point contraction does, it is below that
floor.** Fused multiply-add is baseline on arm64 and clang contracts into it by
default, so a cancellation-prone expression keeps extra bits there that x86-64
without an explicit `-march` does not have. A macOS-only pass is the usual
symptom. To check a suspect tolerance the way the other CI platforms will see
it:

```sh
cmake -B build-nofma -DCMAKE_CXX_FLAGS=-ffp-contract=off
cmake --build build-nofma && ctest --test-dir build-nofma
```

## Testing a numerical method

`unit/math_integrators.cpp` is the pattern to follow for anything whose
correctness is a rate rather than a value. A wrong Butcher tableau does not
crash and does not obviously misbehave: it produces a method that still
converges, just more slowly than advertised. So the test measures the observed
order from how the error falls under refinement, takes the median of the
consecutive ratios rather than a single pair, and refuses to report a number at
all if any error has fallen into rounding noise or has not yet reached the
asymptotic regime. Both of those guards have already caught a bad measurement
that a fixed tolerance would have waved through.
