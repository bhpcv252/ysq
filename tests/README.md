# Tests

CTest + GoogleTest. Built by default; disable with `-DYSQ_BUILD_TESTS=OFF`.

```sh
ctest --test-dir build
ctest --test-dir build --output-on-failure -R CoreClock
```

One executable per test file, registered with `gtest_discover_tests`, so each
`TEST()` case appears in CTest under its own name.

## Layout

| Directory      | Scope                                                            |
| -------------- | ---------------------------------------------------------------- |
| `smoke/`       | The build itself: dependencies link, options took effect          |
| `unit/`        | One module in isolation                                           |
| `integration/` | Modules in combination                                            |
| `e2e/`         | Whole applications, headless, asserting physical invariants       |

`e2e/` arrives with the applications it exercises. `integration/` currently holds
`core_runtime.cpp`, which drives `Config`, `Logger`, `Clock`, `Timer`, `Event` and
`UUID` through one fixed-step run: the wiring an application will do in `main()`,
tested here because `Core` has no application to host it yet.

Everything here runs CPU-only and needs no GPU, no window and no display. The
CPU compute backend is the reference implementation, so correctness is testable
on any machine; GPU backends are validated against it within tolerance rather
than for exact equality, since they generally run `float32` where the reference
runs `float64`.

## Smoke tests

These exist because a dependency that is checked out, configured and compiled
can still be wired up wrong, and the failure would otherwise surface much later
in something unrelated.

- `spdlog_format.cpp` asserts spdlog was built with `SPDLOG_USE_STD_FORMAT`, so
  the project keeps one formatting implementation and no bundled fmt.
- `graphics_link.cpp` links GLAD, GLFW and Dear ImGui including its backends,
  and calls only entry points that need no context or display. Built only when
  `YSQ_BUILD_GRAPHICS=ON`.

## Adding a test

```cmake
ysq_add_test(test_math_vector SOURCES math_vector.cpp LIBS ysq::Math)
```

`ysq_add_test` is defined in `tests/CMakeLists.txt` and links `GTest::gtest_main`
and `ysq::warnings` for you.
