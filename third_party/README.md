# third_party

Vendored dependencies. All of it is built from source as part of the YSQ build;
nothing here is expected to be installed on the system.

`third_party/CMakeLists.txt` owns every dependency target. Module `CMakeLists.txt`
files link those targets and never reach into these directories directly.

## Contents

| Directory     | Version   | Form       | Role                                    |
| ------------- | --------- | ---------- | --------------------------------------- |
| `glfw/`       | `3.4`     | submodule  | Window, input, OpenGL context           |
| `imgui/`      | `v1.92.9` | submodule  | Control and debug panels                |
| `spdlog/`     | `v1.17.0` | submodule  | Logging                                 |
| `googletest/` | `v1.17.0` | submodule  | Test framework (only when tests are on) |
| `glad/`       | GL 4.6 core | generated, committed | OpenGL function loader        |

Submodules are pinned to release tags, not branches. To fetch them:

```sh
git submodule update --init --recursive
```

Forgetting this is the most likely first failure on a fresh clone, so
`third_party/CMakeLists.txt` checks for each submodule the current configuration
actually needs and fails immediately with that command, rather than letting
CMake emit a wall of "non-existent target" errors that never mention submodules.

## Build notes

**spdlog** is built with `SPDLOG_USE_STD_FORMAT=ON`, so it uses `std::format`
rather than its bundled fmt. That keeps exactly one formatting implementation in
the project. `tests/smoke/spdlog_format.cpp` asserts this rather than trusting
the option, because a dependency bump can lose it silently.

**Dear ImGui** ships sources with no build system, so `imgui` is a target we
define: the five core translation units plus the GLFW and OpenGL3 backends.

The OpenGL3 backend keeps its own minimal GL loader (`imgui_impl_opengl3_loader.h`).
That loader is file-local to `imgui_impl_opengl3.cpp`, so it cannot collide with
GLAD, and it is the configuration upstream actually tests. Overriding it with
`IMGUI_IMPL_OPENGL_LOADER_CUSTOM` is possible but buys nothing.

**GLAD** entry points are all runtime-resolved function pointers, so nothing
links against a system OpenGL library. The context provider supplies the loader:

```cpp
gladLoadGL(glfwGetProcAddress);
```

**Graphics** as a whole (`glfw`, `glad`, `imgui`) is gated behind
`YSQ_BUILD_GRAPHICS`. With it `OFF`, none of the three is configured at all,
which is what keeps the headless build honest rather than merely claimed.

## Regenerating GLAD

GLAD is generated rather than submoduled, so the generated sources are committed.
It targets **OpenGL 4.6 core with no extensions**: the loader is a superset and
reports at runtime which functions actually resolved, so a lower context still
works and the version floor is enforced in `Platform`, not here.

```sh
python3 -m venv /tmp/gladenv
/tmp/gladenv/bin/pip install glad2
/tmp/gladenv/bin/glad --api gl:core=4.6 --extensions="" --out-path third_party/glad c
```

That produces exactly three files, all committed:

```
glad/include/glad/gl.h
glad/include/KHR/khrplatform.h
glad/src/gl.c
```

`--extensions=""` matters. Omitting the flag entirely pulls in every extension in
the Khronos registry and inflates the loader by an order of magnitude.

## Adding a dependency

1. `git submodule add <url> third_party/<name>`, then check out a release tag
   inside it so the pin is a commit, not a moving branch.
2. Add it to `third_party/CMakeLists.txt`, forcing its docs/tests/examples/install
   options off and marking its includes `SYSTEM`.
3. Add a row to the table above and a build note if it needs configuring.
4. Add a smoke test under `tests/smoke/` if getting the wiring wrong would fail
   somewhere unrelated later.
