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
| `implot/`     | `v1.0`    | submodule  | Charting on top of Dear ImGui           |
| `spdlog/`     | `v1.17.0` | submodule  | Logging                                 |
| `googletest/` | `v1.17.0` | submodule  | Test framework (only when tests are on) |
| `glad/`       | GL 4.6 core | generated, committed | OpenGL function loader        |
| `stb/`        | `stb_image.h` (public domain) | vendored, committed | Image loading for `Renderer::Texture` |

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

The OpenGL3 backend's default is its own minimal GL loader
(`imgui_impl_opengl3_loader.h`), which `dlopen`s the system's real GL library
independently of GLFW or GLAD. That works with a real display, but not on
headless Linux CI: the runner has OSMesa, which GLAD reaches correctly
through `glfwGetProcAddress`, and no system libGL/GLX for ImGui's own loader
to find, so `ImGui_ImplOpenGL3_Init()` failed there ("Failed to initialize
OpenGL loader!") even though every other GL path in this repo works fine on
the same runner. `IMGUI_IMPL_OPENGL_LOADER_CUSTOM` skips that bundled loader
and force-includes GLAD's header into `imgui_impl_opengl3.cpp` instead (the
file is vendored, so this is a compiler flag, not an edit to it), reusing the
GL entry points GLAD already loaded successfully.

**GLFW** is built with every backend its host supports, including `Null`, which
needs no display and creates its OpenGL contexts through OSMesa. OSMesa is
dlopened at run time (`libOSMesa.so.8` and friends), so it is neither a build
dependency nor linked; where it is absent, context creation on that backend
fails and nothing else is affected. `Platform` is what chooses the backend.

**GLAD** entry points are all runtime-resolved function pointers, so nothing
links against a system OpenGL library. The context provider supplies the loader:

```cpp
gladLoadGL(glfwGetProcAddress);
```

**Dear ImPlot** is the same shape as ImGui: sources with no build system of
its own, so `implot` is a target we define (`implot.cpp`, `implot_items.cpp`),
linked against `imgui` since every ImPlot call needs a live ImGui context.
`implot_demo.cpp` is not built; nothing here needs it.

**stb_image** is a single public-domain header, so it is vendored and
committed rather than submoduled, the same treatment as `glad/`.
`stb/stb_image_impl.cpp` is the one translation unit that defines
`STB_IMAGE_IMPLEMENTATION` and compiles the implementation; every consumer
links the `stb_image` target and gets both the declarations and the symbols,
rather than each risking a duplicate definition by defining the macro itself.

**Graphics** as a whole (`glfw`, `glad`, `imgui`, `implot`, `stb_image`) is
gated behind `YSQ_BUILD_GRAPHICS`. With it `OFF`, none of it is configured at
all, which is what keeps the headless build honest rather than merely claimed.

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
