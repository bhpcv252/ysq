# Platform

The window, the OpenGL context, and input. Everything that talks to the
operating system's windowing layer, and nothing else.

**Target:** `ysq::Platform` (static, graphics builds only)
**Depends on:** `ysq::Core` for logging, linked `PRIVATE`. Externally, GLFW
(`PRIVATE`) and GLAD (`PUBLIC`).

`Platform` sits in the base layer beside `Core`, `Math` and `Units`, because
`Compute`'s OpenGL backend needs a context from it. Nothing in the simulation
core may depend on it: with `-DYSQ_BUILD_GRAPHICS=OFF` this module is not
configured at all, and the rest of the engine still builds and tests.

## Contents

| Header                 | Purpose                                          |
| ---------------------- | ------------------------------------------------ |
| `Platform/Platform.hpp` | Windowing system lifetime, backend selection, event pumping |
| `Platform/Window.hpp`  | Window and OpenGL context, onscreen or offscreen |
| `Platform/Input.hpp`   | Keyboard and mouse state, sampled per frame      |

Nothing here is thread-safe, and macOS additionally requires that all of it
happens on the main thread.

**GLFW does not appear in any header.** `Window.hpp` forward-declares
`GLFWwindow` for one accessor and `Input.hpp` names its own key codes, so
nothing that consumes `Platform` compiles GLFW headers, exactly as `Core/Logger`
keeps spdlog out of its own. GLAD is the deliberate exception: it is `PUBLIC`,
because anything holding a live context calls OpenGL through the same
process-wide entry points this module resolved.

## Platform

```cpp
const auto platform = ysq::Platform::initialize();
if (!platform) { /* no display, or no windowing system at all */ }
```

A reference-counted handle. The first live one initialises the windowing system,
the last one to die shuts it down. Copying retains, which is what lets a `Window`
hold one, so the windowing system cannot be shut down underneath a live window
however the handles are ordered.

Initialising again with a different backend fails rather than silently keeping
the first, because the backend cannot be changed without a full shutdown and a
test that asked for `Null` must not quietly get X11.

### Backends

| Backend  | Where                                            |
| -------- | ------------------------------------------------ |
| `Win32`  | Windows                                          |
| `Cocoa`  | macOS                                            |
| `Wayland`, `X11` | Linux, chosen from the session               |
| `Null`   | No display anywhere. Contexts come from OSMesa   |

`Null` is never auto-selected and must be asked for by name. It is the headless
path: no display server, no window, and an OpenGL context produced in software
by OSMesa, which GLFW loads at run time (`libOSMesa.so.8` and friends). That
library is a runtime dependency and is usually absent; when it is,
`Window::create` fails cleanly and a caller with no need for a context is
unaffected.

OSMesa also refuses forward-compatible contexts, which are on by default because
macOS gives nothing above 2.1 without them. `Window` drops the flag on the
`Null` backend rather than failing: forward compatibility only removes
deprecated functionality, so a context without it can do strictly more, and this
is the difference between a headless context and none at all.

That adjustment is not hidden. `effectiveContext(requested, backend)` is public
and pure, so a caller can ask what a backend will do to its request before
creating anything, and `Window::contextSettings()` reports what was actually
asked of the driver. Whether the profile is forward compatible decides whether
deprecated OpenGL is callable, which is not something a renderer should have to
infer from which backend it happens to be running on.

### Logging

The GLFW error callback logs at **debug**, not error. This module probes: native
backend then `Null`, one OpenGL version then a lower one. Most failures reaching
that callback are expected steps in a sequence that goes on to succeed, and
logging them as errors means a healthy run fills the log with errors, which
teaches people to ignore the level. Every failure is also returned as a
`PlatformError` or `WindowError`, so the code that knows whether it was fatal is
the code that decides how loudly to say so.

## Window

```cpp
auto window = ysq::Window::create({.title = "YSQ"});
while (!window->shouldClose()) {
    window->input().newFrame();
    ysq::Platform::pollEvents();
    // simulate, draw
    window->swapBuffers();
}
```

An offscreen context is a window with `visible` off, so there is one code path
rather than two; `createOffscreen()` is a name for the settings that make one.

The default request is **OpenGL 4.1 core, forward compatible**. 4.1 is the most
macOS offers and enough for the visualiser, so it is the version that works
everywhere the renderer needs to. `Compute`'s OpenGL backend asks for 4.3, where
compute shaders start, and gets a typed failure where that is unavailable.

Settings are checked before the windowing system is asked anything, so a
nonsensical request fails identically on every machine: non-positive sizes, a
core profile below 3.2, forward compatibility below 3.0, and any version above
`kMaxSupportedGL`. That last one is the version floor `third_party/README.md`
delegates here. GLAD is generated for 4.6, so a higher request cannot have its
entry points resolved however enthusiastically a driver answers it.

`kMaxSupportedGL` is checked against what GLAD actually defines, in `Window.cpp`.
Nothing else ties the constant to the loader, and regenerating GLAD lower would
otherwise leave it quietly over-promising, with null entry points at run time as
the payoff. Raise or lower either and the build stops.

`WindowError` distinguishes the five ways this fails, because they call for
different responses:

| Code                     | Means                                              |
| ------------------------ | -------------------------------------------------- |
| `PlatformNotInitialized` | No live `Platform` handle                          |
| `InvalidSettings`        | Rejected before the windowing system saw it        |
| `CreationFailed`         | The driver refused, usually the requested version  |
| `LoaderFailed`           | A context, but no entry points resolved            |
| `VersionBelowRequest`    | The context reports less than was asked for        |

A failed creation leaves nothing behind: the half-built window is destroyed and
whatever context was current before is restored.

### One loader, one context

GLAD's entry points are process-wide globals, so they describe whichever context
was made current last. Two contexts from different drivers cannot both be live,
and after a shutdown the pointers refer to nothing at all.

`makeContextCurrent()` therefore reloads them whenever the context changes, and
destroying a window forgets them if they came from it. Reloading costs a few
hundred symbol lookups, so switch contexts at setup time, not per frame.

### Moves

`Window` is move-only, and a move re-points the windowing system's user pointer
at the new object. The callbacks reach the `InputState` through that pointer, so
a move that forgot would leave input silently feeding the husk it left behind.

## Input

```cpp
if (window->input().keyPressed(ysq::Key::Space)) { clock.stepOnce(); }
```

`Key` names a physical position in the US layout, not a character: `Key::Q` is
the key where Q sits whatever the active layout prints on it, which is what a
movement binding wants. Layout-correct text entry is a different problem and is
not solved here.

The values are dense and start at zero, so key state is a `std::bitset` and the
mapping is an array. GLFW's own codes are sparse; the translation table in
`Input.cpp` is the one place the two orderings meet, and
`tests/unit/platform_input.cpp` round-trips every key through it.

Three questions per button, and they differ:

| Query         | Answers                        |
| ------------- | ------------------------------ |
| `keyDown`     | Held right now                 |
| `keyPressed`  | Went down during this frame    |
| `keyReleased` | Came up during this frame      |

`newFrame()` clears the edges and takes the cursor snapshot that `cursorDelta()`
measures from. Call it before polling. Event delivery is global and input state
is per window, so a program with more than one window calls `newFrame()` on
every one of them before the single `Platform::pollEvents()`.

A key pressed and released inside one frame reports both edges and is not down,
so a fast tap is never missed. Auto-repeat sets neither edge: the key was already
down, and treating repeat as a fresh press fires a once-per-press action at the
keyboard's repeat rate.

`InputState` knows nothing about GLFW. `Window` feeds it from callbacks; a test
feeds it directly, which is why the interesting behaviour here is testable with
no window, no display and no context.

**Losing focus releases everything held.** A key held while the window loses
focus never delivers its release, so without that it stays down forever. The
synthetic releases are reported as edges, so a handler watching for one still
sees it, and the cursor baseline is dropped so that coming back to the window
does not read as one enormous mouse movement.

**`suppressMouseThisFrame()`** makes every mouse button/cursor-delta/scroll
query answer as if nothing were happening, for the rest of the current
frame — a query-time override, not a mutation of the real tracked state, so
it reads correctly again the instant `newFrame()` resets it. `InputState`
still knows nothing about `UI`/ImGui (this module sits below both); a
caller that also draws ImGui panels in the same window (an `Application`)
is the one that knows whether ImGui wants the mouse this frame, and calls
this to keep that click from also acting on whatever else reads mouse
input, such as a `Renderer` camera controller.

## Testing

`tests/unit/platform_input.cpp` and `tests/unit/platform_window.cpp` need no
display: the input logic is pure, and the window tests run on the `Null` backend
and stop short of creating a context, which is where a rasteriser has to exist.

`tests/integration/platform_context.cpp` opens a real context. Whether one can
exist is a property of the machine, so each test tries the native backend, falls
back to `Null`, and skips if neither yields a context. Configuring with
`-DYSQ_REQUIRE_HEADLESS_GL=ON` turns those skips into failures; CI sets it on the
one job with OSMesa installed, so the headless path stays a tested claim rather
than one that quietly skips everywhere.

What is not covered, deliberately: nothing delivers a synthetic key event,
because no backend available here can send one, so the hop from the GLFW
callback into `InputState` is untested. `EveryInputCallbackIsRegistered` covers
the failure that would otherwise be silent, a callback never installed, by
exploiting the fact that GLFW's setters return the callback they replaced.
