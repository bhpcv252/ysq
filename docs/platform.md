# Platform: the window, the context, and input

Everything that talks to the operating system's windowing layer, and
nothing else.

## The idea

A **window** is the rectangle the operating system gives your program to
draw pixels into. A **graphics context** is your program's live connection
to the graphics driver that actually does the drawing; you don't draw
directly into a window, you draw through a context that happens to be
attached to one. **Input** is just the keyboard and mouse, sampled once per
frame rather than reacted to event by event, since a simulation loop wants
to ask "is this key held right now" on its own schedule.

`Platform` is the one place in YSQ that needs any of this. Everything above
it (`Renderer`, `UI`, `Applications`) goes through `Platform` rather than
touching the operating system or GLFW directly.

**Headless** is the other idea worth having up front: sometimes you want to
run a simulation with no window at all (an automated test, a CI job, a
server with no display attached), and still have a real, working graphics
context to render into or compute with. YSQ can do that: a special backend
produces a context in software, with no display server and no window,
which is what lets the entire test suite (including the ones that touch
graphics) run on a machine that has never had a monitor plugged into it.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Platform/Platform.hpp` | Starts and stops the windowing system, picks a backend |
| `Platform/Window.hpp` | A window and its OpenGL context, onscreen or offscreen |
| `Platform/Input.hpp` | Keyboard and mouse state, sampled per frame |

## Using it

```cpp
#include <Platform/Platform.hpp>
#include <Platform/Window.hpp>

const auto platform = ysq::Platform::initialize();
if (!platform) { /* no display, or no windowing system at all */ }

auto window = ysq::Window::create({.title = "YSQ"});
while (!window->shouldClose()) {
    window->input().newFrame();
    ysq::Platform::pollEvents();

    if (window->input().keyPressed(ysq::Key::Space)) {
        // ... whatever a fresh press of Space should do
    }

    // simulate, draw

    window->swapBuffers();
}
```

`Key` names a physical key position (`Key::Q` is wherever Q sits on a US
keyboard layout), not whatever character the active layout happens to print
there, which is what a movement binding actually wants. Three questions per
key, and they answer different things: `keyDown` (held right now),
`keyPressed` (went down this frame), `keyReleased` (came up this frame).

An offscreen, headless context is a window created with `visible` off; there
is one code path for both cases, not two. `Window::createOffscreen()` is a
name for exactly that combination of settings.

## Go deeper

[src/Platform/README.md](../src/Platform/README.md) has the full interface,
the backend table (`Win32`/`Cocoa`/`Wayland`/`X11`/`Null`, and why `Null`,
the headless one, is never chosen automatically), the exact `WindowError`
variants and what each means, and why GLFW itself never appears in any
`Platform` header.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+platform)
and let us know.
