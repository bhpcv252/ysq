# Tutorial 2: adding visualization

[Tutorial 1](01-your-first-simulation.md) built the orbit and printed
numbers. This one takes the same bodies and the same stepper and puts them
on screen: a window, a camera you can orbit with the mouse, the planet
drawn as a sphere with a trail behind it, a slider for time scale, and a
live energy chart.

This draws on [docs/core.md](../core.md), [docs/platform.md](../platform.md),
[docs/renderer.md](../renderer.md), and [docs/ui.md](../ui.md).

## Set up the window and the pieces that draw

Everything from Tutorial 1 (`bodies`, `gravity`, `stepper`) stays exactly
as it was; only `main.cpp` grows. This is also the shape every real
`Application` in YSQ follows: see [docs/applications.md](../applications.md).

```cpp
#include <Platform/Platform.hpp>
#include <Platform/Window.hpp>
#include <Renderer/Renderer.hpp>
#include <Renderer/CameraController.hpp>
#include <UI/ImGuiLayer.hpp>
#include <UI/Panel.hpp>
#include <UI/PlotPanel.hpp>
#include <Core/Clock.hpp>
#include <Core/Timer.hpp>

const auto platform = ysq::Platform::initialize();
auto window = ysq::Window::create({.title = "My First Simulation"});
ysq::Renderer renderer = *ysq::Renderer::create();
ysq::ImGuiLayer ui = *ysq::ImGuiLayer::create(*window);

ysq::Mesh sphere = *ysq::Mesh::sphere();
ysq::OrbitCameraController orbitCamera;
ysq::Camera camera;

ysq::Panel controls("Simulation");
float timeScale = 1.0f;
bool paused = false;
controls.slider("Time scale", timeScale, 0.0f, 20.0f);
controls.checkbox("Paused", paused);

ysq::TimeSeriesPlot energyPlot("Energy drift");

ysq::Clock clock;
ysq::Timer frame;
```

## The loop

This is [docs/core.md](../core.md)'s fixed-step `Clock` pattern, with a
draw call and a UI frame added around it:

```cpp
while (!window->shouldClose()) {
    window->input().newFrame();
    ysq::Platform::pollEvents();
    orbitCamera.update(camera, window->input());

    clock.setTimeScale(timeScale);
    if (paused) { clock.pause(); } else { clock.resume(); }
    clock.advance(frame.lap().count());

    while (clock.consumeStep()) {
        ysq::PhaseState<ysq::NBodyState> next = state;
        stepper.step(gravity, clock.simulationTime(), state, clock.fixedStep(), next);
        state = next;
        ysq::applyState(bodies, state.position, state.velocity);
    }

    const ysq::Extent extent = window->size();
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

    renderer.beginFrame(camera, aspect, extent.width, extent.height);
    renderer.setLights({}, std::array{ysq::DirectionalLight{.direction = {-1.0f, -1.0f, -1.0f}}});
    for (const ysq::Body& body : bodies) {
        const ysq::Vec3 metres = body.position.value();
        const ysq::Vec3f position{static_cast<float>(metres.x), static_cast<float>(metres.y),
                                   static_cast<float>(metres.z)};
        renderer.draw(sphere, material, ysq::Matrix4<float>::translation(position));
    }
    renderer.debugDraw().axes();
    renderer.endFrame();

    ui.beginFrame();
    controls.draw();
    energyPlot.addSample(clock.simulationTime(), energy(bodies).value());
    energyPlot.draw();
    ui.endFrame();

    window->swapBuffers();
}
```

`timeScale` and `paused` are read straight off the `Panel` slider and
checkbox; there's no separate step to wire UI state into the simulation,
because they're the same variables. Watch `energyPlot` while you drag the
time-scale slider up: the physics stays correct (Verlet's bounded energy
error, from [docs/math/integrators.md](../math/integrators.md)) even as
the simulation runs faster or slower than real time.

An orbit trail is one more `DebugDraw` call per body, per frame, from
wherever that body was last frame to where it is now:

```cpp
renderer.debugDraw().line(previousPosition, position);
```

That means keeping one "previous position" per body in your own
`Application` state (an array parallel to `bodies`, updated after each
draw), the same ownership rule [docs/renderer.md](../renderer.md) mentions:
`Renderer` doesn't retain anything, your program does.

## Next

[Tutorial 3](03-choosing-a-gravity-model.md) scales this up to many bodies,
where direct summation stops being fast enough.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+tutorials/02-adding-visualization)
and let us know.
