# UI

Dear ImGui panels and Dear ImPlot charts, drawn as an overlay on whatever
`Renderer` already put in the window's framebuffer that frame — same window,
same frame, not a separate view.

**Target:** `ysq::UI` (static)
**Depends on:** `ysq::Math`, `ysq::Platform`, both `PUBLIC` since `Panel`
hands back a `Math` type and `ImGuiLayer::create()` takes a
`Platform::Window` by reference. `ysq::Core`, `imgui`, `implot` and `glad`
are implementation details, linked `PRIVATE`.

Built only under `YSQ_BUILD_GRAPHICS`, same as `Renderer`.

## Contents

| Header | Purpose |
| --- | --- |
| `UI/ImGuiLayer.hpp` | Owns the ImGui/ImPlot contexts and their GLFW+OpenGL3 backends |
| `UI/Panel.hpp` | Generic bound-widget vocabulary: slider, typed numeric input, checkbox, color edit, readout, button, combo |
| `UI/StatsOverlay.hpp` | Frame time, FPS, draw-call count |
| `UI/CameraOverlay.hpp` | A camera's current status (position, mode-specific detail) as plain text a Renderer-side source builds |
| `UI/PlotPanel.hpp` | `TimeSeriesPlot`, `ScatterPlot`: live charts backed by ImPlot |

## Bind, don't hardcode

`Panel` knows nothing about what a simulation parameter is. It binds plain
references — `slider("Time scale", timeScale, 0.0f, 10.0f)` — so the same
machinery drives whichever `Application` is running without UI carrying any
per-application knowledge.

```cpp
ysq::ImGuiLayer ui = *ysq::ImGuiLayer::create(window);

ysq::Panel controls("Simulation");
controls.slider("Time scale", timeScale, 0.0f, 10.0f);
controls.checkbox("Paused", paused);

ysq::TimeSeriesPlot energyPlot("Energy drift");

// per frame:
ui.beginFrame();
controls.draw();
energyPlot.addSample(simulationTime, totalEnergy);
energyPlot.draw();
ui.endFrame();
```

## Keeping panel clicks out of the 3D view

`ImGuiLayer::wantsMouseCapture()` reports whether a panel widget currently
wants the mouse. An `Application` driving a `Renderer` camera controller in
the same window should check it each frame and call
`Platform::InputState::suppressMouseThisFrame()` when true — otherwise
dragging a slider also drags the 3D camera underneath it, since neither
`OrbitCameraController` nor `FreeFlyCameraController` know anything about
`UI` (`Renderer` and `UI` are peers) and so cannot check this themselves.

## Charts are diagnostics, not the 3D view

A Minkowski diagram or an energy-drift curve is 2D data visualization, not a
3D scene, and rendering the simulated world is a different concern from
charting a measurement taken from it. `Renderer`'s own orthographic `Camera`
already renders a genuinely planar *scene* (a top-down orbit view) when one
is needed; `PlotPanel` is for *data*, and composes with `Renderer`'s 3D
viewport in the same window rather than replacing it.
