# UI

Dear ImGui panels and Dear ImPlot charts, drawn as an overlay on whatever
`Renderer` already put in the window's framebuffer that frame — same window,
same frame, not a separate view. See `docs/rendering.md`.

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
| `UI/Panel.hpp` | Generic bound-widget vocabulary: slider, checkbox, color edit, readout, button, combo |
| `UI/StatsOverlay.hpp` | Frame time, FPS, draw-call count |
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

## Charts are diagnostics, not the 3D view

`docs/rendering.md` covers why plotting lives here rather than as a Renderer
feature: a Minkowski diagram or an energy-drift curve is 2D data
visualization, not a 3D scene, and `PlotPanel` composes with `Renderer`'s 3D
viewport in the same window rather than replacing it.
