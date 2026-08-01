# UI

Controls and charts, drawn over the same window `Renderer` already put a
frame into, not a separate view.

## The idea

A simulation needs two kinds of on-screen output that aren't the 3D scene
itself: controls (sliders, checkboxes, buttons for things like time scale
or pause) and diagnostic charts (is energy actually conserved, is this
orbit actually periodic). `UI` is Dear ImGui for the first and Dear ImPlot
for the second, composed into the same window and the same frame `Renderer`
draws the 3D view into, so you never manage two separate render targets.

`Panel` knows nothing about what a simulation parameter *is*; it binds
directly to a plain variable you already own (`slider("Time scale",
timeScale, 0.0f, 10.0f)` reads and writes your own `timeScale` in place),
so the same machinery drives whatever `Application` happens to be running
without `UI` needing any per-application knowledge.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `UI/ImGuiLayer.hpp` | Owns the ImGui/ImPlot contexts, ties them to the window |
| `UI/Panel.hpp` | Bound widgets: slider, checkbox, color edit, readout, button, combo |
| `UI/StatsOverlay.hpp` | Frame time, FPS, draw-call count |
| `UI/CameraOverlay.hpp` | A camera's current status (position, speed, POV/Focus, ...) as plain text |
| `UI/PlotPanel.hpp` | `TimeSeriesPlot`, `ScatterPlot`: live charts backed by ImPlot |

A chart here (an energy-drift curve, a Minkowski diagram, orbital elements
over time) is 2D *data* visualization, a different concern from rendering
the simulated world itself, which is why it lives in `UI` rather than as a
`Renderer` feature; see [docs/renderer.md](renderer.md) for the 3D side.

## Using it

```cpp
#include <UI/ImGuiLayer.hpp>
#include <UI/Panel.hpp>
#include <UI/PlotPanel.hpp>

ysq::ImGuiLayer ui = *ysq::ImGuiLayer::create(window);

ysq::Panel controls("Simulation");
controls.slider("Time scale", timeScale, 0.0f, 10.0f);
controls.checkbox("Paused", paused);

ysq::TimeSeriesPlot energyPlot("Energy drift");

// once per frame, after Renderer::endFrame():
ui.beginFrame();
controls.draw();
energyPlot.addSample(simulationTime, totalEnergy);
energyPlot.draw();
ui.endFrame();
```

`timeScale` and `paused` here are the exact variables your simulation loop
reads every step; there's no separate "UI state" to keep synchronized with
the simulation's own state.

## Go deeper

[docs/api/ui.md](api/ui.md) has every signature: `ImGuiLayer`, every
`Panel` binding, `TimeSeriesPlot`/`ScatterPlot`, `StatsOverlay`, and
`CameraOverlay`.

[src/UI/README.md](../src/UI/README.md) has the full widget list and the
`ScatterPlot` type for phase-space-style plots rather than time series.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+ui)
and let us know.
