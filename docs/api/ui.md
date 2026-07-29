# UI API reference

Every public type in `UI`: the ImGui/ImPlot context, bound controls,
live plots, and the stats overlay. Start with [docs/ui.md](../ui.md) for how
`UI` composes with `Renderer` into the same window and frame;
[src/UI/README.md](../../src/UI/README.md) has the full widget list.

## `UI/ImGuiLayer.hpp`

Owns the ImGui and ImPlot contexts and their GLFW+OpenGL3 backends, tied to
one `Window`. **One at a time**: ImGui's context is global process state
(the same constraint GLAD's loader has), so a second live `ImGuiLayer` would
fight the first over it.

```cpp
struct ImGuiLayerSettings {
    bool persistLayout = false;   // remember window positions/sizes across runs, in imgui.ini
};

class ImGuiLayer {
public:
    static std::optional<ImGuiLayer> create(Window& window,
                                             const ImGuiLayerSettings& settings = {},
                                             std::string* error = nullptr);
    // move-only

    void beginFrame();   // call once per frame, before any ImGui/Panel/plot calls
    void endFrame();       // renders everything since beginFrame() into the bound framebuffer
};
```

`persistLayout` defaults **off**: a stale `imgui.ini` would silently mask
whether a panel's *default* layout is actually right, which matters for
anything (like `PlotPanel`'s default cascade) whose default position is
itself part of what's being verified. `beginFrame`/`endFrame` don't clear
the framebuffer; they overlay on whatever `Renderer` already drew that
frame.

## `UI/Panel.hpp`

A declarative list of controls bound directly to plain references: `UI`
never needs to know what a simulation parameter *is*.

```cpp
class Panel {
public:
    explicit Panel(std::string title);

    void slider(std::string label, float& value, float min, float max);
    void slider(std::string label, int& value, int min, int max);
    void checkbox(std::string label, bool& value);
    void colorEdit(std::string label, Vec3f& value);
    void text(std::string label, std::string& value);            // a live readout
    void button(std::string label, std::function<void()> onClick);
    void combo(std::string label, std::vector<std::string> options, int& selected);

    void draw();   // call inside an ImGuiLayer frame

    const std::string& title() const noexcept;
};
```

| Member | Description |
| --- | --- |
| Bindings (`slider`, `checkbox`, ...) | Built once against a long-lived reference (typically an `Application`'s own state); `draw()` reads through fresh every call, so there's no separate "UI state" to keep synchronized with the simulation. |
| `text` | Takes a non-`const` reference like the other bindings, even though nothing writes through it: a `const&` would silently accept a temporary (e.g. `text("FPS", std::to_string(fps))`), and the stored lambda would dangle from the next `draw()`. Pass a named `std::string` you keep alive. |
| `combo` | `options` and `selected` share an index. An out-of-range `selected` draws as no current selection rather than being clamped. |

```cpp
ysq::Panel controls("Simulation");
controls.slider("Time scale", timeScale, 0.0f, 10.0f);
controls.checkbox("Paused", paused);
// once per frame, inside an ImGuiLayer frame:
controls.draw();
```

## `UI/PlotPanel.hpp`

Live charts backed by ImPlot, drawn in the same window and frame as the 3D
viewport.

```cpp
class TimeSeriesPlot {
public:
    explicit TimeSeriesPlot(std::string title, std::string yLabel = "value",
                            std::size_t maxSamples = 2000);

    void addSample(double time, double value);   // oldest sample dropped past maxSamples
    void draw();
};

class ScatterPlot {
public:
    explicit ScatterPlot(std::string title, std::string xLabel = "x", std::string yLabel = "y");

    void setPoints(std::vector<double> x, std::vector<double> y);   // replaces plotted points; x, y same length
    void draw();
};
```

| Type | Use for |
| --- | --- |
| `TimeSeriesPlot` | Energy/momentum drift, orbital elements, or any scalar tracked against simulation time. Bounded memory: past `maxSamples`, the oldest sample is dropped, so a long-running simulation's plot doesn't grow forever. |
| `ScatterPlot` | Phase space (position vs. momentum), a Minkowski diagram (`ct` vs. `x`), or any x-y relationship that isn't naturally a function of simulation time. |

Both open into a default vertical cascade on screen, reset at
`ImGuiLayer::create()` so a fresh session starts at the first slot rather
than wherever a previous session's plots left off.

```cpp
ysq::TimeSeriesPlot energyPlot("Energy drift");
// once per frame:
energyPlot.addSample(simulationTime, totalEnergy);
energyPlot.draw();
```

## `UI/StatsOverlay.hpp`

```cpp
class StatsOverlay {
public:
    void update(float deltaSeconds, std::uint32_t drawCallCount) noexcept;
    void draw();
};
```

A small always-on-top overlay: smoothed frame time, FPS, and draw-call count
(feed it `Renderer::drawCallCount()`). Cheap enough to leave on in every
application from day one.

```cpp
statsOverlay.update(frame.lap().count(), renderer.drawCallCount());
statsOverlay.draw();
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/ui)
and let us know.
