#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace ysq {

namespace detail {

/// Resets the default vertical cascade TimeSeriesPlot and ScatterPlot open
/// into, so a fresh ImGuiLayer session starts back at the first slot rather
/// than wherever a previous session's plots left off. Called by
/// ImGuiLayer::create(); not for application code to call directly.
void resetPlotLayoutCascade();

}  // namespace detail

/// A live-updating time-series chart: energy/momentum drift, orbital
/// elements, or any scalar tracked against simulation time. Backed by
/// ImPlot, drawn alongside the 3D viewport in the same window and frame —
/// see docs/rendering.md for why this lives in UI rather than Renderer.
class TimeSeriesPlot {
public:
    explicit TimeSeriesPlot(std::string title, std::string yLabel = "value",
                            std::size_t maxSamples = 2000);

    /// Appends one sample. Past maxSamples, the oldest sample is dropped, so
    /// a long-running simulation's plot stays bounded rather than growing
    /// forever.
    void addSample(double time, double value);

    /// Draws the plot inside one ImGui window. Call inside an ImGuiLayer
    /// frame (between beginFrame() and endFrame()).
    void draw();

private:
    std::string m_title;
    std::string m_yLabel;
    std::size_t m_maxSamples;
    std::deque<double> m_times;
    std::deque<double> m_values;
    /// Which slot in the default vertical cascade this instance opens into,
    /// assigned once at construction; see PlotPanel.cpp.
    int m_layoutSlot;
};

/// A static 2D scatter plot: phase space (position vs. momentum), a
/// Minkowski diagram (ct vs. x), or any other x-y relationship that isn't
/// naturally a function of simulation time.
class ScatterPlot {
public:
    explicit ScatterPlot(std::string title, std::string xLabel = "x",
                         std::string yLabel = "y");

    /// Replaces the plotted points. x and y must be the same length.
    void setPoints(std::vector<double> x, std::vector<double> y);

    void draw();

private:
    std::string m_title;
    std::string m_xLabel;
    std::string m_yLabel;
    std::vector<double> m_x;
    std::vector<double> m_y;
    int m_layoutSlot;
};

}  // namespace ysq
