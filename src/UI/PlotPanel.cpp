#include <UI/PlotPanel.hpp>

#include <imgui.h>
#include <implot.h>

namespace ysq {

namespace {

// Default vertical cascade for plot windows, so a second (or third, ...)
// plot doesn't land exactly on top of the first. Each TimeSeriesPlot and
// ScatterPlot claims the next slot once, at construction; draw() only ever
// applies its own slot's position with ImGuiCond_FirstUseEver, so a plot the
// user has since moved, resized or expanded is left alone.
constexpr float kPlotCascadeOriginY = 70.0f;
constexpr float kPlotCascadeSlotHeight = 32.0f;
constexpr ImVec2 kPlotDefaultSize{450.0f, 300.0f};

// Process-lifetime by construction (an anonymous-namespace variable), but
// not process-lifetime in effect: detail::resetPlotLayoutCascade() zeroes it
// at the start of every ImGuiLayer session, so it only ever counts plots
// within the session currently running rather than every plot ever
// constructed since the process started.
int g_nextPlotLayoutSlot = 0;

int nextPlotLayoutSlot() {
    return g_nextPlotLayoutSlot++;
}

void applyDefaultPlotWindowLayout(int layoutSlot) {
    const float y =
        kPlotCascadeOriginY + static_cast<float>(layoutSlot) * kPlotCascadeSlotHeight;
    ImGui::SetNextWindowPos(ImVec2(10.0f, y), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(kPlotDefaultSize, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
}

}  // namespace

namespace detail {

void resetPlotLayoutCascade() {
    g_nextPlotLayoutSlot = 0;
}

}  // namespace detail

TimeSeriesPlot::TimeSeriesPlot(std::string title, std::string yLabel,
                               std::size_t maxSamples)
    : m_title(std::move(title)),
      m_yLabel(std::move(yLabel)),
      m_maxSamples(maxSamples),
      m_layoutSlot(nextPlotLayoutSlot()) {}

void TimeSeriesPlot::addSample(double time, double value) {
    m_times.push_back(time);
    m_values.push_back(value);
    while (m_times.size() > m_maxSamples) {
        m_times.pop_front();
        m_values.pop_front();
    }
}

void TimeSeriesPlot::draw() {
    if (m_times.empty()) {
        return;
    }
    // ImPlot wants contiguous storage; a deque is not, so this copies once
    // per frame. maxSamples keeps that copy bounded regardless of how long
    // the simulation has been running.
    const std::vector<double> times(m_times.begin(), m_times.end());
    const std::vector<double> values(m_values.begin(), m_values.end());

    applyDefaultPlotWindowLayout(m_layoutSlot);
    ImGui::Begin(m_title.c_str());
    if (ImPlot::BeginPlot(m_title.c_str())) {
        ImPlot::SetupAxes("time", m_yLabel.c_str());
        ImPlot::PlotLine(m_yLabel.c_str(), times.data(), values.data(),
                         static_cast<int>(times.size()));
        ImPlot::EndPlot();
    }
    ImGui::End();
}

ScatterPlot::ScatterPlot(std::string title, std::string xLabel, std::string yLabel)
    : m_title(std::move(title)),
      m_xLabel(std::move(xLabel)),
      m_yLabel(std::move(yLabel)),
      m_layoutSlot(nextPlotLayoutSlot()) {}

void ScatterPlot::setPoints(std::vector<double> x, std::vector<double> y) {
    m_x = std::move(x);
    m_y = std::move(y);
}

void ScatterPlot::draw() {
    if (m_x.empty() || m_x.size() != m_y.size()) {
        return;
    }
    applyDefaultPlotWindowLayout(m_layoutSlot);
    ImGui::Begin(m_title.c_str());
    if (ImPlot::BeginPlot(m_title.c_str())) {
        ImPlot::SetupAxes(m_xLabel.c_str(), m_yLabel.c_str());
        ImPlot::PlotScatter(m_title.c_str(), m_x.data(), m_y.data(),
                            static_cast<int>(m_x.size()));
        ImPlot::EndPlot();
    }
    ImGui::End();
}

}  // namespace ysq
