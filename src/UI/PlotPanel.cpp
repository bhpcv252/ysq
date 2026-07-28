#include <UI/PlotPanel.hpp>

#include <implot.h>

namespace ysq {

TimeSeriesPlot::TimeSeriesPlot(std::string title, std::string yLabel,
                               std::size_t maxSamples)
    : m_title(std::move(title)), m_yLabel(std::move(yLabel)), m_maxSamples(maxSamples) {}

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

    if (ImPlot::BeginPlot(m_title.c_str())) {
        ImPlot::SetupAxes("time", m_yLabel.c_str());
        ImPlot::PlotLine(m_yLabel.c_str(), times.data(), values.data(),
                         static_cast<int>(times.size()));
        ImPlot::EndPlot();
    }
}

ScatterPlot::ScatterPlot(std::string title, std::string xLabel, std::string yLabel)
    : m_title(std::move(title)),
      m_xLabel(std::move(xLabel)),
      m_yLabel(std::move(yLabel)) {}

void ScatterPlot::setPoints(std::vector<double> x, std::vector<double> y) {
    m_x = std::move(x);
    m_y = std::move(y);
}

void ScatterPlot::draw() {
    if (m_x.empty() || m_x.size() != m_y.size()) {
        return;
    }
    if (ImPlot::BeginPlot(m_title.c_str())) {
        ImPlot::SetupAxes(m_xLabel.c_str(), m_yLabel.c_str());
        ImPlot::PlotScatter(m_title.c_str(), m_x.data(), m_y.data(),
                            static_cast<int>(m_x.size()));
        ImPlot::EndPlot();
    }
}

}  // namespace ysq
