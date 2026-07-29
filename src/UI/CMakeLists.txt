# UI: Dear ImGui panels and Dear ImPlot charts, in the same window and frame
# as whatever Renderer draws. Presentation layer, so ysq::warnings not
# _strict; see the root README's Warnings section.

add_library(ysq_ui STATIC
    ImGuiLayer.hpp
    ImGuiLayer.cpp
    Panel.hpp
    Panel.cpp
    StatsOverlay.hpp
    StatsOverlay.cpp
    PlotPanel.hpp
    PlotPanel.cpp)
add_library(ysq::UI ALIAS ysq_ui)

target_include_directories(ysq_ui PUBLIC ${YSQ_SOURCE_INCLUDE_DIR})

# Math and Platform are PUBLIC: Panel's colorEdit hands back a Math type and
# ImGuiLayer::create() takes a Platform::Window by reference, so a consumer
# of UI needs both. Core, imgui, implot and glad are implementation details,
# PRIVATE. See src/UI/README.md.
target_link_libraries(ysq_ui
    PUBLIC ysq::Math ysq::Platform
    PRIVATE ysq::Core imgui implot glad ysq::warnings)
