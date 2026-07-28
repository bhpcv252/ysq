#include <Math/Vector3.hpp>
#include <UI/ImGuiLayer.hpp>
#include <UI/Panel.hpp>
#include <UI/PlotPanel.hpp>
#include <UI/StatsOverlay.hpp>

#include <support/GLContext.hpp>
#include <support/MathApprox.hpp>

#include <imgui.h>

#include <gtest/gtest.h>

#include <optional>
#include <string>

// ImGuiLayer's lifecycle and Panel's bindings are both real ImGui, not a
// headless stand-in: ImGui::NewFrame() asserts on an unset display size and
// several other invariants that only a real ImGui_ImplGlfw/OpenGL3 init
// satisfies, so this needs the same offscreen context every other
// Renderer/UI integration test does rather than a lighter substitute.
//
// The viewport is 256x256, not the 64x64 other GL tests use: ImGui's
// default window cascade position is a fixed pixel offset with no idea how
// small the display is, and at 64x64 it can place a fresh window entirely
// outside the visible area.
//
// Every check below renders two frames, not one: a window's first
// ("appearing") frame measures its auto-fit content size and paints
// nothing, so real draw data only appears starting the second frame.
//
// ImDrawData::CmdListsCount is an obsolete field this ImGui version no
// longer populates (see imgui.h: "Use CmdLists.Size instead"); CmdLists.Size
// is what these tests actually check.
//
// IniFilename is disabled on every context: ImGui persists window
// positions/sizes to disk by default, and a test suite must not depend on
// -- or leave behind -- state from a previous run.

namespace {

using ysq::ImGuiLayer;
using ysq::Panel;
using ysq::StatsOverlay;
using ysq::TimeSeriesPlot;
using ysq::Vec3f;
using ysq::test::GLSession;
using ysq::test::openGLSession;

}  // namespace

TEST(UIImGuiLayer, ARealFrameProducesDrawDataWithoutCrashing) {
    GLSession session = openGLSession(256, 256);
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::string error;
    std::optional<ImGuiLayer> ui = ImGuiLayer::create(*session.window, &error);
    ASSERT_TRUE(ui) << error;
    ImGui::GetIO().IniFilename = nullptr;

    for (int frame = 0; frame < 2; ++frame) {
        ui->beginFrame();
        ImGui::Begin("Test");
        ImGui::Text("hello");
        ImGui::End();
        ui->endFrame();
    }

    ASSERT_NE(ImGui::GetDrawData(), nullptr);
    EXPECT_GT(ImGui::GetDrawData()->CmdLists.Size, 0);
}

// Binding is the whole point of Panel: it hands ImGui a direct reference, so
// the widget reads and (on interaction) writes through it. No mouse input
// happens here, so nothing should touch the bound values — this is the
// wiring proof that the reference plumbing does not corrupt them even when
// nothing interacts with the widget.
TEST(UIImGuiLayer, APanelBindingRoundTripsThroughARealFrameUnchanged) {
    GLSession session = openGLSession(256, 256);
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<ImGuiLayer> ui = ImGuiLayer::create(*session.window);
    ASSERT_TRUE(ui);
    ImGui::GetIO().IniFilename = nullptr;

    float timeScale = 2.5f;
    bool paused = true;
    Vec3f tint{0.1f, 0.2f, 0.3f};
    std::string status = "idle";

    Panel panel("Simulation");
    panel.slider("Time scale", timeScale, 0.0f, 10.0f);
    panel.checkbox("Paused", paused);
    panel.colorEdit("Tint", tint);
    panel.text("Status", status);

    for (int frame = 0; frame < 2; ++frame) {
        ui->beginFrame();
        panel.draw();
        ui->endFrame();
    }

    EXPECT_FLOAT_EQ(timeScale, 2.5f);
    EXPECT_TRUE(paused);
    EXPECT_VEC_APPROX(tint, (Vec3f{0.1f, 0.2f, 0.3f}));
    EXPECT_EQ(status, "idle");

    ASSERT_NE(ImGui::GetDrawData(), nullptr);
    EXPECT_GT(ImGui::GetDrawData()->CmdLists.Size, 0)
        << "a panel with bound widgets must actually emit geometry";
}

TEST(UIImGuiLayer, StatsOverlayAndPlotPanelDrawWithoutCrashing) {
    GLSession session = openGLSession(256, 256);
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<ImGuiLayer> ui = ImGuiLayer::create(*session.window);
    ASSERT_TRUE(ui);
    ImGui::GetIO().IniFilename = nullptr;

    StatsOverlay stats;
    stats.update(0.016f, 5);

    TimeSeriesPlot plot("Energy");
    plot.addSample(0.0, 1.0);
    plot.addSample(1.0, 0.99);

    for (int frame = 0; frame < 2; ++frame) {
        ui->beginFrame();
        stats.draw();
        plot.draw();
        ui->endFrame();
    }

    ASSERT_NE(ImGui::GetDrawData(), nullptr);
    EXPECT_GT(ImGui::GetDrawData()->CmdLists.Size, 0);
}
