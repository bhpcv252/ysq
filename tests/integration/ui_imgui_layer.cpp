#include <Math/Vector3.hpp>
#include <UI/ImGuiLayer.hpp>
#include <UI/Panel.hpp>
#include <UI/PlotPanel.hpp>
#include <UI/StatsOverlay.hpp>

#include <support/GLContext.hpp>
#include <support/MathApprox.hpp>

#include <imgui.h>
// FindWindowByName is how StatsOverlayAndPlotPanelDrawWithoutCrashing checks
// that each plot actually opened its own window rather than falling through
// to ImGui's implicit fallback one; there is no public-API way to ask that.
#include <imgui_internal.h>

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

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
// ImGuiLayer::create() disables ini persistence by default, for exactly the
// reason a test suite cares about: ImGui persists window positions/sizes to
// disk by default, and a test must not depend on -- or leave behind --
// state from a previous run. PersistLayoutSettingControlsIniFile is the one
// test that asks for it explicitly, to prove the setting actually reaches
// ImGui; it only checks that IniFilename got set, not any real file I/O.

namespace {

using ysq::ImGuiLayer;
using ysq::ImGuiLayerSettings;
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
    std::optional<ImGuiLayer> ui = ImGuiLayer::create(*session.window, {}, &error);
    ASSERT_TRUE(ui) << error;

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

TEST(UIImGuiLayer, WantsMouseCaptureIsFalseWithNoWidgetInteraction) {
    GLSession session = openGLSession(256, 256);
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<ImGuiLayer> ui = ImGuiLayer::create(*session.window);
    ASSERT_TRUE(ui);

    for (int frame = 0; frame < 2; ++frame) {
        ui->beginFrame();
        ImGui::Begin("Test");
        ImGui::Text("hello");
        ImGui::End();
        ui->endFrame();
    }

    // Simulating an actual hover/click to drive this true is fragile across
    // ImGui versions (the input queue vs. direct IO field writes), so this
    // only checks the always-safe, deterministic case: nothing claims the
    // mouse when nothing has been interacted with.
    EXPECT_FALSE(ui->wantsMouseCapture());
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
    float simSpeed = 42.0f;
    bool paused = true;
    Vec3f tint{0.1f, 0.2f, 0.3f};
    std::string status = "idle";

    Panel panel("Simulation");
    panel.slider("Time scale", timeScale, 0.0f, 10.0f);
    panel.inputFloat("Sim speed", simSpeed);
    panel.checkbox("Paused", paused);
    panel.colorEdit("Tint", tint);
    panel.text("Status", status);

    for (int frame = 0; frame < 2; ++frame) {
        ui->beginFrame();
        panel.draw();
        ui->endFrame();
    }

    EXPECT_FLOAT_EQ(timeScale, 2.5f);
    EXPECT_FLOAT_EQ(simSpeed, 42.0f);
    EXPECT_TRUE(paused);
    EXPECT_VEC_APPROX(tint, (Vec3f{0.1f, 0.2f, 0.3f}));
    EXPECT_EQ(status, "idle");

    ASSERT_NE(ImGui::GetDrawData(), nullptr);
    EXPECT_GT(ImGui::GetDrawData()->CmdLists.Size, 0)
        << "a panel with bound widgets must actually emit geometry";
}

TEST(UIImGuiLayer, ALiveComboReflectsOptionsChangedBetweenDraws) {
    GLSession session = openGLSession(256, 256);
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<ImGuiLayer> ui = ImGuiLayer::create(*session.window);
    ASSERT_TRUE(ui);
    ImGui::GetIO().IniFilename = nullptr;

    std::vector<std::string> options{"Free", "Sun", "Earth", "Moon"};
    int selected = 2;  // "Earth"

    Panel panel("Simulation");
    panel.comboLive("Focus", options, selected);

    ui->beginFrame();
    panel.draw();
    ui->endFrame();
    EXPECT_EQ(selected, 2);

    // A body becoming POV excludes it from Focus's own list, the same way
    // SceneCameraController::focusOptions() does -- the list a live combo
    // is bound to can shrink between frames, and the previously-selected
    // index must not now point at the wrong entry or crash.
    options = std::vector<std::string>{"Free", "Sun", "Moon"};

    for (int frame = 0; frame < 2; ++frame) {
        ui->beginFrame();
        panel.draw();
        ui->endFrame();
    }

    ASSERT_NE(ImGui::GetDrawData(), nullptr);
    EXPECT_GT(ImGui::GetDrawData()->CmdLists.Size, 0);
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

    // Two plots, not one: a single plot could still land inside ImGui's
    // implicit fallback window and look fine. Two only both come out right
    // if each opened its own window, which is the bug this guards against --
    // TimeSeriesPlot::draw() used to call ImPlot::BeginPlot() with no
    // surrounding ImGui::Begin()/End(), so every plot drawn in the same
    // frame piled into the same implicit "Debug" window instead of each
    // getting one of its own.
    TimeSeriesPlot energyPlot("Energy");
    energyPlot.addSample(0.0, 1.0);
    energyPlot.addSample(1.0, 0.99);

    TimeSeriesPlot momentumPlot("Momentum");
    momentumPlot.addSample(0.0, 0.0);
    momentumPlot.addSample(1.0, 1e-9);

    for (int frame = 0; frame < 2; ++frame) {
        ui->beginFrame();
        stats.draw();
        energyPlot.draw();
        momentumPlot.draw();
        ui->endFrame();
    }

    ASSERT_NE(ImGui::GetDrawData(), nullptr);
    EXPECT_GT(ImGui::GetDrawData()->CmdLists.Size, 0);

    EXPECT_NE(ImGui::FindWindowByName("Energy"), nullptr);
    EXPECT_NE(ImGui::FindWindowByName("Momentum"), nullptr);
    EXPECT_EQ(ImGui::FindWindowByName("Debug"), nullptr)
        << "a plot that opened no window of its own falls through to ImGui's "
           "implicit fallback window";
}

TEST(UIImGuiLayer, PersistLayoutSettingControlsIniFile) {
    GLSession session = openGLSession(256, 256);
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    {
        const std::optional<ImGuiLayer> ui = ImGuiLayer::create(*session.window);
        ASSERT_TRUE(ui);
        EXPECT_EQ(ImGui::GetIO().IniFilename, nullptr)
            << "off by default: nothing should persist unless asked to";
    }

    {
        ImGuiLayerSettings settings;
        settings.persistLayout = true;
        const std::optional<ImGuiLayer> ui =
            ImGuiLayer::create(*session.window, settings);
        ASSERT_TRUE(ui);
        EXPECT_NE(ImGui::GetIO().IniFilename, nullptr);
    }
}
