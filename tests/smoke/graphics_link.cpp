#include <gtest/gtest.h>

#define GLFW_INCLUDE_NONE  // GLAD provides the GL headers, not GLFW
#include <GLFW/glfw3.h>
#include <glad/gl.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <string_view>

// Compiles and links against GLAD, GLFW and Dear ImGui, and exercises only the
// entry points that need no window, no GL context and no display, so this runs
// headless in CI. It proves the wiring; it renders nothing.

namespace {

// Referencing these forces the loader and the ImGui backends to be pulled out
// of their static libraries and resolved, which is the actual thing under test.
auto* const kGladLoadGL = &gladLoadGL;
auto* const kImGuiGlfwInit = &ImGui_ImplGlfw_InitForOpenGL;
auto* const kImGuiOpenGL3Init = &ImGui_ImplOpenGL3_Init;

}  // namespace

TEST(GraphicsSmoke, GladLoaderLinks) {
    EXPECT_TRUE(kGladLoadGL != nullptr);
}

TEST(GraphicsSmoke, GlfwIsThePinnedVersion) {
    // glfwGetVersion is documented to work before glfwInit, so no display needed.
    int major = 0;
    int minor = 0;
    int revision = 0;
    glfwGetVersion(&major, &minor, &revision);

    EXPECT_EQ(major, 3);
    EXPECT_EQ(minor, 4);
}

TEST(GraphicsSmoke, ImGuiContextLifecycle) {
    IMGUI_CHECKVERSION();

    ImGuiContext* context = ImGui::CreateContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(std::string_view{ImGui::GetVersion()}, std::string_view{IMGUI_VERSION});
    ImGui::DestroyContext(context);
}

TEST(GraphicsSmoke, ImGuiBackendsLink) {
    EXPECT_TRUE(kImGuiGlfwInit != nullptr);
    EXPECT_TRUE(kImGuiOpenGL3Init != nullptr);
}
