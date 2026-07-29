#include <Renderer/Camera.hpp>
#include <Renderer/Renderer.hpp>

#include <support/GLContext.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

// DebugDraw::text()/textFixed() rasterize onto a real offscreen framebuffer,
// the same pixel-readback proof renderer_framebuffer.cpp uses for meshes and
// debug lines. Whether a context can exist at all is a property of the
// machine, not the code; see support/GLContext.hpp.

namespace {

using ysq::Camera;
using ysq::Renderer;
using ysq::RenderTarget;
using ysq::Vec3f;
using ysq::test::GLSession;
using ysq::test::openGLSession;

bool anyPixelBrighterThan(const std::vector<std::uint8_t>& rgba, std::uint8_t threshold) {
    for (std::size_t i = 0; i + 2 < rgba.size(); i += 4) {
        if (rgba[i] > threshold || rgba[i + 1] > threshold || rgba[i + 2] > threshold) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST(RendererText, BillboardTextAppearsInTheFramebuffer) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RenderTarget> target = RenderTarget::create(64, 64);
    ASSERT_TRUE(target);
    std::optional<Renderer> renderer = Renderer::create();
    ASSERT_TRUE(renderer);

    target->bind();

    Camera camera;
    camera.position = {0.0f, 0.0f, 5.0f};
    camera.target = Vec3f::zero();

    renderer->beginFrame(camera, 1.0f, target->width(), target->height());
    renderer->debugDraw().text(Vec3f::zero(), "Hi", 2.0f);
    renderer->endFrame();

    const std::vector<std::uint8_t> pixels = target->readPixels();
    EXPECT_TRUE(anyPixelBrighterThan(pixels, 10))
        << "a label facing the camera must leave some trace against a black clear";

    RenderTarget::bindDefault();
}

TEST(RendererText, FixedOrientationTextAlsoAppearsInTheFramebuffer) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RenderTarget> target = RenderTarget::create(64, 64);
    ASSERT_TRUE(target);
    std::optional<Renderer> renderer = Renderer::create();
    ASSERT_TRUE(renderer);

    target->bind();

    Camera camera;
    camera.position = {0.0f, 0.0f, 5.0f};
    camera.target = Vec3f::zero();

    renderer->beginFrame(camera, 1.0f, target->width(), target->height());
    renderer->debugDraw().textFixed(Vec3f::zero(), Vec3f::unitX(), Vec3f::unitY(), "Hi",
                                    2.0f);
    renderer->endFrame();

    const std::vector<std::uint8_t> pixels = target->readPixels();
    EXPECT_TRUE(anyPixelBrighterThan(pixels, 10))
        << "fixed-orientation text facing the camera must also leave some trace";

    RenderTarget::bindDefault();
}

TEST(RendererText, EmptyAndAllUnsupportedTextDrawNothingWithoutCrashing) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RenderTarget> target = RenderTarget::create(64, 64);
    ASSERT_TRUE(target);
    std::optional<Renderer> renderer = Renderer::create();
    ASSERT_TRUE(renderer);

    target->bind();

    Camera camera;
    camera.position = {0.0f, 0.0f, 5.0f};
    camera.target = Vec3f::zero();

    renderer->beginFrame(camera, 1.0f, target->width(), target->height());
    renderer->debugDraw().text(Vec3f::zero(), "");
    renderer->debugDraw().text(Vec3f::zero(), "\x01\x02");
    renderer->endFrame();

    const std::vector<std::uint8_t> pixels = target->readPixels();
    EXPECT_FALSE(anyPixelBrighterThan(pixels, 10))
        << "no renderable glyphs means nothing should have drawn";

    RenderTarget::bindDefault();
}
