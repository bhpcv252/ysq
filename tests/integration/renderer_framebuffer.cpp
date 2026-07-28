#include <Renderer/Camera.hpp>
#include <Renderer/Material.hpp>
#include <Renderer/Mesh.hpp>
#include <Renderer/Renderer.hpp>

#include <support/GLContext.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// "Geometry draws to an offscreen framebuffer" is Stage 7's own done
// criterion, so this is that criterion made literal: real Mesh/Material/
// Camera objects, a real RenderTarget, and a pixel readback that proves
// something actually landed there. Whether a context can exist at all is a
// property of the machine, not the code; see support/GLContext.hpp.

namespace {

using ysq::Camera;
using ysq::Material;
using ysq::Matrix4;
using ysq::Mesh;
using ysq::PointLight;
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

TEST(RendererFramebuffer, ALitSphereDrawsToAnOffscreenFramebuffer) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RenderTarget> target = RenderTarget::create(64, 64);
    ASSERT_TRUE(target);
    std::optional<Renderer> renderer = Renderer::create();
    ASSERT_TRUE(renderer);
    std::optional<Mesh> sphere = Mesh::sphere();
    ASSERT_TRUE(sphere);

    target->bind();

    Camera camera;
    camera.position = {0.0f, 0.0f, 5.0f};
    camera.target = Vec3f::zero();

    Material material;
    material.albedo = {1.0f, 0.2f, 0.2f};
    material.ambient = 0.2f;
    material.diffuse = 0.8f;

    PointLight light;
    light.position = {5.0f, 5.0f, 5.0f};
    light.color = Vec3f::splat(1.0f);
    light.intensity = 2.0f;
    const std::array<PointLight, 1> lights{light};

    renderer->beginFrame(camera, 1.0f, target->width(), target->height());
    renderer->setLights(lights, {});
    renderer->draw(*sphere, material, Matrix4<float>::identity());
    renderer->endFrame();

    const std::vector<std::uint8_t> pixels = target->readPixels();
    EXPECT_TRUE(anyPixelBrighterThan(pixels, 10))
        << "a lit sphere filling most of the frame must not read back as a black clear";

    RenderTarget::bindDefault();
}

TEST(RendererFramebuffer, InstancedDrawingIssuesOneDrawCallForManyBodies) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RenderTarget> target = RenderTarget::create(64, 64);
    ASSERT_TRUE(target);
    std::optional<Renderer> renderer = Renderer::create();
    ASSERT_TRUE(renderer);
    std::optional<Mesh> sphere = Mesh::sphere(0.3f, 8, 16);
    ASSERT_TRUE(sphere);

    const std::array<Matrix4<float>, 3> transforms{
        Matrix4<float>::translation({-1.0f, 0.0f, 0.0f}),
        Matrix4<float>::translation({0.0f, 0.0f, 0.0f}),
        Matrix4<float>::translation({1.0f, 0.0f, 0.0f}),
    };
    sphere->setInstanceTransforms(transforms);

    target->bind();

    Camera camera;
    camera.position = {0.0f, 0.0f, 5.0f};
    camera.target = Vec3f::zero();
    camera.perspectiveSettings.fovYRadians = 1.2f;

    // Self-lit, so this is provably lit regardless of light placement.
    Material material;
    material.emissive = Vec3f::splat(0.8f);
    material.ambient = 0.0f;
    material.diffuse = 0.0f;

    renderer->beginFrame(camera, 1.0f, target->width(), target->height());
    renderer->drawInstanced(*sphere, material);
    EXPECT_EQ(renderer->drawCallCount(), 1u)
        << "three instances of one mesh must still be a single draw call";
    renderer->endFrame();

    const std::vector<std::uint8_t> pixels = target->readPixels();
    EXPECT_TRUE(anyPixelBrighterThan(pixels, 10));

    RenderTarget::bindDefault();
}

TEST(RendererFramebuffer, DebugDrawLinesAppearInTheFramebuffer) {
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
    camera.position = {2.0f, 2.0f, 2.0f};
    camera.target = Vec3f::zero();

    renderer->beginFrame(camera, 1.0f, target->width(), target->height());
    renderer->debugDraw().axes(2.0f);
    renderer->debugDraw().grid(5.0f, 10);
    renderer->endFrame();

    const std::vector<std::uint8_t> pixels = target->readPixels();
    EXPECT_TRUE(anyPixelBrighterThan(pixels, 10))
        << "axes and a grid through the origin must leave some trace against a black "
           "clear";

    RenderTarget::bindDefault();
}

// readPixels() against a multisampled target directly is meaningless (GL
// rejects it outright); resolveTo() is the only supported path from a
// multisampled render to actual pixels, and this is what exercises it.
TEST(RendererFramebuffer, AMultisampledTargetResolvesToReadablePixels) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RenderTarget> msaaTarget = RenderTarget::create(64, 64, /*samples=*/4);
    if (!msaaTarget) {
        // Unlike "no context at all", this isn't gated by
        // YSQ_REQUIRE_HEADLESS_GL: that flag promises a context exists, not
        // that every optional capability does. A software rasterizer (e.g.
        // OSMesa on headless CI) may simply not support multisampled
        // textures, and that is a legitimate skip, not a failure.
        GTEST_SKIP() << "multisampled framebuffers not supported on this machine";
    }
    EXPECT_TRUE(msaaTarget->multisampled());
    std::optional<RenderTarget> resolved = RenderTarget::create(64, 64);
    ASSERT_TRUE(resolved);
    EXPECT_FALSE(resolved->multisampled());

    std::optional<Renderer> renderer = Renderer::create();
    ASSERT_TRUE(renderer);
    std::optional<Mesh> sphere = Mesh::sphere();
    ASSERT_TRUE(sphere);

    msaaTarget->bind();

    Camera camera;
    camera.position = {0.0f, 0.0f, 5.0f};
    camera.target = Vec3f::zero();

    Material material;
    material.emissive = Vec3f::splat(0.8f);
    material.ambient = 0.0f;
    material.diffuse = 0.0f;

    renderer->beginFrame(camera, 1.0f, msaaTarget->width(), msaaTarget->height());
    renderer->draw(*sphere, material, Matrix4<float>::identity());
    renderer->endFrame();

    msaaTarget->resolveTo(*resolved);
    const std::vector<std::uint8_t> pixels = resolved->readPixels();
    EXPECT_TRUE(anyPixelBrighterThan(pixels, 10))
        << "a resolved multisampled render must read back the same as a direct one";

    RenderTarget::bindDefault();
}
