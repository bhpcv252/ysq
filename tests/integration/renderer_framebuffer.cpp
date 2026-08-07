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

// A point light's own attenuation is baked into basic.frag, not exposed as
// a standalone C++ function, so the only way to prove it is really
// inverse-square (Renderer/Light.hpp's own documented contract) is to
// render it and read pixels back, the same way DebugDrawLinesAppearInTheFramebuffer
// below proves debug geometry actually lands rather than trusting the call
// compiled.
TEST(RendererFramebuffer, PointLightAttenuationIsRealInverseSquare) {
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

    Camera camera;
    camera.position = {0.0f, 0.0f, 5.0f};
    camera.target = Vec3f::zero();

    // No ambient or specular: isolates the diffuse term, whose only
    // distance dependence is the light's own attenuation. The sphere's
    // front-center point (0, 0, 1), dead center on screen from this camera,
    // has normal (0, 0, 1) -- directly toward a light placed anywhere along
    // +z -- so dot(normal, L) is exactly 1 regardless of the light's own
    // distance, leaving attenuation as the only thing that can change
    // between the two renders below.
    Material material;
    material.albedo = Vec3f::splat(1.0f);
    material.ambient = 0.0f;
    material.diffuse = 1.0f;
    material.specular = 0.0f;

    const auto renderAtLightDistance = [&](float lightZ) {
        target->bind();
        renderer->beginFrame(camera, 1.0f, target->width(), target->height());
        PointLight light;
        light.position = {0.0f, 0.0f, lightZ};
        light.color = Vec3f::splat(1.0f);
        // Chosen so the nearer case (light-to-surface distance 5) lands at
        // 0.7, comfortably under 1.0: intensity / distance^2 = 17.5 / 25.
        light.intensity = 17.5f;
        const std::array<PointLight, 1> lights{light};
        renderer->setLights(lights, {});
        renderer->draw(*sphere, material, Matrix4<float>::identity());
        renderer->endFrame();
        const std::vector<std::uint8_t> pixels = target->readPixels();
        RenderTarget::bindDefault();
        const std::size_t centerIndex =
            (static_cast<std::size_t>(target->height() / 2) *
                 static_cast<std::size_t>(target->width()) +
             static_cast<std::size_t>(target->width() / 2)) *
            4;
        return static_cast<double>(pixels[centerIndex]);
    };

    // Light-to-surface-point distance 5 (light at z=6, point at z=1) and 10
    // (light at z=11): exactly double, so real inverse-square predicts the
    // farther case reads at 1/4 the nearer one's brightness.
    const double nearBrightness = renderAtLightDistance(6.0f);
    const double farBrightness = renderAtLightDistance(11.0f);

    ASSERT_GT(nearBrightness, 20.0) << "near case should be comfortably above the noise floor";
    const double ratio = nearBrightness / farBrightness;
    EXPECT_NEAR(ratio, 4.0, 0.6) << "doubling the light's distance must quarter the "
                                    "brightness under real inverse-square falloff (near="
                                 << nearBrightness << ", far=" << farBrightness << ")";
}

// Real eclipse/shadow support (Renderer/Mesh.hpp's own
// setInstanceLightMultipliers, driven in a real scenario by
// Physics/Optics/Illumination.hpp's discOcclusionFraction): the same "render
// twice, compare center pixel" technique
// PointLightAttenuationIsRealInverseSquare above uses, here proving a
// per-instance shadow factor of 0.0 actually reads darker than 1.0 on the
// same instance, not just that the call compiles.
TEST(RendererFramebuffer, PerInstanceLightMultiplierActuallyDarkensAnInstance) {
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

    Camera camera;
    camera.position = {0.0f, 0.0f, 5.0f};
    camera.target = Vec3f::zero();

    Material material;
    material.albedo = Vec3f::splat(1.0f);
    material.ambient = 0.1f;
    material.diffuse = 0.9f;
    material.specular = 0.0f;

    const auto renderAtLightMultiplier = [&](float lightMultiplier) {
        target->bind();
        renderer->beginFrame(camera, 1.0f, target->width(), target->height());
        PointLight light;
        light.position = {0.0f, 0.0f, 10.0f};
        light.color = Vec3f::splat(1.0f);
        light.intensity = 100.0f;
        const std::array<PointLight, 1> lights{light};
        renderer->setLights(lights, {});

        const std::array<Matrix4<float>, 1> transforms{Matrix4<float>::identity()};
        sphere->setInstanceTransforms(transforms);
        const std::array<float, 1> lightMultipliers{lightMultiplier};
        sphere->setInstanceLightMultipliers(lightMultipliers);
        renderer->drawInstanced(*sphere, material);
        renderer->endFrame();

        const std::vector<std::uint8_t> pixels = target->readPixels();
        RenderTarget::bindDefault();
        const std::size_t centerIndex =
            (static_cast<std::size_t>(target->height() / 2) *
                 static_cast<std::size_t>(target->width()) +
             static_cast<std::size_t>(target->width() / 2)) *
            4;
        return static_cast<int>(pixels[centerIndex]);
    };

    const int litBrightness = renderAtLightMultiplier(1.0f);
    const int shadowedBrightness = renderAtLightMultiplier(0.0f);

    EXPECT_GT(litBrightness, shadowedBrightness)
        << "lightMultiplier=0 must read darker than lightMultiplier=1 on the same instance (lit="
        << litBrightness << ", shadowed=" << shadowedBrightness << ")";
    // lightMultiplier=0 should leave only the ambient term (material.ambient
    // = 0.1, well under the fully-lit ambient+diffuse response), not zero
    // it out entirely -- an eclipsed body still has ambient light.
    EXPECT_GT(shadowedBrightness, 0);
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
