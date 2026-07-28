#include <Renderer/Camera.hpp>
#include <Renderer/Light.hpp>
#include <Renderer/Material.hpp>
#include <Renderer/RayTracer.hpp>
#include <Renderer/Renderer.hpp>

#include <support/GLContext.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

// RayTracer::render draws to whatever framebuffer is currently bound, the
// same convention as Renderer, so this reuses RenderTarget for the readback.
// Each test is built so the geometry alone guarantees the outcome — no pixel
// thresholds tuned by trial and error — which is what makes a raw color
// comparison a trustworthy proof of shadows or reflections rather than a
// coincidence of one particular scene.

namespace {

using ysq::Camera;
using ysq::PointLight;
using ysq::RaytracedScene;
using ysq::RaytracedSphere;
using ysq::RayTracer;
using ysq::RenderTarget;
using ysq::Vec3f;
using ysq::test::GLSession;
using ysq::test::openGLSession;

struct Rgb {
    int r = 0;
    int g = 0;
    int b = 0;
};

Rgb pixelAt(const std::vector<std::uint8_t>& rgba, int width, int x, int y) {
    const std::size_t index =
        (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
         static_cast<std::size_t>(x)) *
        4;
    return {rgba[index], rgba[index + 1], rgba[index + 2]};
}

int brightness(Rgb color) {
    return color.r + color.g + color.b;
}

}  // namespace

TEST(RendererRayTracer, ALitSphereProducesNonBackgroundPixelsAgainstABlackBackground) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RenderTarget> target = RenderTarget::create(32, 32);
    ASSERT_TRUE(target);
    std::optional<RayTracer> tracer = RayTracer::create();
    ASSERT_TRUE(tracer);

    RaytracedScene scene;
    RaytracedSphere sphere;
    sphere.center = Vec3f::zero();
    sphere.radius = 1.0f;
    sphere.material.albedo = {0.8f, 0.2f, 0.2f};
    sphere.material.ambient = 0.2f;
    sphere.material.diffuse = 0.8f;
    scene.spheres.push_back(sphere);

    PointLight light;
    light.position = {5.0f, 5.0f, 5.0f};
    light.intensity = 3.0f;
    scene.pointLights.push_back(light);
    scene.backgroundColor = Vec3f::zero();

    Camera camera;
    camera.position = {0.0f, 0.0f, 4.0f};
    camera.target = Vec3f::zero();

    target->bind();
    tracer->render(scene, camera, 1.0f, target->width(), target->height());
    const std::vector<std::uint8_t> pixels = target->readPixels();
    RenderTarget::bindDefault();

    const Rgb center =
        pixelAt(pixels, target->width(), target->width() / 2, target->height() / 2);
    const Rgb corner = pixelAt(pixels, target->width(), 0, 0);
    EXPECT_GT(brightness(center), 10) << "the sphere fills the center of the frame";
    EXPECT_EQ(corner.r, 0);
    EXPECT_EQ(corner.g, 0);
    EXPECT_EQ(corner.b, 0);
}

// A point directly under an off-axis light, on a plane, seen from directly
// above: the camera ray to that point never passes near the occluder (their
// x-coordinates are 5 apart), but the occluder sits exactly on the segment
// from the light to that point, so the light ray to it is always blocked.
// Rendered with and without the occluder, everything else held fixed, that
// same pixel must get dimmer with it present. See the geometry worked out
// in the review that shaped this test: the occluder's center is the exact
// midpoint of the light-to-point segment, so this is not a near-miss that
// luck could flip.
TEST(RendererRayTracer, AnOccludedSurfaceIsDarkerThanTheSameSurfaceUnoccluded) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RayTracer> tracer = RayTracer::create();
    ASSERT_TRUE(tracer);

    const Vec3f shadedPoint{0.0f, -2.0f, 0.0f};

    RaytracedScene scene;
    ysq::RaytracedPlane plane;
    plane.point = shadedPoint;
    plane.normal = Vec3f::unitY();
    plane.material.albedo = Vec3f::splat(1.0f);
    plane.material.ambient = 0.05f;
    plane.material.diffuse = 0.9f;
    plane.material.specular = 0.0f;
    scene.planes.push_back(plane);

    PointLight light;
    light.position = {5.0f, 5.0f, 0.0f};
    light.intensity = 4.0f;
    scene.pointLights.push_back(light);
    scene.backgroundColor = Vec3f::zero();

    RaytracedSphere occluder;
    occluder.center = {2.5f, 1.5f, 0.0f};  // midpoint of light -> shadedPoint
    occluder.radius = 1.0f;
    occluder.material.ambient = 0.0f;
    occluder.material.diffuse = 0.0f;
    scene.spheres.push_back(occluder);

    Camera camera;
    camera.position = {0.0f, 10.0f, 5.0f};
    camera.target = shadedPoint;  // the frame's center pixel is exactly this point

    std::optional<RenderTarget> target = RenderTarget::create(32, 32);
    ASSERT_TRUE(target);
    const int cx = target->width() / 2;
    const int cy = target->height() / 2;

    // readPixels() leaves the default framebuffer bound, so target must be
    // re-bound before every render() that follows one.
    target->bind();
    tracer->render(scene, camera, 1.0f, target->width(), target->height());
    const int shadowed =
        brightness(pixelAt(target->readPixels(), target->width(), cx, cy));

    scene.spheres.clear();  // same scene, occluder removed
    target->bind();
    tracer->render(scene, camera, 1.0f, target->width(), target->height());
    const int unshadowed =
        brightness(pixelAt(target->readPixels(), target->width(), cx, cy));
    RenderTarget::bindDefault();

    EXPECT_LT(shadowed, unshadowed)
        << "the same surface point must read darker with the occluder blocking its light";
}

// A perfectly reflective sphere seen head-on bounces the primary ray back
// along the direction it came from (reflecting a straight-on ray off a
// sphere's near face returns it the way it arrived), which then escapes to
// the background regardless of camera position, since the background is a
// function of ray direction, not of where the ray started. So its center
// pixel must equal the background exactly; a matte sphere's must not.
TEST(RendererRayTracer, APerfectlyReflectiveSphereShowsTheBackgroundNotItsOwnAlbedo) {
    GLSession session = openGLSession();
    if (!session.opened()) {
        YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED("no OpenGL context: " + session.failure);
    }

    std::optional<RayTracer> tracer = RayTracer::create();
    ASSERT_TRUE(tracer);
    std::optional<RenderTarget> target = RenderTarget::create(32, 32);
    ASSERT_TRUE(target);
    const int cx = target->width() / 2;
    const int cy = target->height() / 2;

    Camera camera;
    camera.position = {0.0f, 0.0f, 4.0f};
    camera.target = Vec3f::zero();

    const Vec3f background{229.0f / 255.0f, 25.0f / 255.0f, 229.0f / 255.0f};

    RaytracedScene mirrorScene;
    RaytracedSphere mirror;
    mirror.center = Vec3f::zero();
    mirror.radius = 1.0f;
    mirror.material.reflectivity = 1.0f;
    mirror.material.ambient = 0.0f;
    mirror.material.diffuse = 0.0f;
    mirrorScene.spheres.push_back(mirror);
    mirrorScene.backgroundColor = background;

    // readPixels() leaves the default framebuffer bound, so target must be
    // re-bound before every render() that follows one.
    target->bind();
    tracer->render(mirrorScene, camera, 1.0f, target->width(), target->height(), 1);
    const Rgb mirrorPixel = pixelAt(target->readPixels(), target->width(), cx, cy);

    RaytracedScene matteScene;
    RaytracedSphere matte;
    matte.center = Vec3f::zero();
    matte.radius = 1.0f;
    matte.material.albedo = {0.0f, 1.0f, 0.0f};
    matte.material.ambient = 0.6f;
    matte.material.reflectivity = 0.0f;
    matteScene.spheres.push_back(matte);
    matteScene.backgroundColor = background;

    target->bind();
    tracer->render(matteScene, camera, 1.0f, target->width(), target->height(), 1);
    const Rgb mattePixel = pixelAt(target->readPixels(), target->width(), cx, cy);
    RenderTarget::bindDefault();

    const Rgb expectedBackground{static_cast<int>(background.x * 255.0f),
                                 static_cast<int>(background.y * 255.0f),
                                 static_cast<int>(background.z * 255.0f)};
    EXPECT_NEAR(mirrorPixel.r, expectedBackground.r, 2);
    EXPECT_NEAR(mirrorPixel.g, expectedBackground.g, 2);
    EXPECT_NEAR(mirrorPixel.b, expectedBackground.b, 2);

    EXPECT_NE(mattePixel.r, mirrorPixel.r);
    EXPECT_NE(mattePixel.g, mirrorPixel.g);
}
