#pragma once

#include <Math/Vector3.hpp>

namespace ysq {

/// A light with a position: attenuates with distance. Shared by Renderer's
/// forward shader and RayTracer's shading, so both light a scene the same
/// way regardless of where the trace runs; see docs/rendering.md.
struct PointLight {
    Vec3f position = Vec3f::zero();
    Vec3f color = Vec3f::splat(1.0f);
    float intensity = 1.0f;
    /// Distance at which the inverse-square falloff has halved the light's
    /// contribution. Zero disables attenuation entirely.
    float radius = 0.0f;
};

/// A light with no position, only a direction: sunlight, or any source far
/// enough that its rays are effectively parallel. Does not attenuate.
struct DirectionalLight {
    /// Points from the light toward the scene, not toward the light.
    Vec3f direction = {0.0f, -1.0f, 0.0f};
    Vec3f color = Vec3f::splat(1.0f);
    float intensity = 1.0f;
};

}  // namespace ysq
