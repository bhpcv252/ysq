#pragma once

#include <Math/Vector3.hpp>

namespace ysq {

/// A light with a position: attenuates as `intensity / distance^2`, real
/// inverse-square falloff in whatever units the scene's own positions are
/// expressed in (render units, meters, whatever a caller chose) -- there is
/// no separate tunable here, because the real law does not have one; a
/// caller wanting a certain body to read at a certain brightness picks
/// `intensity` for that, the same way a photographer picks exposure rather
/// than the inverse-square law itself. Shared by Renderer's forward shader
/// and RayTracer's shading, so both light a scene the same way regardless
/// of where the trace runs; see src/Renderer/README.md.
struct PointLight {
    Vec3f position = Vec3f::zero();
    Vec3f color = Vec3f::splat(1.0f);
    float intensity = 1.0f;
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
