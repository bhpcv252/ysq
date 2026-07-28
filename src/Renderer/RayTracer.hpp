#pragma once

#include <Renderer/Camera.hpp>
#include <Renderer/Light.hpp>
#include <Renderer/Material.hpp>
#include <Renderer/Mesh.hpp>
#include <Renderer/Shader.hpp>
#include <Renderer/Texture.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ysq {

struct RaytracedSphere {
    Vec3f center = Vec3f::zero();
    float radius = 1.0f;
    Material material{};
};

struct RaytracedPlane {
    Vec3f point = Vec3f::zero();
    Vec3f normal = Vec3f::unitY();
    Material material{};
};

struct RaytracedDisk {
    Vec3f center = Vec3f::zero();
    Vec3f normal = Vec3f::unitY();
    float innerRadius = 1.0f;
    float outerRadius = 2.0f;
    Material material{};
};

/// The scene a RayTracer traces: a small set of analytic primitives and
/// lights. RayTracer::render() truncates silently at the Max* capacities
/// below rather than failing, so a scene that outgrows one frame's uniform
/// arrays degrades instead of stopping the program mid-run.
struct RaytracedScene {
    std::vector<RaytracedSphere> spheres;
    std::vector<RaytracedPlane> planes;
    std::vector<RaytracedDisk> disks;
    std::vector<PointLight> pointLights;
    std::vector<DirectionalLight> directionalLights;
    /// Sampled by rays that hit nothing in the scene, primary or reflected.
    /// Null falls back to backgroundColor.
    const Cubemap* environment = nullptr;
    Vec3f backgroundColor = Vec3f::zero();
};

/// A full-screen-quad fragment-shader ray tracer: shadows and reflections
/// against an analytic scene. Not a compute shader — compute shaders are a
/// 4.3 feature and this stays portable to OpenGL 4.1 (macOS), the same
/// portability call docs/architecture.md flagged before Renderer existed.
/// See docs/rendering.md.
///
/// Renders into whatever framebuffer is currently bound, same convention as
/// Renderer; the context that created this must already be current for every
/// method here.
class RayTracer {
public:
    [[nodiscard]] static std::optional<RayTracer> create(std::string* error = nullptr);

    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;
    RayTracer(RayTracer&&) noexcept = default;
    RayTracer& operator=(RayTracer&&) noexcept = default;
    ~RayTracer() = default;

    /// `maxBounces` is clamped to maxBounceDepth() by the shader itself, so a
    /// caller cannot ask for more than one frame can afford.
    void render(const RaytracedScene& scene, const Camera& camera, float aspect,
                int viewportWidth, int viewportHeight, int maxBounces = 4);

    [[nodiscard]] static constexpr std::size_t maxSpheres() noexcept { return 32; }
    [[nodiscard]] static constexpr std::size_t maxPlanes() noexcept { return 8; }
    [[nodiscard]] static constexpr std::size_t maxDisks() noexcept { return 8; }
    [[nodiscard]] static constexpr std::size_t maxPointLights() noexcept { return 8; }
    [[nodiscard]] static constexpr std::size_t maxDirectionalLights() noexcept {
        return 4;
    }
    [[nodiscard]] static constexpr int maxBounceDepth() noexcept { return 8; }

private:
    RayTracer(Shader shader, Mesh quad) noexcept
        : m_shader(std::move(shader)), m_quad(std::move(quad)) {}

    Shader m_shader;
    Mesh m_quad;
};

}  // namespace ysq
