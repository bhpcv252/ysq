#pragma once

#include <Math/Matrix4.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>

#include <cstddef>
#include <optional>
#include <span>

namespace ysq {

struct Vertex {
    Vec3f position;
    Vec3f normal;
    Vec2f uv;
};

/// A GPU vertex/index buffer pair (VAO+VBO+EBO), RAII, move-only.
///
/// Geometry only: no material, no transform. Renderer combines a Mesh with a
/// Material and a model matrix at draw time, which is what lets the same
/// sphere mesh be a planet in one draw call and a star in the next.
class Mesh {
public:
    [[nodiscard]] static std::optional<Mesh> create(std::span<const Vertex> vertices,
                                                    std::span<const unsigned> indices);

    /// A UV sphere: `latitudeSegments` rings from pole to pole,
    /// `longitudeSegments` around each ring. What every body in this engine
    /// renders as, whatever its actual scale.
    [[nodiscard]] static std::optional<Mesh>
    sphere(float radius = 1.0f, int latitudeSegments = 24, int longitudeSegments = 48);

    /// A flat square in the XY plane, facing +Z. The full-screen pass
    /// RayTracer draws into is two triangles of this in clip space; a
    /// world-space quad (a light gizmo, a ground plane) is the same mesh.
    [[nodiscard]] static std::optional<Mesh> quad(float size = 1.0f);

    /// An annulus in the XY plane, facing +Z: an accretion disk or a
    /// planetary ring.
    [[nodiscard]] static std::optional<Mesh> disk(float innerRadius, float outerRadius,
                                                  int segments = 64);

    [[nodiscard]] static std::optional<Mesh> cube(float size = 1.0f);

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;
    ~Mesh();

    /// Uploads a per-instance model matrix buffer for drawInstanced(). Each
    /// Matrix4<float> occupies four consecutive vertex attribute slots (one
    /// vec4 per column), the layout shaders/instanced.vert expects. Replaces
    /// whatever was set before; an empty span makes drawInstanced() draw
    /// nothing rather than reading stale data. Also (re)fills the per-instance
    /// light multiplier buffer (see setInstanceLightMultipliers below) to all
    /// 1.0 -- fully lit, uncompensated -- at the new instance count, so a
    /// caller that never calls setInstanceLightMultipliers at all sees
    /// exactly today's behavior.
    void setInstanceTransforms(std::span<const Matrix4<float>> transforms);

    /// A per-instance scale on the light-dependent (diffuse and specular,
    /// not ambient or emissive) part of drawInstanced()'s own shading. Not
    /// bounded to [0, 1] -- two different real reasons a caller might scale
    /// an instance's own light this way: under 1.0 for real, geometric
    /// eclipse/shadow support (a caller computes each instance's own real
    /// occlusion, see Physics/Optics/Illumination.hpp's own
    /// discOcclusionFraction, and uploads it here) without a shadow-mapping pass,
    /// or above 1.0 for a real, distance-based exposure compensation (see
    /// Material::lightMultiplier's own doc comment for why that is real,
    /// not fake brightness). `factors.size()` must equal the instance count
    /// setInstanceTransforms was last called with, same order. Call after
    /// setInstanceTransforms, which otherwise leaves every instance at the
    /// fully-lit, uncompensated default.
    void setInstanceLightMultipliers(std::span<const float> factors);

    void draw() const;
    void drawInstanced() const;

    [[nodiscard]] std::size_t indexCount() const noexcept { return m_indexCount; }
    [[nodiscard]] std::size_t instanceCount() const noexcept { return m_instanceCount; }

private:
    Mesh(unsigned vao, unsigned vbo, unsigned ebo, std::size_t indexCount) noexcept
        : m_vao(vao), m_vbo(vbo), m_ebo(ebo), m_indexCount(indexCount) {}
    void destroy() noexcept;

    unsigned m_vao = 0;
    unsigned m_vbo = 0;
    unsigned m_ebo = 0;
    unsigned m_instanceVbo = 0;
    unsigned m_instanceLightMultiplierVbo = 0;
    std::size_t m_indexCount = 0;
    std::size_t m_instanceCount = 0;
};

}  // namespace ysq
