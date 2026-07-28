#pragma once

#include <Math/Matrix4.hpp>
#include <Math/Vector3.hpp>
#include <Renderer/Shader.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ysq {

/// Immediate-mode line and point drawing: orbit trails, vector fields,
/// geodesics, velocity/force vectors, grids and axes. Every physics theory
/// in this engine eventually wants to show a curve or a vector, and this is
/// the one place that does it.
///
/// Calls accumulate a vertex batch; flush() uploads it once and issues at
/// most two draw calls, so drawing ten thousand debug lines a frame is one
/// buffer update, not ten thousand `glDrawArrays` calls. The context this was
/// created under must already be current for every method here, the same
/// rule as Shader.
class DebugDraw {
public:
    [[nodiscard]] static std::optional<DebugDraw> create(std::string* error = nullptr);

    DebugDraw(const DebugDraw&) = delete;
    DebugDraw& operator=(const DebugDraw&) = delete;
    DebugDraw(DebugDraw&& other) noexcept;
    DebugDraw& operator=(DebugDraw&& other) noexcept;
    ~DebugDraw();

    void line(const Vec3f& a, const Vec3f& b, const Vec3f& color = Vec3f::splat(1.0f));
    void point(const Vec3f& p, const Vec3f& color = Vec3f::splat(1.0f));

    /// A line from `origin` in `direction`, with a small arrowhead at the tip
    /// — a force, a velocity, a field sample.
    void arrow(const Vec3f& origin, const Vec3f& direction,
               const Vec3f& color = Vec3f::splat(1.0f));

    /// A flat grid in the XY plane, centered at the origin, `divisions`
    /// lines per side.
    void grid(float extent, int divisions, const Vec3f& color = Vec3f::splat(0.4f));

    /// Unit-length X (red), Y (green) and Z (blue) axes through the origin.
    void axes(float length = 1.0f);

    /// Uploads the accumulated batch, draws it with `viewProjection`, and
    /// clears the batch for the next frame. Point size is fixed in
    /// shaders/debug.vert; there is no per-call size, since GL_PROGRAM_POINT_SIZE
    /// state is what would otherwise leak across an unrelated draw call.
    void flush(const Matrix4<float>& viewProjection);

private:
    struct Vertex {
        Vec3f position;
        Vec3f color;
    };

    DebugDraw(Shader shader, unsigned lineVao, unsigned lineVbo, unsigned pointVao,
              unsigned pointVbo) noexcept
        : m_shader(std::move(shader)),
          m_lineVao(lineVao),
          m_lineVbo(lineVbo),
          m_pointVao(pointVao),
          m_pointVbo(pointVbo) {}
    void destroy() noexcept;
    static void setupAttributes(unsigned vao, unsigned vbo);

    Shader m_shader;
    unsigned m_lineVao = 0;
    unsigned m_lineVbo = 0;
    unsigned m_pointVao = 0;
    unsigned m_pointVbo = 0;
    std::vector<Vertex> m_lineVertices;
    std::vector<Vertex> m_pointVertices;
};

}  // namespace ysq
