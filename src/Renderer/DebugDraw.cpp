#include <Renderer/DebugDraw.hpp>

#include <Renderer/shaders/Debug.frag.hpp>
#include <Renderer/shaders/Debug.vert.hpp>

#include <glad/gl.h>

#include <cmath>
#include <cstddef>

namespace ysq {

void DebugDraw::setupAttributes(unsigned vao, unsigned vbo) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<const void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<const void*>(offsetof(Vertex, color)));
    glBindVertexArray(0);
}

std::optional<DebugDraw> DebugDraw::create(std::string* error) {
    std::optional<Shader> shader =
        Shader::compile(shaders::kDebugVertSource, shaders::kDebugFragSource, error);
    if (!shader) {
        return std::nullopt;
    }

    unsigned lineVao = 0;
    unsigned lineVbo = 0;
    unsigned pointVao = 0;
    unsigned pointVbo = 0;
    glGenVertexArrays(1, &lineVao);
    glGenBuffers(1, &lineVbo);
    glGenVertexArrays(1, &pointVao);
    glGenBuffers(1, &pointVbo);
    setupAttributes(lineVao, lineVbo);
    setupAttributes(pointVao, pointVbo);

    return std::optional<DebugDraw>{
        DebugDraw{std::move(*shader), lineVao, lineVbo, pointVao, pointVbo}};
}

DebugDraw::DebugDraw(DebugDraw&& other) noexcept
    : m_shader(std::move(other.m_shader)),
      m_lineVao(std::exchange(other.m_lineVao, 0u)),
      m_lineVbo(std::exchange(other.m_lineVbo, 0u)),
      m_pointVao(std::exchange(other.m_pointVao, 0u)),
      m_pointVbo(std::exchange(other.m_pointVbo, 0u)),
      m_lineVertices(std::move(other.m_lineVertices)),
      m_pointVertices(std::move(other.m_pointVertices)) {}

DebugDraw& DebugDraw::operator=(DebugDraw&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();
    m_shader = std::move(other.m_shader);
    m_lineVao = std::exchange(other.m_lineVao, 0u);
    m_lineVbo = std::exchange(other.m_lineVbo, 0u);
    m_pointVao = std::exchange(other.m_pointVao, 0u);
    m_pointVbo = std::exchange(other.m_pointVbo, 0u);
    m_lineVertices = std::move(other.m_lineVertices);
    m_pointVertices = std::move(other.m_pointVertices);
    return *this;
}

DebugDraw::~DebugDraw() {
    destroy();
}

void DebugDraw::destroy() noexcept {
    if (m_lineVbo != 0) {
        glDeleteBuffers(1, &m_lineVbo);
    }
    if (m_lineVao != 0) {
        glDeleteVertexArrays(1, &m_lineVao);
    }
    if (m_pointVbo != 0) {
        glDeleteBuffers(1, &m_pointVbo);
    }
    if (m_pointVao != 0) {
        glDeleteVertexArrays(1, &m_pointVao);
    }
    m_lineVao = m_lineVbo = m_pointVao = m_pointVbo = 0;
}

void DebugDraw::line(const Vec3f& a, const Vec3f& b, const Vec3f& color) {
    m_lineVertices.push_back({a, color});
    m_lineVertices.push_back({b, color});
}

void DebugDraw::point(const Vec3f& p, const Vec3f& color) {
    m_pointVertices.push_back({p, color});
}

void DebugDraw::arrow(const Vec3f& origin, const Vec3f& direction, const Vec3f& color) {
    const Vec3f tip = origin + direction;
    line(origin, tip, color);

    const std::optional<Vec3f> dir = tryNormalized(direction);
    if (!dir) {
        return;
    }
    // Any vector not parallel to *dir seeds a perpendicular basis for the
    // arrowhead's two short strokes.
    const Vec3f seed = (std::abs(dir->y) < 0.99f) ? Vec3f::unitY() : Vec3f::unitX();
    const Vec3f side = normalized(cross(*dir, seed));
    const float headLength = length(direction) * 0.15f;
    const Vec3f headBase = tip - *dir * headLength;
    line(tip, headBase + side * headLength * 0.5f, color);
    line(tip, headBase - side * headLength * 0.5f, color);
}

void DebugDraw::grid(float extent, int divisions, const Vec3f& color) {
    if (divisions < 1) {
        return;
    }
    const float step = (2.0f * extent) / static_cast<float>(divisions);
    for (int i = 0; i <= divisions; ++i) {
        const float offset = -extent + step * static_cast<float>(i);
        line({offset, -extent, 0.0f}, {offset, extent, 0.0f}, color);
        line({-extent, offset, 0.0f}, {extent, offset, 0.0f}, color);
    }
}

void DebugDraw::axes(float length) {
    line(Vec3f::zero(), Vec3f::unitX() * length, {1.0f, 0.0f, 0.0f});
    line(Vec3f::zero(), Vec3f::unitY() * length, {0.0f, 1.0f, 0.0f});
    line(Vec3f::zero(), Vec3f::unitZ() * length, {0.0f, 0.0f, 1.0f});
}

void DebugDraw::flush(const Matrix4<float>& viewProjection) {
    m_shader.use();
    m_shader.setUniform("uViewProjection", viewProjection);

    if (!m_lineVertices.empty()) {
        glBindVertexArray(m_lineVao);
        glBindBuffer(GL_ARRAY_BUFFER, m_lineVbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(m_lineVertices.size() * sizeof(Vertex)),
                     m_lineVertices.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_lineVertices.size()));
    }
    if (!m_pointVertices.empty()) {
        // Program-controlled point size lives only inside this scope: it is
        // global GL state, and leaving it enabled would silently affect any
        // other point draw elsewhere in a frame.
        glEnable(GL_PROGRAM_POINT_SIZE);
        glBindVertexArray(m_pointVao);
        glBindBuffer(GL_ARRAY_BUFFER, m_pointVbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(m_pointVertices.size() * sizeof(Vertex)),
                     m_pointVertices.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_pointVertices.size()));
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
    glBindVertexArray(0);

    m_lineVertices.clear();
    m_pointVertices.clear();
}

}  // namespace ysq
