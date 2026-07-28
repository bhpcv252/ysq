#include <Renderer/Mesh.hpp>

#include <glad/gl.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ysq {

namespace {

void bindVertexAttributes() {
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<const void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<const void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<const void*>(offsetof(Vertex, uv)));
}

}  // namespace

std::optional<Mesh> Mesh::create(std::span<const Vertex> vertices,
                                 std::span<const unsigned> indices) {
    if (vertices.empty() || indices.empty()) {
        return std::nullopt;
    }

    unsigned vao = 0;
    unsigned vbo = 0;
    unsigned ebo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size_bytes()),
                 vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size_bytes()),
                 indices.data(), GL_STATIC_DRAW);
    bindVertexAttributes();
    glBindVertexArray(0);

    return std::optional<Mesh>{Mesh{vao, vbo, ebo, indices.size()}};
}

std::optional<Mesh> Mesh::sphere(float radius, int latitudeSegments,
                                 int longitudeSegments) {
    if (latitudeSegments < 2 || longitudeSegments < 3) {
        return std::nullopt;
    }

    std::vector<Vertex> vertices;
    vertices.reserve(static_cast<std::size_t>(latitudeSegments + 1) *
                     static_cast<std::size_t>(longitudeSegments + 1));
    for (int lat = 0; lat <= latitudeSegments; ++lat) {
        const float v = static_cast<float>(lat) / static_cast<float>(latitudeSegments);
        const float theta = v * kPi<float>;  // 0 at the north pole, pi at the south
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);
        for (int lon = 0; lon <= longitudeSegments; ++lon) {
            const float u =
                static_cast<float>(lon) / static_cast<float>(longitudeSegments);
            const float phi = u * kTau<float>;
            const Vec3f normal{sinTheta * std::cos(phi), cosTheta,
                               sinTheta * std::sin(phi)};
            vertices.push_back(Vertex{normal * radius, normal, {u, v}});
        }
    }

    std::vector<unsigned> indices;
    const auto stride = static_cast<unsigned>(longitudeSegments + 1);
    for (unsigned lat = 0; lat < static_cast<unsigned>(latitudeSegments); ++lat) {
        for (unsigned lon = 0; lon < static_cast<unsigned>(longitudeSegments); ++lon) {
            const unsigned a = lat * stride + lon;
            const unsigned b = a + stride;
            indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    }

    return create(vertices, indices);
}

std::optional<Mesh> Mesh::quad(float size) {
    const float h = size / 2.0f;
    const std::array<Vertex, 4> vertices{{
        {{-h, -h, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
        {{h, -h, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
        {{h, h, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        {{-h, h, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    }};
    const std::array<unsigned, 6> indices{0, 1, 2, 2, 3, 0};
    return create(vertices, indices);
}

std::optional<Mesh> Mesh::disk(float innerRadius, float outerRadius, int segments) {
    if (segments < 3 || innerRadius < 0.0f || outerRadius <= innerRadius) {
        return std::nullopt;
    }

    std::vector<Vertex> vertices;
    vertices.reserve(static_cast<std::size_t>(segments + 1) * 2);
    for (int i = 0; i <= segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segments);
        const float angle = t * kTau<float>;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        constexpr Vec3f normal{0.0f, 0.0f, 1.0f};
        vertices.push_back(
            Vertex{{innerRadius * c, innerRadius * s, 0.0f}, normal, {t, 0.0f}});
        vertices.push_back(
            Vertex{{outerRadius * c, outerRadius * s, 0.0f}, normal, {t, 1.0f}});
    }

    std::vector<unsigned> indices;
    indices.reserve(static_cast<std::size_t>(segments) * 6);
    for (unsigned i = 0; i < static_cast<unsigned>(segments); ++i) {
        const unsigned a = i * 2;
        indices.insert(indices.end(), {a, a + 1, a + 2, a + 1, a + 3, a + 2});
    }

    return create(vertices, indices);
}

std::optional<Mesh> Mesh::cube(float size) {
    struct Face {
        Vec3f normal;
        Vec3f u;
        Vec3f v;
    };
    // Each face's (u, v) frame is chosen so bottom-left, bottom-right,
    // top-right, top-left winds counter-clockwise as seen from outside the
    // cube, matching every other mesh generator here.
    const std::array<Face, 6> faces{{
        {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},    // +Z
        {{0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},  // -Z
        {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},   // +X
        {{-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},   // -X
        {{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},   // +Y
        {{0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},   // -Y
    }};

    const float h = size / 2.0f;
    std::vector<Vertex> vertices;
    std::vector<unsigned> indices;
    vertices.reserve(24);
    indices.reserve(36);
    for (const Face& face : faces) {
        const Vec3f center = face.normal * h;
        const auto base = static_cast<unsigned>(vertices.size());
        vertices.push_back({center - face.u * h - face.v * h, face.normal, {0.0f, 0.0f}});
        vertices.push_back({center + face.u * h - face.v * h, face.normal, {1.0f, 0.0f}});
        vertices.push_back({center + face.u * h + face.v * h, face.normal, {1.0f, 1.0f}});
        vertices.push_back({center - face.u * h + face.v * h, face.normal, {0.0f, 1.0f}});
        indices.insert(indices.end(),
                       {base, base + 1, base + 2, base + 2, base + 3, base});
    }

    return create(vertices, indices);
}

Mesh::Mesh(Mesh&& other) noexcept
    : m_vao(std::exchange(other.m_vao, 0u)),
      m_vbo(std::exchange(other.m_vbo, 0u)),
      m_ebo(std::exchange(other.m_ebo, 0u)),
      m_instanceVbo(std::exchange(other.m_instanceVbo, 0u)),
      m_indexCount(std::exchange(other.m_indexCount, std::size_t{0})),
      m_instanceCount(std::exchange(other.m_instanceCount, std::size_t{0})) {}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();
    m_vao = std::exchange(other.m_vao, 0u);
    m_vbo = std::exchange(other.m_vbo, 0u);
    m_ebo = std::exchange(other.m_ebo, 0u);
    m_instanceVbo = std::exchange(other.m_instanceVbo, 0u);
    m_indexCount = std::exchange(other.m_indexCount, std::size_t{0});
    m_instanceCount = std::exchange(other.m_instanceCount, std::size_t{0});
    return *this;
}

Mesh::~Mesh() {
    destroy();
}

void Mesh::destroy() noexcept {
    if (m_instanceVbo != 0) {
        glDeleteBuffers(1, &m_instanceVbo);
    }
    if (m_ebo != 0) {
        glDeleteBuffers(1, &m_ebo);
    }
    if (m_vbo != 0) {
        glDeleteBuffers(1, &m_vbo);
    }
    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
    }
    m_vao = m_vbo = m_ebo = m_instanceVbo = 0;
    m_indexCount = m_instanceCount = 0;
}

void Mesh::setInstanceTransforms(std::span<const Matrix4<float>> transforms) {
    m_instanceCount = transforms.size();
    if (m_instanceVbo == 0) {
        glGenBuffers(1, &m_instanceVbo);
    }

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_instanceVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(transforms.size_bytes()),
                 transforms.data(), GL_DYNAMIC_DRAW);

    // One vec4 attribute per column: a mat4 vertex attribute is four
    // consecutive locations, there is no single "mat4 attribute" in the GL
    // API. Locations 3-6 follow position/normal/uv at 0-2.
    constexpr auto kStride = static_cast<GLsizei>(sizeof(Matrix4<float>));
    for (unsigned column = 0; column < 4; ++column) {
        const unsigned location = 3 + column;
        const auto offset = static_cast<std::uintptr_t>(column) * 4 * sizeof(float);
        glEnableVertexAttribArray(location);
        glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<const void*>(offset));
        glVertexAttribDivisor(location, 1);
    }
    glBindVertexArray(0);
}

void Mesh::draw() const {
    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_indexCount), GL_UNSIGNED_INT,
                   nullptr);
    glBindVertexArray(0);
}

void Mesh::drawInstanced() const {
    glBindVertexArray(m_vao);
    glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(m_indexCount),
                            GL_UNSIGNED_INT, nullptr,
                            static_cast<GLsizei>(m_instanceCount));
    glBindVertexArray(0);
}

}  // namespace ysq
