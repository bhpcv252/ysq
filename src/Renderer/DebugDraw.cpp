#include <Renderer/DebugDraw.hpp>

#include <Renderer/Font.hpp>
#include <Renderer/shaders/Debug.frag.hpp>
#include <Renderer/shaders/Debug.vert.hpp>
#include <Renderer/shaders/Text.frag.hpp>
#include <Renderer/shaders/Text.vert.hpp>

#include <glad/gl.h>

#include <cmath>
#include <cstddef>
#include <utility>

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

void DebugDraw::setupTextAttributes(unsigned vao, unsigned vbo) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TextVertex),
                          reinterpret_cast<const void*>(offsetof(TextVertex, anchor)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex),
        reinterpret_cast<const void*>(offsetof(TextVertex, localOffset)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex),
                          reinterpret_cast<const void*>(offsetof(TextVertex, uv)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(TextVertex),
                          reinterpret_cast<const void*>(offsetof(TextVertex, color)));
    glBindVertexArray(0);
}

std::optional<DebugDraw> DebugDraw::create(std::string* error) {
    std::optional<Shader> shader =
        Shader::compile(shaders::kDebugVertSource, shaders::kDebugFragSource, error);
    if (!shader) {
        return std::nullopt;
    }
    std::optional<Shader> textShader =
        Shader::compile(shaders::kTextVertSource, shaders::kTextFragSource, error);
    if (!textShader) {
        return std::nullopt;
    }

    // Mipmaps are off, deliberately: mipmapping an alpha-discard atlas
    // averages coverage toward zero at distance, which thins or vanishes
    // glyphs rather than just aliasing a little -- a worse tradeoff than a
    // small amount of shimmer for text that is read up close.
    TextureSettings atlasSettings;
    atlasSettings.filter = TextureFilter::Linear;
    atlasSettings.wrap = TextureWrap::ClampToEdge;
    atlasSettings.generateMipmaps = false;
    std::optional<Texture> fontAtlas =
        Texture::fromPixels(font::buildAtlasPixels(), font::kAtlasWidth,
                            font::kAtlasHeight, TextureFormat::RGBA8, atlasSettings);
    if (!fontAtlas) {
        if (error) {
            *error = "failed to build font atlas texture";
        }
        return std::nullopt;
    }

    unsigned lineVao = 0;
    unsigned lineVbo = 0;
    unsigned pointVao = 0;
    unsigned pointVbo = 0;
    unsigned textVao = 0;
    unsigned textVbo = 0;
    glGenVertexArrays(1, &lineVao);
    glGenBuffers(1, &lineVbo);
    glGenVertexArrays(1, &pointVao);
    glGenBuffers(1, &pointVbo);
    glGenVertexArrays(1, &textVao);
    glGenBuffers(1, &textVbo);
    setupAttributes(lineVao, lineVbo);
    setupAttributes(pointVao, pointVbo);
    setupTextAttributes(textVao, textVbo);

    return std::optional<DebugDraw>{DebugDraw{std::move(*shader), lineVao, lineVbo,
                                              pointVao, pointVbo, std::move(*textShader),
                                              std::move(*fontAtlas), textVao, textVbo}};
}

DebugDraw::DebugDraw(DebugDraw&& other) noexcept
    : m_shader(std::move(other.m_shader)),
      m_lineVao(std::exchange(other.m_lineVao, 0u)),
      m_lineVbo(std::exchange(other.m_lineVbo, 0u)),
      m_pointVao(std::exchange(other.m_pointVao, 0u)),
      m_pointVbo(std::exchange(other.m_pointVbo, 0u)),
      m_lineVertices(std::move(other.m_lineVertices)),
      m_pointVertices(std::move(other.m_pointVertices)),
      m_textShader(std::move(other.m_textShader)),
      m_fontAtlas(std::move(other.m_fontAtlas)),
      m_textVao(std::exchange(other.m_textVao, 0u)),
      m_textVbo(std::exchange(other.m_textVbo, 0u)),
      m_textVertices(std::move(other.m_textVertices)) {}

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
    m_textShader = std::move(other.m_textShader);
    m_fontAtlas = std::move(other.m_fontAtlas);
    m_textVao = std::exchange(other.m_textVao, 0u);
    m_textVbo = std::exchange(other.m_textVbo, 0u);
    m_textVertices = std::move(other.m_textVertices);
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
    if (m_textVbo != 0) {
        glDeleteBuffers(1, &m_textVbo);
    }
    if (m_textVao != 0) {
        glDeleteVertexArrays(1, &m_textVao);
    }
    m_lineVao = m_lineVbo = m_pointVao = m_pointVbo = m_textVao = m_textVbo = 0;
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

namespace {

struct GlyphCorner {
    float x;
    float y;
    float u;
    float v;
};

}  // namespace

template <class ResolveCorner>
void DebugDraw::appendTextVertices(const Vec3f& worldPosition, std::string_view text,
                                   float worldHeight, const Vec3f& color,
                                   const Vec3f& outlineColor, float outlineThickness,
                                   ResolveCorner resolve) {
    std::size_t glyphCount = 0;
    for (char c : text) {
        if (font::glyphUV(c)) {
            ++glyphCount;
        }
    }
    if (glyphCount == 0) {
        return;
    }

    const float totalWidth = static_cast<float>(glyphCount) * worldHeight;
    const float startX = -totalWidth / 2.0f;  // centered horizontally on worldPosition

    // Walks the whole string once, laying every glyph down at `passOffset`
    // (in the same local (x, y) space as the glyph quads themselves) in
    // `passColor`. Called once per outline direction and once, at zero
    // offset, for the real text on top.
    const auto appendPass = [&](const Vec2f& passOffset, const Vec3f& passColor) {
        float penX = startX;
        for (char c : text) {
            const std::optional<font::GlyphUV> uv = font::glyphUV(c);
            if (!uv) {
                continue;
            }

            const float left = penX + passOffset.x;
            const float right = penX + worldHeight + passOffset.x;
            const float bottom = passOffset.y;  // baseline sits at worldPosition
            const float top = worldHeight + passOffset.y;

            // Two triangles, no index buffer, matching every other
            // DebugDraw batch. uvMin.y samples the glyph's top row
            // (Font.hpp's own convention), so it pairs with the corner at
            // y = top.
            const GlyphCorner corners[6] = {
                {left, bottom, uv->uvMin.x, uv->uvMax.y},
                {right, bottom, uv->uvMax.x, uv->uvMax.y},
                {right, top, uv->uvMax.x, uv->uvMin.y},
                {left, bottom, uv->uvMin.x, uv->uvMax.y},
                {right, top, uv->uvMax.x, uv->uvMin.y},
                {left, top, uv->uvMin.x, uv->uvMin.y},
            };

            for (const GlyphCorner& corner : corners) {
                const auto [anchor, localOffset] =
                    resolve(worldPosition, Vec2f{corner.x, corner.y});
                m_textVertices.push_back(TextVertex{
                    anchor, localOffset, Vec2f{corner.u, corner.v}, passColor});
            }

            penX += worldHeight;
        }
    };

    // outlineThickness is a fraction of worldHeight, not a fixed amount, so
    // the outline stays proportionally the same thickness whatever size
    // the text is drawn at, the same reasoning kGlyphWidth == kGlyphHeight
    // keeps glyphs themselves proportional regardless of worldHeight.
    const float outlineWidth = worldHeight * outlineThickness;
    appendPass(Vec2f{-outlineWidth, 0.0f}, outlineColor);
    appendPass(Vec2f{outlineWidth, 0.0f}, outlineColor);
    appendPass(Vec2f{0.0f, -outlineWidth}, outlineColor);
    appendPass(Vec2f{0.0f, outlineWidth}, outlineColor);
    appendPass(Vec2f::zero(), color);
}

void DebugDraw::text(const Vec3f& worldPosition, std::string_view text, float worldHeight,
                     const Vec3f& color, const Vec3f& outlineColor,
                     float outlineThickness) {
    appendTextVertices(worldPosition, text, worldHeight, color, outlineColor,
                       outlineThickness, [](const Vec3f& anchor, const Vec2f& local) {
                           return std::pair<Vec3f, Vec2f>{anchor, local};
                       });
}

void DebugDraw::textFixed(const Vec3f& worldPosition, const Vec3f& right, const Vec3f& up,
                          std::string_view text, float worldHeight, const Vec3f& color,
                          const Vec3f& outlineColor, float outlineThickness) {
    appendTextVertices(worldPosition, text, worldHeight, color, outlineColor,
                       outlineThickness,
                       [&right, &up](const Vec3f& anchor, const Vec2f& local) {
                           return std::pair<Vec3f, Vec2f>{
                               anchor + right * local.x + up * local.y, Vec2f::zero()};
                       });
}

void DebugDraw::flush(const Matrix4<float>& viewProjection, const Vec3f& cameraRight,
                      const Vec3f& cameraUp) {
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

    if (!m_textVertices.empty()) {
        m_textShader.use();
        m_textShader.setUniform("uViewProjection", viewProjection);
        m_textShader.setUniform("uCameraRight", cameraRight);
        m_textShader.setUniform("uCameraUp", cameraUp);
        m_fontAtlas.bind(0);
        m_textShader.setUniform("uFontAtlas", 0);

        // Nothing else in this renderer blends, so this is scoped as
        // tightly as possible: enabled only around this draw call, restored
        // immediately after, rather than left as ambient state some other
        // draw later in the frame would silently inherit. Depth-tested
        // (text behind a body is occluded) but not depth-written (glyphs in
        // the same label, and overlapping labels, don't z-fight each
        // other).
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);

        glBindVertexArray(m_textVao);
        glBindBuffer(GL_ARRAY_BUFFER, m_textVbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(m_textVertices.size() * sizeof(TextVertex)),
                     m_textVertices.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_textVertices.size()));
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    m_lineVertices.clear();
    m_pointVertices.clear();
    m_textVertices.clear();
}

}  // namespace ysq
