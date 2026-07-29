#pragma once

#include <Math/Matrix4.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>
#include <Renderer/Shader.hpp>
#include <Renderer/Texture.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ysq {

/// Immediate-mode line, point and text drawing: orbit trails, vector fields,
/// geodesics, velocity/force vectors, grids, axes, and labels tied to a
/// world position. Every physics theory in this engine eventually wants to
/// show a curve, a vector, or what to call it, and this is the one place
/// that does it.
///
/// Calls accumulate a vertex batch; flush() uploads it once and issues at
/// most three draw calls (lines, points, text), so drawing ten thousand
/// debug lines a frame is one buffer update, not ten thousand `glDrawArrays`
/// calls. The context this was created under must already be current for
/// every method here, the same rule as Shader.
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

    /// Billboard text: always faces the camera, so it stays readable from
    /// any angle — a label above a planet, an axis name, a readout tied to
    /// a world position. `worldHeight` is one line's rendered height in
    /// world units; the font is monospace, so each glyph is worldHeight
    /// wide too. Anchored center-horizontal, baseline-vertical. Characters
    /// outside printable ASCII (32-126) are skipped; see Font.hpp.
    ///
    /// `outlineColor` draws four extra copies offset by `outlineThickness`
    /// (a fraction of worldHeight, up/down/left/right) underneath the
    /// main-colored one, so text stays legible against a background of any
    /// color rather than only against whatever it happens to be drawn
    /// over. Expressed as a fraction rather than a world-unit width so it
    /// scales with worldHeight automatically, staying proportionally the
    /// same thickness at any distance or font size. Pass a color equal to
    /// `color` to make it disappear.
    void text(const Vec3f& worldPosition, std::string_view text, float worldHeight = 0.5f,
              const Vec3f& color = Vec3f::splat(1.0f),
              const Vec3f& outlineColor = Vec3f::zero(), float outlineThickness = 0.08f);

    /// Fixed-orientation text: `right` and `up` set the text quad's plane
    /// once, in world space, and do not track the camera the way text()
    /// does — the label foreshortens and skews with perspective exactly
    /// like any other piece of scene geometry, which is what a sign fixed
    /// to a surface or a label baked into an orbital plane wants. `right`
    /// and `up` are used as given, not normalized or orthogonalized: pass
    /// unit, perpendicular vectors for undistorted text. `outlineColor` and
    /// `outlineThickness` are the same four-copy outline text() draws.
    void textFixed(const Vec3f& worldPosition, const Vec3f& right, const Vec3f& up,
                   std::string_view text, float worldHeight = 0.5f,
                   const Vec3f& color = Vec3f::splat(1.0f),
                   const Vec3f& outlineColor = Vec3f::zero(),
                   float outlineThickness = 0.08f);

    /// Uploads the accumulated batch, draws it with `viewProjection`, and
    /// clears the batch for the next frame. Point size is fixed in
    /// shaders/debug.vert; there is no per-call size, since GL_PROGRAM_POINT_SIZE
    /// state is what would otherwise leak across an unrelated draw call.
    /// `cameraRight`/`cameraUp` drive text()'s billboard offset; unused by
    /// lines, points or textFixed()'s already-resolved vertices.
    void flush(const Matrix4<float>& viewProjection, const Vec3f& cameraRight,
               const Vec3f& cameraUp);

private:
    struct Vertex {
        Vec3f position;
        Vec3f color;
    };

    /// anchor + localOffset (billboarded by the camera's right/up in
    /// text.vert) is the final world position. textFixed() resolves that
    /// sum itself, on the CPU, against its own fixed right/up, and passes
    /// localOffset = (0, 0) so the shader's billboard term is a no-op.
    struct TextVertex {
        Vec3f anchor;
        Vec2f localOffset;
        Vec2f uv;
        Vec3f color;
    };

    DebugDraw(Shader shader, unsigned lineVao, unsigned lineVbo, unsigned pointVao,
              unsigned pointVbo, Shader textShader, Texture fontAtlas, unsigned textVao,
              unsigned textVbo) noexcept
        : m_shader(std::move(shader)),
          m_lineVao(lineVao),
          m_lineVbo(lineVbo),
          m_pointVao(pointVao),
          m_pointVbo(pointVbo),
          m_textShader(std::move(textShader)),
          m_fontAtlas(std::move(fontAtlas)),
          m_textVao(textVao),
          m_textVbo(textVbo) {}
    void destroy() noexcept;
    static void setupAttributes(unsigned vao, unsigned vbo);
    static void setupTextAttributes(unsigned vao, unsigned vbo);

    /// Shared glyph-layout walk behind text() and textFixed(): advances a
    /// pen across `text`, centering the whole string horizontally and
    /// anchoring its baseline at `worldPosition`, and lays it down five
    /// times -- four small offset copies in `outlineColor`, then `color` at
    /// zero offset -- for the outline effect both public methods document.
    /// `resolve` turns one glyph corner's local (x, y) offset into the
    /// anchor/localOffset pair this pushes as a TextVertex — text() passes
    /// it straight through for the shader to billboard, textFixed() folds
    /// it against a fixed right/up right there instead.
    template <class ResolveCorner>
    void appendTextVertices(const Vec3f& worldPosition, std::string_view text,
                            float worldHeight, const Vec3f& color,
                            const Vec3f& outlineColor, float outlineThickness,
                            ResolveCorner resolve);

    Shader m_shader;
    unsigned m_lineVao = 0;
    unsigned m_lineVbo = 0;
    unsigned m_pointVao = 0;
    unsigned m_pointVbo = 0;
    std::vector<Vertex> m_lineVertices;
    std::vector<Vertex> m_pointVertices;

    Shader m_textShader;
    Texture m_fontAtlas;
    unsigned m_textVao = 0;
    unsigned m_textVbo = 0;
    std::vector<TextVertex> m_textVertices;
};

}  // namespace ysq
