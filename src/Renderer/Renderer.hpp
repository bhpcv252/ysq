#pragma once

#include <Renderer/Camera.hpp>
#include <Renderer/DebugDraw.hpp>
#include <Renderer/Light.hpp>
#include <Renderer/Material.hpp>
#include <Renderer/Mesh.hpp>
#include <Renderer/Shader.hpp>
#include <Renderer/Texture.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ysq {

/// An offscreen color+depth render target (a framebuffer object). The
/// context that created it must already be current for every method here,
/// the same rule as Shader. This is what makes a headless render a first-class
/// path rather than a test-only hack: an Application can draw into one of
/// these exactly as it would the window's own framebuffer.
class RenderTarget {
public:
    /// `samples` requests multisampling; zero disables it.
    [[nodiscard]] static std::optional<RenderTarget> create(int width, int height,
                                                            int samples = 0);

    RenderTarget(const RenderTarget&) = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;
    RenderTarget(RenderTarget&& other) noexcept;
    RenderTarget& operator=(RenderTarget&& other) noexcept;
    ~RenderTarget();

    void bind() const;
    /// Binds the window's own framebuffer (object 0), undoing bind().
    static void bindDefault();

    /// Reads the color attachment back to the CPU as tightly-packed RGBA8,
    /// row 0 at the bottom, OpenGL's own convention. Meaningless called on a
    /// multisampled target directly (OpenGL rejects glReadPixels against one
    /// with GL_INVALID_OPERATION); resolveTo() a non-multisampled target
    /// first.
    [[nodiscard]] std::vector<std::uint8_t> readPixels() const;

    /// Blits this target's color attachment into `destination`, which must
    /// be the same size and not itself multisampled. The only way to turn a
    /// multisampled render into pixels readPixels() can return.
    void resolveTo(const RenderTarget& destination) const;

    [[nodiscard]] int width() const noexcept { return m_width; }
    [[nodiscard]] int height() const noexcept { return m_height; }
    [[nodiscard]] unsigned colorTexture() const noexcept { return m_colorTexture; }
    [[nodiscard]] bool multisampled() const noexcept { return m_multisampled; }

private:
    RenderTarget(unsigned fbo, unsigned colorTexture, unsigned depthRenderbuffer,
                 int width, int height, bool multisampled) noexcept
        : m_fbo(fbo),
          m_colorTexture(colorTexture),
          m_depthRenderbuffer(depthRenderbuffer),
          m_width(width),
          m_height(height),
          m_multisampled(multisampled) {}
    void destroy() noexcept;

    unsigned m_fbo = 0;
    unsigned m_colorTexture = 0;
    unsigned m_depthRenderbuffer = 0;
    int m_width = 0;
    int m_height = 0;
    bool m_multisampled = false;
};

/// Draw-call-level forward renderer. Deliberately no scene graph: a caller
/// (an Application, or a test) owns its own list of objects and calls draw()
/// once per Mesh/Material/transform it wants on screen this frame, plus
/// drawInstanced() for many identical ones and debugDraw() for lines and
/// points. Nothing here decides what exists in a scene or how it is
/// organized; see src/Renderer/README.md for why.
class Renderer {
public:
    [[nodiscard]] static std::optional<Renderer> create(std::string* error = nullptr);

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) noexcept = default;
    Renderer& operator=(Renderer&&) noexcept = default;
    ~Renderer() = default;

    /// Sets the viewport, clears color and depth, and stores `camera`'s
    /// view-projection for every draw() until the matching endFrame(). Call
    /// once per frame before any draw() calls, after binding whichever
    /// RenderTarget (or RenderTarget::bindDefault()) is being drawn into.
    void beginFrame(const Camera& camera, float aspect, int viewportWidth,
                    int viewportHeight, const Vec3f& clearColor = Vec3f::zero());

    /// Replaces the lights every draw() until the next call sees. Cleared to
    /// none at the start of every beginFrame(), so a caller with a static
    /// scene calls this once per frame the same as one with a changing one.
    void setLights(std::span<const PointLight> pointLights,
                   std::span<const DirectionalLight> directionalLights);

    void draw(const Mesh& mesh, const Material& material, const Matrix4<float>& model);

    /// Draws `mesh` once per transform already uploaded via
    /// Mesh::setInstanceTransforms.
    void drawInstanced(const Mesh& mesh, const Material& material);

    /// `sky`'s rotation follows the camera; its translation does not, so it
    /// never appears to move as the camera does.
    void drawSkybox(const Cubemap& sky);

    /// A soft, additive-blended, camera-facing glow disc of world-space
    /// diameter `2 * worldRadius` centered at `position`: for a
    /// self-luminous body a plain lit/emissive mesh alone does not read as
    /// what it is once its real angular size drops below a pixel (a star
    /// seen from far enough away), the same reason a rasterized point
    /// source needs *some* on-screen representation general to any distance
    /// scale, not physics-specific to any one scenario. `intensity` is
    /// entirely the caller's own -- this does not know or assume any
    /// falloff law, real or otherwise; a caller wanting it to fade
    /// correctly with distance computes `intensity` from whatever law it is
    /// already using for its lights (real inverse-square, say) and passes
    /// the result straight through. Depth-tested against the rest of the
    /// scene (so a body in front of it still occludes it) but does not
    /// write depth itself, so it never occludes anything drawn after it.
    void drawGlow(const Vec3f& position, float worldRadius, const Vec3f& color,
                 float intensity);

    /// Accumulates into the batch flushed at endFrame(). Call draw() calls
    /// and debugDraw() calls in any order within a frame.
    [[nodiscard]] DebugDraw& debugDraw() noexcept { return m_debugDraw; }

    /// Flushes the accumulated DebugDraw batch. Call once per frame, after
    /// every draw() and debugDraw() call.
    void endFrame();

    /// How many draw()/drawInstanced()/drawSkybox() calls this frame has
    /// issued so far. Reset by beginFrame(); UI::StatsOverlay is the
    /// intended reader.
    [[nodiscard]] std::uint32_t drawCallCount() const noexcept { return m_drawCallCount; }

private:
    Renderer(Shader basicShader, Shader instancedShader, Shader skyboxShader,
             Shader glowShader, DebugDraw debugDraw, Mesh skyboxCube, Mesh glowQuad) noexcept
        : m_basicShader(std::move(basicShader)),
          m_instancedShader(std::move(instancedShader)),
          m_skyboxShader(std::move(skyboxShader)),
          m_glowShader(std::move(glowShader)),
          m_debugDraw(std::move(debugDraw)),
          m_skyboxCube(std::move(skyboxCube)),
          m_glowQuad(std::move(glowQuad)) {}

    void applyLights(const Shader& shader) const;

    Shader m_basicShader;
    Shader m_instancedShader;
    Shader m_skyboxShader;
    Shader m_glowShader;
    DebugDraw m_debugDraw;
    /// Drawn from the inside for drawSkybox(): skybox.vert reads aPosition
    /// directly as a sample direction, so any cube works regardless of size.
    Mesh m_skyboxCube;
    /// drawGlow()'s own unit quad, billboarded and scaled per call; see
    /// shaders/glow.vert.
    Mesh m_glowQuad;

    std::vector<PointLight> m_pointLights;
    std::vector<DirectionalLight> m_directionalLights;

    Matrix4<float> m_view = Matrix4<float>::identity();
    Matrix4<float> m_projection = Matrix4<float>::identity();
    Vec3f m_cameraPosition = Vec3f::zero();
    /// Cached from the Camera passed to beginFrame(), for DebugDraw's
    /// billboard text: flush() needs them but a Camera is not otherwise
    /// part of Renderer's per-frame state.
    Vec3f m_cameraRight = Vec3f::unitX();
    Vec3f m_cameraUp = Vec3f::unitY();
    std::uint32_t m_drawCallCount = 0;
};

}  // namespace ysq
