# Renderer API reference

Every public type and function in `Renderer`: camera, meshes, lights,
materials, textures, debug drawing, the frame orchestrator, and the ray
tracer. Start with [docs/renderer.md](../renderer.md) for how the pieces fit
together; [src/Renderer/README.md](../../src/Renderer/README.md) has the
matrix/coordinate conventions and the ray tracer's scene-upload design in
full. Unless noted, a class here follows the same rule as `Shader`: the
GL context it was created under must already be current for every method.

## `Renderer/Camera.hpp`

A view into a scene: eye, target, up, and a projection. Holds no window or
context state.

```cpp
enum class ProjectionMode { Perspective, Orthographic };

struct PerspectiveSettings {
    float fovYRadians = 0.9599311f;   // 55 degrees, full vertical FOV
    float nearPlane = 0.1f, farPlane = 10000.0f;
};

struct OrthographicSettings {
    float height = 10.0f;              // full vertical span; width derives from aspect
    float nearPlane = -10000.0f, farPlane = 10000.0f;
};

struct Camera {
    Vec3f position = {0.0f, 0.0f, 5.0f};
    Vec3f target = Vec3f::zero();
    Vec3f up = Vec3f::unitY();

    ProjectionMode projection = ProjectionMode::Perspective;
    PerspectiveSettings perspectiveSettings{};
    OrthographicSettings orthographicSettings{};

    Vec3f forward() const;
    Vec3f right() const;
    Vec3f trueUp() const;    // cross(right(), forward()): the orthonormal up, unlike the `up` member

    Matrix4<float> viewMatrix() const;
    Matrix4<float> projectionMatrix(float aspect) const;
    Matrix4<float> viewProjectionMatrix(float aspect) const;
};
```

An orthographic projection with everything at `z = 0` is a 2D scene: a
planar orbit view, drawn with the exact same `Mesh`/`DebugDraw` calls as a
full 3D one.

## `Renderer/CameraController.hpp`

Two ready-made controllers, both driven from `Platform::InputState`.

```cpp
struct OrbitCameraController {
    Vec3f target = Vec3f::zero();
    float distance = 10.0f;
    float azimuthRadians = 0.0f;       // around the up axis; zero looks down -Z
    float elevationRadians = 0.3f;     // clamped away from the poles in update()
    float rotateSpeed = 0.005f;        // radians per pixel of cursor delta
    float zoomSpeed = 0.1f;            // fraction of current distance per scroll notch
    float minDistance = 0.01f;

    void update(Camera& camera, const InputState& input) noexcept;
    // call once per frame, after input.newFrame() and Platform::pollEvents()
};

struct FreeFlyCameraController {
    Vec3f position = Vec3f::zero();
    float yawRadians = -1.5707963f;    // zero looks down -Z, matching Camera's default
    float pitchRadians = 0.0f;
    float lookSpeed = 0.0025f;         // radians per pixel of cursor delta
    float moveSpeed = 5.0f;             // units per second
    float fastMultiplier = 4.0f;        // applied while Shift is held

    void update(Camera& camera, const InputState& input, float deltaSeconds) noexcept;
};
```

`OrbitCameraController`: hold right mouse to look, scroll to zoom, the fit
for a scene with an obvious center (a star, a black hole). `FreeFlyCameraController`:
WASD relative to look direction, Q/E straight down/up, right mouse to look,
Shift to move faster, the fit for a scene with no single center (a galaxy
collision).

## `Renderer/Light.hpp`

```cpp
struct PointLight {
    Vec3f position = Vec3f::zero();
    Vec3f color = Vec3f::splat(1.0f);
    float intensity = 1.0f;
    float radius = 0.0f;   // distance at which inverse-square falloff halves the contribution; 0 disables attenuation
};

struct DirectionalLight {
    Vec3f direction = {0.0f, -1.0f, 0.0f};   // points from the light toward the scene
    Vec3f color = Vec3f::splat(1.0f);
    float intensity = 1.0f;                    // does not attenuate
};
```

Shared by `Renderer`'s forward shader and `RayTracer`'s shading, so both
light a scene identically regardless of where the trace runs.

## `Renderer/Material.hpp`

```cpp
struct Material {
    Vec3f albedo = Vec3f::splat(0.8f);
    Vec3f emissive = Vec3f::zero();    // self-lit, independent of any light: what makes a star a light source
    float ambient = 0.1f, diffuse = 0.9f, specular = 0.3f, shininess = 32.0f;
    float reflectivity = 0.0f;          // RayTracer only, [0, 1]; the rasterizer ignores this
};
```

Blinn-Phong, not physically based: nothing this engine renders yet
(analytic bodies, fields, grids) needs a BRDF pipeline.

## `Renderer/Mesh.hpp`

A GPU vertex/index buffer pair (VAO+VBO+EBO), RAII, move-only, geometry
only, no material, no transform.

```cpp
struct Vertex { Vec3f position; Vec3f normal; Vec2f uv; };

class Mesh {
public:
    static std::optional<Mesh> create(std::span<const Vertex> vertices,
                                       std::span<const unsigned> indices);
    static std::optional<Mesh> sphere(float radius = 1.0f, int latitudeSegments = 24,
                                       int longitudeSegments = 48);
    static std::optional<Mesh> quad(float size = 1.0f);       // XY plane, facing +Z
    static std::optional<Mesh> disk(float innerRadius, float outerRadius, int segments = 64);
    static std::optional<Mesh> cube(float size = 1.0f);
    // move-only

    void setInstanceTransforms(std::span<const Matrix4<float>> transforms);
    void draw() const;
    void drawInstanced() const;

    std::size_t indexCount() const noexcept;
    std::size_t instanceCount() const noexcept;
};
```

| Member | Description |
| --- | --- |
| `sphere` | What every body in the engine renders as, whatever its actual scale. |
| `quad` | The full-screen pass `RayTracer` draws into is two triangles of this in clip space; a world-space quad (light gizmo, ground plane) is the same mesh. |
| `setInstanceTransforms` | Uploads a per-instance model-matrix buffer for `drawInstanced()`. An empty span makes `drawInstanced()` draw nothing rather than reading stale data. |

```cpp
ysq::Mesh sphere = *ysq::Mesh::sphere();
sphere.setInstanceTransforms(perBodyTransforms);
renderer.drawInstanced(sphere, material);   // a galaxy's worth of bodies, one draw call
```

## `Renderer/Texture.hpp`

RAII, move-only 2D textures and cubemaps.

```cpp
enum class TextureFormat { RGB8, RGBA8 };
enum class TextureFilter { Nearest, Linear };
enum class TextureWrap { Repeat, ClampToEdge };
struct TextureSettings { TextureFilter filter = TextureFilter::Linear;
                        TextureWrap wrap = TextureWrap::Repeat; bool generateMipmaps = true; };

class Texture {
public:
    static std::optional<Texture> fromPixels(std::span<const std::uint8_t> pixels,
                                              int width, int height, TextureFormat format,
                                              const TextureSettings& settings = {});
    static std::optional<Texture> fromFile(std::string_view path,
                                            const TextureSettings& settings = {},
                                            std::string* error = nullptr);
    // move-only

    void bind(unsigned unit = 0) const;
    int width() const noexcept;
    int height() const noexcept;
    unsigned handle() const noexcept;
};

class Cubemap {
public:
    static std::optional<Cubemap> fromFiles(const std::array<std::string, 6>& facePaths,
                                             std::string* error = nullptr);
    // faces in OpenGL order: +X, -X, +Y, -Y, +Z, -Z
    // move-only
    void bind(unsigned unit = 0) const;
    unsigned handle() const noexcept;
};
```

`fromFile` decodes PNG/JPEG/whatever `stb_image` reads; `nullopt` if the
path can't be read or decoded, with `error` set to `stb_image`'s own
failure reason. `Cubemap` is used for skyboxes and, in `RayTracer`, as the
environment sampled by rays that hit nothing.

## `Renderer/Shader.hpp`

```cpp
class Shader {
public:
    static std::optional<Shader> compile(std::string_view vertexSource,
                                          std::string_view fragmentSource,
                                          std::string* error = nullptr);
    // move-only

    void use() const;
    void setUniform(std::string_view name, int/float/const Vec3f&/const Matrix4<float>& value) const;
    void setUniformArray(std::string_view name, std::span<const float> values) const;
    void setUniformArray(std::string_view name, std::span<const Vec3f> values) const;
};
```

`setUniformArray` is how `RayTracer` uploads its scene (plain uniform
arrays rather than a UBO: no manual std140 padding, and still GL 4.1
portable). Leaves array elements beyond `values.size()` untouched; the
shader is expected to know the true count from a separate `int` uniform.

## `Renderer/DebugDraw.hpp`

Immediate-mode line, point, and text drawing: one place every physics
theory reaches for to show a curve, a vector, or a label.

```cpp
class DebugDraw {
public:
    static std::optional<DebugDraw> create(std::string* error = nullptr);
    // move-only

    void line(const Vec3f& a, const Vec3f& b, const Vec3f& color = Vec3f::splat(1.0f));
    void point(const Vec3f& p, const Vec3f& color = Vec3f::splat(1.0f));
    void arrow(const Vec3f& origin, const Vec3f& direction, const Vec3f& color = Vec3f::splat(1.0f));
    void grid(float extent, int divisions, const Vec3f& color = Vec3f::splat(0.4f));
    void axes(float length = 1.0f);   // unit X (red), Y (green), Z (blue) through the origin

    void text(const Vec3f& worldPosition, std::string_view text, float worldHeight = 0.5f,
             const Vec3f& color = Vec3f::splat(1.0f),
             const Vec3f& outlineColor = Vec3f::zero(), float outlineThickness = 0.08f);
    void textFixed(const Vec3f& worldPosition, const Vec3f& right, const Vec3f& up,
                  std::string_view text, float worldHeight = 0.5f,
                  const Vec3f& color = Vec3f::splat(1.0f),
                  const Vec3f& outlineColor = Vec3f::zero(), float outlineThickness = 0.08f);

    void flush(const Matrix4<float>& viewProjection, const Vec3f& cameraRight, const Vec3f& cameraUp);
};
```

| Member | Description |
| --- | --- |
| `text` | Billboard: always faces the camera. Monospace font, printable ASCII (32-126) only; characters outside that range are skipped. Anchored center-horizontal, baseline-vertical. |
| `textFixed` | `right`/`up` fix the text quad's plane once in world space (used as given, not normalized); it foreshortens with perspective like ordinary scene geometry, for a label baked onto a surface or an orbital plane. |
| `outlineColor`/`outlineThickness` | Four extra offset copies drawn under the main-colored text so it stays legible against any background. Thickness is a fraction of `worldHeight`, so it scales automatically. Pass `outlineColor == color` to disable. |
| `flush` | Uploads the batch, draws with at most three draw calls (lines, points, text) regardless of how many `line`/`point`/`text` calls preceded it, and clears for the next frame. |

```cpp
renderer.debugDraw().axes();
renderer.debugDraw().text(labelPosition, "Jupiter");
renderer.debugDraw().arrow(origin, velocityDirection);
```

## `Renderer/Font.hpp`

The bitmap font `DebugDraw::text`/`textFixed` render with: an 8x8
monospace atlas, ASCII 32-126, baked at `DebugDraw::create()` time. Not
usually called directly by `Applications/` code.

```cpp
namespace font {
    inline constexpr int kGlyphWidth = 8, kGlyphHeight = 8;
    inline constexpr int kFirstChar = 32, kLastChar = 126, kGlyphCount = 95;

    struct GlyphUV { Vec2f uvMin, uvMax; };
    std::optional<GlyphUV> glyphUV(char c);           // nullopt outside [kFirstChar, kLastChar]
    std::vector<std::uint8_t> buildAtlasPixels();      // RGBA8 atlas texture data
}
```

## `Renderer/Renderer.hpp`

The frame orchestrator. Deliberately no scene graph: a caller owns its own
list of objects and calls `draw()` per mesh/material/transform it wants on
screen this frame.

```cpp
class RenderTarget {
public:
    static std::optional<RenderTarget> create(int width, int height, int samples = 0);
    // move-only

    void bind() const;
    static void bindDefault();                    // binds the window's own framebuffer
    std::vector<std::uint8_t> readPixels() const;   // RGBA8, row 0 at the bottom; not valid on a multisampled target
    void resolveTo(const RenderTarget& destination) const;  // the only way to read a multisampled render

    int width() / height() const noexcept;
    unsigned colorTexture() const noexcept;
    bool multisampled() const noexcept;
};

class Renderer {
public:
    static std::optional<Renderer> create(std::string* error = nullptr);

    void beginFrame(const Camera& camera, float aspect, int viewportWidth, int viewportHeight,
                    const Vec3f& clearColor = Vec3f::zero());
    void setLights(std::span<const PointLight> pointLights,
                   std::span<const DirectionalLight> directionalLights);

    void draw(const Mesh& mesh, const Material& material, const Matrix4<float>& model);
    void drawInstanced(const Mesh& mesh, const Material& material);
    void drawSkybox(const Cubemap& sky);   // rotation follows the camera; translation does not

    DebugDraw& debugDraw() noexcept;
    void endFrame();

    std::uint32_t drawCallCount() const noexcept;   // reset by beginFrame(); read by UI::StatsOverlay
};
```

| Member | Description |
| --- | --- |
| `beginFrame` | Sets the viewport, clears color/depth, stores `camera`'s view-projection for every `draw()` until `endFrame()`. Call once per frame, after binding whichever `RenderTarget` (or `RenderTarget::bindDefault()`) is being drawn into. |
| `setLights` | Cleared to none at the start of every `beginFrame()`: a static-lighting scene calls this once per frame the same as a changing one. |
| `endFrame` | Flushes the accumulated `DebugDraw` batch. Call once per frame, after every `draw()`/`debugDraw()` call. |

```cpp
ysq::Renderer renderer = *ysq::Renderer::create();
renderer.beginFrame(camera, aspect, width, height);
renderer.setLights(pointLights, directionalLights);
renderer.draw(sphere, material, ysq::Matrix4<float>::translation(position));
renderer.debugDraw().axes();
renderer.endFrame();
```

`RenderTarget` is what makes a headless render first-class rather than a
test-only hack: an `Application` draws into one exactly as it would the
window's own framebuffer.

## `Renderer/RayTracer.hpp`

A second, independent way to render the same kind of scene: a full-screen
fragment shader tracing rays against a small set of analytic primitives,
with real shadows and reflections. Deliberately a fragment shader, not a
compute shader, to stay portable to OpenGL 4.1 (macOS).

```cpp
struct RaytracedSphere { Vec3f center = Vec3f::zero(); float radius = 1.0f; Material material{}; };
struct RaytracedPlane  { Vec3f point = Vec3f::zero(); Vec3f normal = Vec3f::unitY(); Material material{}; };
struct RaytracedDisk   { Vec3f center = Vec3f::zero(); Vec3f normal = Vec3f::unitY();
                         float innerRadius = 1.0f, outerRadius = 2.0f; Material material{}; };

struct RaytracedScene {
    std::vector<RaytracedSphere> spheres;
    std::vector<RaytracedPlane> planes;
    std::vector<RaytracedDisk> disks;
    std::vector<PointLight> pointLights;
    std::vector<DirectionalLight> directionalLights;
    const Cubemap* environment = nullptr;   // sampled by rays that hit nothing; null falls back to backgroundColor
    Vec3f backgroundColor = Vec3f::zero();
};

class RayTracer {
public:
    static std::optional<RayTracer> create(std::string* error = nullptr);

    void render(const RaytracedScene& scene, const Camera& camera, float aspect,
               int viewportWidth, int viewportHeight, int maxBounces = 4);

    static constexpr std::size_t maxSpheres() noexcept;              // 32
    static constexpr std::size_t maxPlanes() noexcept;                // 8
    static constexpr std::size_t maxDisks() noexcept;                  // 8
    static constexpr std::size_t maxPointLights() noexcept;            // 8
    static constexpr std::size_t maxDirectionalLights() noexcept;      // 4
    static constexpr int maxBounceDepth() noexcept;                     // 8
};
```

`render` **truncates silently** at the `max*` capacities rather than
failing, so a scene that outgrows one frame's uniform arrays degrades
instead of crashing mid-run. `maxBounces` is clamped to `maxBounceDepth()`
by the shader itself. Renders into whatever framebuffer is currently bound,
the same convention as `Renderer`.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/renderer)
and let us know.
