# Renderer

Draws a scene: camera, shaders, meshes, textures, a forward rasterizer, a
fragment-shader ray tracer, and immediate-mode debug drawing. Presentation
layer; see `docs/architecture.md` for why it depends only on `Math` and
`Platform`, never `Physics`.

**Target:** `ysq::Renderer` (static)
**Depends on:** `ysq::Math`, `ysq::Platform`, both `PUBLIC` since `Camera` and
`CameraController` hand back `Math` types and take a `Platform::InputState`
by reference. `ysq::Core`, `glad` and `stb_image` are implementation details,
linked `PRIVATE`.

Built only under `YSQ_BUILD_GRAPHICS`, same as `Platform`.

## Contents

| Header | Purpose |
| --- | --- |
| `Renderer/Camera.hpp` | Eye/target/up plus perspective or orthographic projection |
| `Renderer/CameraController.hpp` | `OrbitCameraController`, `FreeFlyCameraController`: drive a `Camera` from input |
| `Renderer/Light.hpp` | `PointLight`, `DirectionalLight` |
| `Renderer/Material.hpp` | Blinn-Phong surface parameters, shared by the rasterizer and the ray tracer |
| `Renderer/Shader.hpp` | Vertex+fragment GLSL program, RAII |
| `Renderer/Mesh.hpp` | Vertex/index buffer, RAII; sphere/quad/disk/cube generators; instanced draw |
| `Renderer/DebugDraw.hpp` | Immediate-mode lines, points, arrows, grid, axes, and billboard/fixed text labels |
| `Renderer/Font.hpp` | The bitmap font `DebugDraw`'s text labels rasterize with |
| `Renderer/Texture.hpp` | 2D texture and `Cubemap`, from raw pixels or a decoded file |
| `Renderer/Renderer.hpp` | `RenderTarget` (offscreen FBO) and `Renderer`, the draw-call-level frame orchestrator |
| `Renderer/RayTracer.hpp` | Full-screen fragment-shader ray tracer: shadows, reflections |

## No scene graph

`Renderer` is deliberately just draw calls: `beginFrame(camera, ...)`, then
any number of `draw()` / `drawInstanced()` / `debugDraw()` calls, then
`endFrame()`. A caller — an `Application`, or a test — owns its own list of
objects and decides what exists and how it's organized. Nothing here is a
scene graph, because nothing this engine renders (analytic bodies, fields,
grids, geodesics) needs one; see `docs/rendering.md`.

```cpp
ysq::Renderer renderer = *ysq::Renderer::create();
ysq::Mesh sphere = *ysq::Mesh::sphere();

ysq::Camera camera;
renderer.beginFrame(camera, aspect, width, height);
renderer.setLights(pointLights, directionalLights);
renderer.draw(sphere, material, ysq::Matrix4<float>::translation(position));
renderer.debugDraw().axes();
renderer.endFrame();
```

## 2D is not a separate path

An orthographic `Camera` with every object at `z = 0` renders a planar scene
— a top-down orbit view is this, not a different renderer. `Mesh` and
`DebugDraw` both work unchanged in that plane.

## Instancing and debug drawing

`Mesh::setInstanceTransforms` plus `Renderer::drawInstanced` draw many
identical meshes (a galaxy's worth of stars) in one draw call rather than one
per body. `Renderer::debugDraw()` accumulates lines and points — orbit
trails, force/velocity vectors, field lines, geodesics — into one batch,
flushed once at `endFrame()`.

## Text labels

`DebugDraw::text()` and `textFixed()` draw short strings tied to a world
position — a planet's name, an axis label, a readout — using a classic 8x8
monospace bitmap font (`Font.hpp`; ASCII 32-126, the widely-redistributed
public-domain IBM PC/VGA ROM charset), baked into a small texture atlas at
`DebugDraw::create()` time. No font file and no TrueType parsing: this
engine's text is short debug/label strings, and a fixed bitmap this small is
worth owning outright rather than taking a dependency for.

```cpp
renderer.debugDraw().text(labelPosition, "Jupiter");
renderer.debugDraw().textFixed(signPosition, right, up, "N");
```

`text()` billboards: the vertex shader adds the glyph quad's local offset
along the *camera's* right/up every frame, so a label always faces the
viewer regardless of orbit. `textFixed()` resolves that same offset once, on
the CPU, against a caller-supplied fixed right/up instead — the quad's
orientation is baked in and does not track the camera, so it foreshortens
and skews with perspective exactly like any other piece of scene geometry.
Both share one vertex format and one shader; `textFixed()` simply passes a
zero local offset so the shader's billboard term is a no-op.

Text is depth-tested against the rest of the scene, so a label behind a body
is properly occluded, but not depth-written, so glyphs within one label (or
overlapping labels) don't z-fight each other. `GL_BLEND` is enabled only for
the scoped duration of the text draw call inside `flush()` — nothing else in
this renderer blends, so leaving it enabled any longer would be ambient
state a later, unrelated draw call would silently inherit.

## Multisampling

`RenderTarget::create(width, height, samples)` requests MSAA; `readPixels()`
against a multisampled target directly is meaningless (OpenGL rejects it),
so `resolveTo()` blits it into an ordinary `RenderTarget` first. Not every
machine supports multisampled framebuffers — a software rasterizer in
particular may not — so treat a failed multisampled `create()` as absent
capability, not an error.

## RayTracer

A full-screen-quad fragment shader, not a compute shader: compute shaders
are a 4.3 feature, and this stays portable to OpenGL 4.1, where macOS caps
out. It traces a small, capped set of analytic primitives (spheres, planes,
disks) with hard shadows and reflections up to a configurable bounce depth.
Scene data uploads as plain "structure of arrays" uniform arrays rather than
a UBO — no manual `std140` padding to get right by hand, and exactly as
GL-4.1-portable. See `docs/rendering.md` for the capacities and the full
design reasoning.

## Shader embedding

Shaders live as real `.vert`/`.frag` files under `shaders/`, not string
literals in a `.cpp`. `cmake/YsqEmbedShader.cmake` expands each into a
generated header (`constexpr std::string_view`) at configure time, the same
shape as `Core/Version.hpp.in`. The compiled binary carries no runtime
filesystem dependency to resolve a shader path.
