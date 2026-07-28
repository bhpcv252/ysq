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
| `Renderer/DebugDraw.hpp` | Immediate-mode lines, points, arrows, grid, axes |
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
