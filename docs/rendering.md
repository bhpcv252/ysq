# Rendering

## Conventions

Column-major matrices, column-vector convention, OpenGL clip space (`z` in
`[-1, 1]`, right-handed eye space looking down `-Z`) — the same conventions
`Math/Matrix4.hpp` already documents, since `Renderer` builds directly on its
`lookAt`/`perspective`/`orthographic` factories rather than reimplementing
them.

`float`, not `double`, throughout `Renderer` and `UI`: OpenGL and ImGui are
`float`/`int` APIs, and this is the presentation layer, where that narrowing
is constant and deliberate rather than a physics bug. See
`docs/architecture.md`'s note on why `ysq::warnings_strict` stops at the
engine core.

## No scene graph

`Renderer` is draw calls, not a scene: `beginFrame(camera, ...)`, any number
of `draw()` / `drawInstanced()` / `debugDraw()` calls, then `endFrame()`. A
caller owns its own list of objects and decides what exists and how it is
organized. Nothing this engine currently renders — analytic bodies, vector
fields, grids, geodesics — needs a retained scene graph, so building one now
would be complexity spent on a problem that doesn't exist yet. If a future
`Application` genuinely needs one, it can be built on top of this layer
without Renderer itself changing.

## 2D is not a separate renderer

A screen is 2D; the simulated world is 3D plus time. `Camera`'s orthographic
projection with every object at `z = 0` renders a planar scene — a top-down
view of a binary orbit, say — without a parallel 2D pipeline: `Mesh` and
`DebugDraw` both work unchanged in that plane. Spacetime's own 4th dimension
(time) is handled by re-rendering a 3D snapshot each simulation step, not by
a 4th render axis; see `docs/architecture.md`. Charting 2D *data* (energy
drift, phase space, a Minkowski diagram) is a different concern again and
lives in `UI/PlotPanel.hpp`, backed by ImPlot, composed alongside the 3D
viewport rather than folded into `Renderer`.

## RayTracer: a fragment shader, not a compute shader

`docs/architecture.md` flagged this choice before `Renderer` existed:
written as an OpenGL compute shader, the ray tracer would not run on macOS,
which never gets past OpenGL 4.1 and has no compute shaders (a 4.3 feature).
Written as a fragment shader doing the same ray/scene intersection work, it
runs everywhere the rasterizer already does. `RayTracer` draws a full-screen
quad and does the trace per-pixel in `shaders/raytrace.frag`; light physics
stays in `Physics/Optics` regardless of where the trace executes, this only
decides that.

### Scene upload: structure-of-arrays uniforms, not a UBO

The scene `RayTracer` traces — spheres, planes, disks, lights — has to reach
the shader somehow, and a UBO (uniform buffer object, `std140` layout) is
the obvious first idea. It was rejected: `std140`'s alignment rules for an
array of structs (every element padded to a 16-byte boundary, nested
`vec3`s padded to `vec4`) are easy to get subtly wrong by hand, and getting
them wrong produces silently corrupted scene data rather than a compile or
link error.

Instead, each field of each primitive type is its own plain `uniform`
array — "structure of arrays" rather than "array of structures":
`uSphereCenter[MAX_SPHERES]`, `uSphereRadius[MAX_SPHERES]`,
`uSphereAlbedo[MAX_SPHERES]`, and so on. `glUniform3fv`/`glUniform1fv` have
well-defined layouts with no padding to reason about, and this is exactly as
GL-4.1-portable as a UBO would have been — the actual constraint was never
UBOs themselves, only SSBOs and compute, which are 4.3+. See
`Renderer/RayTracer.cpp`'s `MaterialArrays` helper for the upload, and
`RayTracer.hpp` for the resulting capacities:

| Primitive             | Capacity |
| ---------------------- | -------- |
| Spheres                | 32       |
| Planes                 | 8        |
| Disks                  | 8        |
| Point lights           | 8        |
| Directional lights     | 4        |
| Reflection bounce depth | 8 (shader-side hard cap; `render()`'s `maxBounces` argument is clamped to it) |

A scene beyond these capacities is truncated, not rejected: `render()`
uploads only the first `N` of each list. An `Application` that needs more
either raises the constant (recompiling the shader) or accepts the
truncation; neither has come up yet.

Reflections are iterative, not recursive — GLSL has no function
recursion — accumulating a `throughput` factor across up to `maxBounces`
trace-and-reflect steps and breaking early once a surface isn't reflective.
Shadow rays are hard shadows only: a single visibility test per light per
shaded point, no penumbra. Rays that hit nothing, primary or reflected,
sample `RaytracedScene::environment` (a `Cubemap`) if set, or fall back to
`backgroundColor`.

## Shader embedding

Shaders live as real `.vert`/`.frag` files under `Renderer/shaders/`, not
string literals inside a `.cpp` — unlike `Compute/OpenGL`'s two small
compute kernels, `Renderer` has enough shader source that real files with
proper syntax highlighting are worth it. `cmake/YsqEmbedShader.cmake`
expands each into a generated header (a `constexpr std::string_view`) at
CMake configure time, the same shape `Core/Version.hpp.in` already uses to
generate `Core/Version.hpp`. The compiled binary ends up self-contained,
with no runtime filesystem path to a shader directory to get wrong.

## Instancing and debug drawing

Galaxy collision means thousands of visually identical bodies; one draw call
per body would be the wrong default from day one. `Mesh::setInstanceTransforms`
plus `Renderer::drawInstanced` draw an arbitrary number of copies of one
mesh in a single draw call, via `shaders/instanced.vert` reading a
per-instance model matrix from four consecutive vertex attributes (GL has no
single "mat4 attribute").

`Renderer::debugDraw()` is the other standing need this engine's own physics
creates: orbit trails, force/velocity vectors, field lines, geodesic paths.
It accumulates a batch of lines and points and flushes them once per frame
in one draw call each, rather than drawing them as `Mesh` objects one at a
time.

## Text labels

`DebugDraw::text()`/`textFixed()` are the same idea applied to naming what's
being shown, not just drawing it. The one design choice worth writing down:
text is depth-tested against the rest of the scene but not depth-written.
Depth-testing is what makes a label properly disappear behind an occluding
body rather than always drawing on top like a HUD element; not writing
depth is what keeps two overlapping glyphs (or two overlapping labels) from
z-fighting each other, since a text quad's "background" pixels are fully
transparent rather than absent; the fragment shader still discards them,
but the quad's four corners are still real triangles as far as the depth
buffer is concerned. `GL_BLEND` itself is enabled only for the scoped
duration of that one draw call inside `flush()`, since nothing else in this
renderer blends and leaving it on would be silent, ambient state.

Billboarding — always facing the camera — happens in `shaders/text.vert` by
adding the glyph's local offset along the camera's right/up, extracted from
the `Camera` passed to `beginFrame()` each frame rather than the raw
combined view-projection matrix, which only `Renderer` (not `DebugDraw`)
had cached before this. `textFixed()` reuses the identical shader and
vertex format by resolving that same offset on the CPU against its own
fixed right/up instead, and passing a zero local offset — the billboard
term in the shader becomes a no-op, and the vertex sits exactly where it
was baked, skewing with perspective like any other piece of geometry.
