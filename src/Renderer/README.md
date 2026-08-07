# Renderer

Draws a scene: camera, shaders, meshes, textures, a forward rasterizer, a
fragment-shader ray tracer, and immediate-mode debug drawing. Presentation
layer: it depends only on `Math` and `Platform`, never `Physics`. A scene
needs positions, colors and geometry, not what a body's mass or charge
means, and keeping that split is what lets the same `Renderer` draw a solar
system, a galaxy collision or a black hole without knowing anything about
any of them.

**Target:** `ysq::Renderer` (static)
**Depends on:** `ysq::Math`, `ysq::Platform`, both `PUBLIC` since `Camera` and
`CameraController` hand back `Math` types and take a `Platform::InputState`
by reference. `ysq::Core`, `glad` and `stb_image` are implementation details,
linked `PRIVATE`.

Built only under `YSQ_BUILD_GRAPHICS`, same as `Platform`.

## Conventions

Column-major matrices, column-vector convention, OpenGL clip space (`z` in
`[-1, 1]`, right-handed eye space looking down `-Z`), the same conventions
`Math/Matrix4.hpp` already documents, since `Renderer` builds directly on its
`lookAt`/`perspective`/`orthographic` factories rather than reimplementing
them.

**Depth is reversed-Z**: `Camera::projectionMatrix()` calls
`Matrix4::perspective`/`orthographic` with `nearPlane`/`farPlane` swapped —
algebraically identical to a dedicated reversed matrix, so `Matrix4` itself
stays the standard-convention primitive `Math/Matrix4.hpp` documents; only
`Renderer`'s own use of it is reversed. Near maps to NDC/window depth 1.0,
far to 0.0 (the opposite of the textbook mapping), paired with
`glClearDepth(0.0)`, `glDepthFunc(GL_GREATER)` (both set fresh every
`Renderer::beginFrame()`), and a floating-point depth attachment
(`GL_DEPTH32F_STENCIL8` in `RenderTarget::create()`). A standard depth
buffer spends nearly all its precision right next to the near plane; at the
dynamic range this engine's true-to-scale scenarios produce (a camera
standing on a small body's surface while the far plane reaches system-wide
extent — ratios past 1e8 are real, not hypothetical), that leaves almost
none for anything farther out, which reads as z-fighting. Reversed-Z with a
float depth buffer distributes precision evenly across the range instead,
since float's own representable values are naturally denser near 0.0 —
where far geometry now lands — rather than compounding the projection's own
bias toward the near plane. `Renderer::drawSkybox()`'s temporary depth-func
override and `shaders/skybox.vert`'s explicit far-plane pin
(`gl_Position.z = -gl_Position.w`, not the more familiar `.xyww`) both
follow the same convention. `RayTracer` is unaffected — its full-screen
pass disables `GL_DEPTH_TEST` and never reads near/far in its shader.

`float`, not `double`, throughout `Renderer` and `UI`: OpenGL and ImGui are
`float`/`int` APIs, and this is the presentation layer, where that narrowing
is constant and deliberate rather than a physics bug. See the root
`README.md`'s Warnings section for why `ysq::warnings_strict` stops at the
engine core.

## Contents

| Header | Purpose |
| --- | --- |
| `Renderer/Camera.hpp` | Eye/target/up plus perspective or orthographic projection |
| `Renderer/CameraController.hpp` | `OrbitCameraController`, `FreeFlyCameraController`: drive a `Camera` from input |
| `Renderer/SceneCameraController.hpp` | `SceneCameraController`: a drop-in Orbit/FreeFly camera plus an opt-in POV/Focus mode |
| `Renderer/Light.hpp` | `PointLight` (real inverse-square falloff), `DirectionalLight` |
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
grids, geodesics) needs one. If a future `Application` genuinely needs one,
it can be built on top of this layer without `Renderer` itself changing.

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

## Lighting: real inverse-square, and staying visible at any distance

`PointLight` attenuates as `intensity / distance^2` -- real inverse-square
falloff in whatever units a scene's own positions are expressed in, no
separate tunable, because the real law does not have one. (An earlier
version of this struct had a `radius` "half-intensity distance" knob with
an artificial soft-knee formula; it is gone, not deprecated, since a real
point light has nothing to tune besides `intensity` itself.) Both
consumers -- `basic.frag` (the forward rasterizer) and `raytrace.frag`
(`RayTracer`) -- apply the identical formula, `PointLight`'s own doc
comment's stated invariant.

A consequence worth knowing before tuning `intensity`: real falloff means
a light calibrated to look right up close reads as correctly, dramatically
dimmer far away -- Neptune really is a few hundred times dimmer than Earth
under its own Sun, and this renders that faithfully rather than flattening
it the way the old soft-knee formula did. `Applications/KeplerSolarSystem`'s
own `main.cpp` calibrates `intensity` once against Earth's real distance,
the same convention `LunarEclipse/main.cpp` uses for its own Sun light.

**`Renderer::drawGlow(position, worldRadius, color, intensity)`** is the
answer to a real problem that formula alone does not solve: a
self-luminous body's own emissive mesh stops rasterizing as anything once
its true angular size drops under a pixel, real inverse-square falloff or
not -- a rendering problem, not a lighting one. `drawGlow` draws a soft,
additive, camera-facing disc (`shaders/glow.vert`/`glow.frag`) instead,
depth-tested against the rest of the scene but writing no depth of its
own, so a body in front of it still occludes it correctly. It carries no
opinion of its own about falloff or minimum brightness -- `intensity` and
`worldRadius` are entirely the caller's; a caller wanting the glow to stay
locatable at any distance picks a `worldRadius` pinned to a fixed
on-screen pixel size (the same technique `DebugDraw`'s billboard text
already uses) and an `intensity` floored above zero, both scenario-level
choices rather than something this primitive bakes in. See
`Applications/KeplerSolarSystem/main.cpp`'s own Sun-glow call for the
worked example, and `src/Applications/README.md`'s "Closed-form
propagation vs. real N-body" section for why only that application draws
one today.

**Real eclipse/shadow and exposure, no shadow-mapping or HDR pass.**
`Material::lightMultiplier` scales every light's own contribution to one
`Renderer::draw()` call (1.0, a no-op, by default); `Mesh
::setInstanceLightMultipliers(std::span<const float>)` does the same per
instance for a `drawInstanced()` batch -- one asteroid in a belt can read
fully lit while its neighbor, behind a planet, reads dark, in the same
draw call. Both are real physics, not a single knob for one purpose: a
value under 1.0 is `Physics/Optics/Illumination.hpp`'s own
`discOcclusionFraction` (see `src/Physics/README.md`'s own section on it), a
real, closed-form occlusion fraction, without a shadow-mapping pass;
a value above 1.0 is a real, distance-based exposure compensation, the
same reason a real photograph of Saturn needs a far longer exposure than
one of Earth to look properly lit, without an HDR/tonemapping pass. Both
scale only the light-dependent (diffuse and specular) terms in
`basic.frag` -- an eclipsed or under-exposed body still shows its own
ambient light and its own emission, the same way a real one does.
`setInstanceTransforms` itself defaults every instance to 1.0 (fully lit,
uncompensated) whenever it runs, so a caller that never calls
`setInstanceLightMultipliers` at all sees exactly the behavior this had
before the method existed. `KeplerSolarSystem/main.cpp` is the worked
example for both paths and both reasons: a moon's own real shadow from
its planet and a real exposure compensation for its own real distance
(`draw()`), and the same for a ring or belt particle (`drawInstanced()`).

## Scene camera: Orbit, FreeFly, and POV/Focus

`OrbitCameraController` and `FreeFlyCameraController` both use a fixed,
simultaneous button mapping, not a mode to switch between: the right mouse
button looks/orbits (unchanged from before pan existed), the left mouse
button pans — drags the target/position sideways, content following the
cursor the way dragging a canvas does in Figma, without rotating the view.
Scroll still zooms (`Orbit`) or adjusts move speed (`FreeFly`). Neither
controller (nor `SceneCameraController`, which composes them) checks
`UI::ImGuiLayer::wantsMouseCapture()` itself — an `Application` that also
draws `UI` panels in the same window must call
`InputState::suppressMouseThisFrame()` when it does, or a click on a panel
widget also drags the 3D camera underneath it; see
`Applications/SolarSystem/main.cpp` and `LunarEclipse/main.cpp` for the
one-line pattern both use.

`SceneCameraController` (`Renderer/SceneCameraController.hpp`) composes
`OrbitCameraController` and `FreeFlyCameraController` into one drop-in
camera, plus an opt-in POV/Focus mode that stands the camera on one body's
surface and looks at another. It follows the same "no scene graph" rule as
the rest of `Renderer`: it operates on a plain `std::span<const
NamedSphere>` (name, position, radius) a caller rebuilds fresh every frame,
never on `Physics::Body` or any application-specific type, so it is a
generic camera primitive rather than something that knows what "Earth" or
"Moon" is.

Left at its defaults (`povIndex == -1`, "Free"), it behaves exactly like
driving `orbit`/`freeFly` directly — nothing about POV/Focus is active
until a caller sets `povIndex`. The full behavior matrix, keyed by
POV/Focus:

| POV | Focus | Behavior |
| --- | --- | --- |
| Free | Free | Plain `Orbit`/`FreeFly` navigation, unchanged. |
| Free | body | Auto-track: in `Orbit` mode, `orbit.target` snaps to the focus body's position every frame while mouse/scroll still work normally (left-drag pan has no lasting effect here — the same snap overwrites it again next frame — for the same reason manually setting `orbit.target` yourself would not stick either). A no-op in `FreeFly` mode — forcing the look direction would fight WASD/mouse steering. |
| body | other body | Locked: camera anchors to the point on the POV body's surface closest to the focus body and looks at it. Only scroll (an FOV zoom) and `R` (reset the zoom) do anything. |
| body | Free | The camera stands on the POV body's surface. Right-mouse-drag picks which point (the same azimuth/elevation math `OrbitCameraController` uses), scroll walks a height axis from the surface up to a "space view" altitude, and the look target blends from looking straight outward near the surface to the body's center further out, so it reads as an ordinary view of the body once you've pulled back. `R` resets the angle and height. |

The POV body (`X` above) is shown by default in both cases — `hidePov` is a
manual toggle a consumer binds to e.g. a checkbox, not automatic. There is
no terrain on these plain, untextured low-poly spheres, so a consumer that
does hide it isn't losing any detail by doing so.

`povOptions()`/`focusOptions()` return plain `std::vector<std::string>`
option lists ("Free" first, `focusOptions()` excluding whichever body is
currently POV) for building a dropdown in whatever UI toolkit a consumer
uses — `Renderer` and `UI` are peers, so this is data, not a widget.
`indexFromPovSelection()`/`indexFromFocusSelection()` translate a selection
made against those lists back into the index `povIndex`/`focusIndex`
expects.

`statusText()` returns a short, human-readable multi-line summary of the
camera's current state — position plus whichever of mode/speed/POV detail
is relevant right now — for a HUD like `UI::CameraOverlay`. Same reasoning
as the option lists: plain text, not a shared struct type, since neither
`Renderer` nor `UI` may depend on the other.

The up hint both POV submodes write to `camera.up` is derived continuously
from the previous frame's own hint (`continuousUpHint()` in
`SceneCameraController.cpp`), not by switching between two fixed world
axes near a pole — that switch is what a real regression looked like: a
tracked body sweeping close to directly overhead visibly flipped the
camera as the switch's threshold crossed.

See the [Controls](../../docs/renderer.md#controls) section of the
consumer docs for every key binding across `Orbit`, `FreeFly`, and
POV/Focus.

## 2D is not a separate path

An orthographic `Camera` with every object at `z = 0` renders a planar scene
— a top-down orbit view is this, not a different renderer. `Mesh` and
`DebugDraw` both work unchanged in that plane. Spacetime's own 4th dimension
(time) is handled by re-rendering a 3D snapshot each simulation step, not by
a 4th render axis. Charting 2D *data* (energy drift, phase space, a
Minkowski diagram) is a different concern again and lives in
`UI/PlotPanel.hpp`, composed alongside the 3D viewport rather than folded
into `Renderer`.

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
GL-4.1-portable. See
[Scene upload: structure-of-arrays uniforms, not a UBO](#scene-upload-structure-of-arrays-uniforms-not-a-ubo)
below for the capacities and the full design reasoning.

### Scene upload: structure-of-arrays uniforms, not a UBO

The scene `RayTracer` traces (spheres, planes, disks, lights) has to reach
the shader somehow, and a UBO (uniform buffer object, `std140` layout) is
the obvious first idea. It was rejected: `std140`'s alignment rules for an
array of structs (every element padded to a 16-byte boundary, nested
`vec3`s padded to `vec4`) are easy to get subtly wrong by hand, and getting
them wrong produces silently corrupted scene data rather than a compile or
link error.

Instead, each field of each primitive type is its own plain `uniform`
array: "structure of arrays" rather than "array of structures":
`uSphereCenter[MAX_SPHERES]`, `uSphereRadius[MAX_SPHERES]`,
`uSphereAlbedo[MAX_SPHERES]`, and so on. `glUniform3fv`/`glUniform1fv` have
well-defined layouts with no padding to reason about, and this is exactly as
GL-4.1-portable as a UBO would have been: the actual constraint was never
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

Reflections are iterative, not recursive (GLSL has no function
recursion), accumulating a `throughput` factor across up to `maxBounces`
trace-and-reflect steps and breaking early once a surface isn't reflective.
Shadow rays are hard shadows only: a single visibility test per light per
shaded point, no penumbra. Rays that hit nothing, primary or reflected,
sample `RaytracedScene::environment` (a `Cubemap`) if set, or fall back to
`backgroundColor`.

## Shader embedding

Shaders live as real `.vert`/`.frag` files under `shaders/`, not string
literals in a `.cpp`. `cmake/YsqEmbedShader.cmake` expands each into a
generated header (`constexpr std::string_view`) at configure time, the same
shape as `Core/Version.hpp.in`. The compiled binary carries no runtime
filesystem dependency to resolve a shader path.
