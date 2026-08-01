# Renderer

Draws whatever your simulation hands it: a camera, meshes, lights, debug
lines, and two ways to render a scene.

## The idea

`Renderer` doesn't know what a planet is, or what a body's mass means; it
only knows about positions, colors, and geometry. That's deliberate: the
physics belongs in `Physics`, and keeping `Renderer` ignorant of it is what
lets the same renderer draw a solar system, a galaxy collision, and a black
hole without any special-casing for any of them.

There's no scene graph here either. A **scene graph** would be a retained
tree of objects the renderer owns and manages; `Renderer` instead is just a
sequence of draw calls you make yourself, once per frame: begin a frame,
draw whatever you want, end the frame. Your `Application` owns the list of
what exists and decides how it's organized, since nothing YSQ currently
renders (bodies, fields, grids, geodesic paths) actually needs a retained
structure.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Renderer/Camera.hpp` | Perspective or orthographic projection, eye/target/up |
| `Renderer/CameraController.hpp` | Orbit and free-fly controls, driven from `Platform` input |
| `Renderer/SceneCameraController.hpp` | A drop-in camera combining Orbit, FreeFly, and an opt-in POV/Focus mode — see [Controls](#controls) |
| `Renderer/Light.hpp` | Point and directional lights |
| `Renderer/Material.hpp` | Blinn-Phong surface parameters |
| `Renderer/Mesh.hpp` | Sphere/quad/disk/cube generators, plus instanced drawing |
| `Renderer/DebugDraw.hpp` | Immediate-mode lines, points, arrows, a grid, axes, and text labels |
| `Renderer/Texture.hpp` | 2D textures and cubemaps |
| `Renderer/Renderer.hpp` | The frame orchestrator: `beginFrame`/`draw`/`endFrame` |
| `Renderer/RayTracer.hpp` | A second way to render the same scene: ray-traced shadows and reflections |
| `Renderer/Shader.hpp` | A compiled GLSL vertex+fragment program; `Renderer` and `RayTracer` build on this, not usually reached for directly |
| `Renderer/Font.hpp` | The bitmap font `DebugDraw`'s text labels render with |

An **orthographic** camera with everything at `z = 0` renders a genuinely
planar scene (a top-down orbit view, say) with the exact same `Mesh` and
`DebugDraw` calls as a full 3D view; there's no separate 2D renderer to
learn.

## Using it

```cpp
#include <Renderer/Renderer.hpp>

ysq::Renderer renderer = *ysq::Renderer::create();
ysq::Mesh sphere = *ysq::Mesh::sphere();

ysq::Camera camera;
renderer.beginFrame(camera, aspect, width, height);
renderer.setLights(pointLights, directionalLights);
renderer.draw(sphere, material, ysq::Matrix4<float>::translation(position));
renderer.debugDraw().axes();
renderer.debugDraw().text(labelPosition, "Jupiter");
renderer.endFrame();
```

A galaxy's worth of visually identical bodies is one draw call, not
thousands: `Mesh::setInstanceTransforms` plus `Renderer::drawInstanced`
draws any number of copies of one mesh at once.

**`RayTracer`** is a second, independent way to render the same kind of
scene: a full-screen fragment shader tracing rays against a small set of
analytic primitives (spheres, planes, disks) with real shadows and
reflections, deliberately written as a fragment shader rather than a
compute shader so it stays portable to OpenGL 4.1, which is as far as macOS
goes. The physics of light itself, if you need more than a rasterizer's
lighting model, lives in [docs/physics/optics.md](physics/optics.md), not
here; `RayTracer` only decides where the trace runs.

## Controls

Every binding below is driven from `Platform::InputState` by
`Renderer/CameraController.hpp` and `Renderer/SceneCameraController.hpp`.
`SceneCameraController` composes the first two navigation modes plus
POV/Focus into one drop-in camera; see
[src/Renderer/README.md](../src/Renderer/README.md#scene-camera-orbit-freefly-and-povfocus)
for the full POV/Focus behavior matrix.

Left and right mouse buttons are both always active, each with a fixed
meaning (pan and look/orbit respectively) — there is no mode to switch
between. An `Application` that also draws `UI` panels in the same window
should call `InputState::suppressMouseThisFrame()` when
`ImGuiLayer::wantsMouseCapture()` is true, or a click on a panel widget
also drags the 3D camera underneath it — see [docs/api/ui.md](api/ui.md).

### Orbit (`OrbitCameraController`)

Rotates and zooms around a fixed target. The default mode, and the natural
fit for a scene with an obvious center.

| Input | Effect |
| --- | --- |
| Right mouse, drag | Orbit (azimuth/elevation) around `target` |
| Left mouse, drag | Pan: drag `target` sideways, content follows the cursor (à la Figma) |
| Scroll | Zoom (`distance`), clamped to `minDistance` |

### FreeFly (`FreeFlyCameraController`)

First-person navigation for a scene with no single center, or for crossing
the enormous scale range between an AU-wide orbit and a planet's own
surface.

| Input | Effect |
| --- | --- |
| W / A / S / D | Move forward/left/back/right, relative to look direction |
| Q / E | Move straight down/up |
| Right mouse, drag (or T to toggle look on) | Look around (yaw/pitch) |
| Left mouse, drag | Pan: slide sideways without rotating, content follows the cursor |
| T | Toggle look-locked: look responds to the mouse without holding the right button |
| Shift | Move faster (`fastMultiplier`) |
| Scroll | Change move speed (`moveSpeed`), so exploring stays a sane speed at any scale |
| Z / C | Roll left/right (rotates the render-facing up hint only; WASD movement stays relative to true world-up) |

Movement eases toward the input-driven target velocity
(`accelerationPerSecond`) rather than snapping instantly, so starting and
stopping aren't a hard cut. `invertY` flips the pitch direction for mouse
look.

### POV/Focus (`SceneCameraController`)

Off by default (`povIndex == -1`, "Free"). Once a POV body is picked:

| Input | Effect |
| --- | --- |
| Right mouse, drag | (POV set, Focus Free only) picks which point on the POV body's surface you occupy |
| Scroll | (POV set, Focus Free) walks a height axis from the surface to a "space view" altitude; (POV and Focus both set) FOV zoom instead |
| R | Reset the current POV submode's live-adjusted state (zoom back to 1x, or angle/height back to their defaults) |

## Go deeper

[docs/api/renderer.md](api/renderer.md) has every signature: `Camera`,
`CameraController`s, `Mesh`, `Material`, `Light`, `Texture`, `Shader`,
`DebugDraw`, `Renderer`, and `RayTracer`.

[src/Renderer/README.md](../src/Renderer/README.md) has the full interface,
the matrix and coordinate conventions (column-major, OpenGL clip space),
text label billboarding, multisampling, and the ray tracer's scene-upload
design and primitive capacities.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+renderer)
and let us know.
