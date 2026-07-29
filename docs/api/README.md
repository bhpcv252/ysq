# API reference

Every public class and function `Applications/` code can call, organized by
engine module: signatures, parameters, return values, edge-case behavior,
and a minimal usage snippet for each.

This is the middle of three tiers of documentation:

1. **[docs/](../README.md)**: concept and worked examples. Read this first;
   it explains *why* a piece of the engine exists.
2. **This directory**: pure reference. Look something up while writing
   `Applications/` code; it doesn't explain the reasoning, just the shape.
3. **`src/<Module>/README.md`**: design rationale, invariants, performance
   tradeoffs. Read this when a signature here isn't enough and you need to
   know *why* it's built this way.

## Engine modules

| Page | Covers |
| --- | --- |
| [Core](core.md) | `Version`, `Logger`, `Timer`/`Clock`, `UUID`, `Event`, `Config` |
| [Math: vectors, matrices, scalars](math/algebra.md) | `Vector2/3/4`, `Matrix2/3/4`, `Quaternion`, `Complex`, `Dual`, `Tensor`, `Statistics`, `Interpolation`, `Calculus`, `CoordinateSystems`, `Format`, `Grid` |
| [Math: ODE integrators](math/integrators.md) | The stepper interface, `Euler`/`RK4`/`Adaptive`/`Symplectic` steppers, the fixed-step and adaptive drivers |
| [Units](units.md) | `Quantity`, `Dimension`, every per-quantity header, unit constants, literals, the SI-defining constants |
| [Platform](platform.md) | `Platform`, `Window`, `Input`/`InputState` |
| [Compute](compute.md) | `ComputeBackend`, `CpuBackend`, `OpenGLBackend`, `CudaBackend`/`VulkanBackend` |
| [Physics/Mechanics](physics/mechanics.md) | `Body`, `Frame`, relativistic `Kinematics`, `NBodyState` |
| [Physics/Gravity](physics/gravity.md) | `NewtonianField`, `BarnesHutTree`, the 1PN correction |
| [Physics/Spacetime](physics/spacetime.md) | The metric concept, `christoffelSymbols`, `Minkowski`/`Schwarzschild`/`Kerr`/`FLRW`, `geodesicSystem` |
| [Physics/Electromagnetism](physics/electromagnetism.md) | `electricField`/`magneticField`, `lorentzForce`, `MaxwellField1D` |
| [Physics/Fluids](physics/fluids.md) | `SPHParticle` and the SPH functions, `EulerianFluid1D` |
| [Physics/Thermodynamics](physics/thermodynamics.md) | Ideal gas law, black-body radiation, `HeatEquation1D` |
| [Physics/Optics](physics/optics.md) | `nullTangent`/`propagate`, `deflectionAngle`, `frequencyShift` |
| [Renderer](renderer.md) | `Camera`, `CameraController`s, `Mesh`, `Material`, `Light`, `Texture`, `Shader`, `DebugDraw`, `Renderer`, `RayTracer` |
| [UI](ui.md) | `ImGuiLayer`, `Panel`, `TimeSeriesPlot`/`ScatterPlot`, `StatsOverlay` |

`Applications/` itself has no reference page here: it's a convention
(`Scenario.hpp`/`main.cpp` per app), not an engine API. See
[docs/applications.md](../applications.md) for that convention in full.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/README)
and let us know.
