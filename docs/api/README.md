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
| [Core](core.md) | `Version`, `Logger`, `Timer`/`Clock`, `UUID`, `Event`, `Config`, `Csv`, `ExecutablePath` |
| [Math: vectors, matrices, scalars](math/algebra.md) | `Vector2/3/4`, `Matrix2/3/4`, `Quaternion`, `Complex`, `Dual`, `Tensor`, `Statistics`, `Interpolation`, `Calculus`, `CoordinateSystems`, `Format`, `Grid`/`Grid3D`, `FiniteDifference`, `Multigrid` |
| [Math: ODE integrators](math/integrators.md) | The stepper interface, `Euler`/`RK4`/`Adaptive`/`Symplectic` steppers, the fixed-step and adaptive drivers |
| [Math: numerical methods](math/numerics.md) | `RootFinding`, `LinearSolve` (`MatrixN`/`VectorN`), `Eigen` (symmetric/general eigenvalues, SVD, QR), `SpecialFunctions`, `Polynomial`, `Random`, `Optimization`, `FFT` (1D and 3D) |
| [Math: Euclidean geometry and spatial partitioning](math/geometry.md) | `Geometry/Primitives` (incl. `OBB3`), `Intersection`, `Queries`, `ConvexHull`, `SpatialPartition/KdTree`/`Bvh`/`Octree` |
| [Units](units.md) | `Quantity`, `Dimension`, every per-quantity header, unit constants, literals, the SI-defining constants |
| [Platform](platform.md) | `Platform`, `Window`, `Input`/`InputState` |
| [Compute](compute.md) | `ComputeBackend`, `CpuBackend`, `OpenGLBackend`, `CudaBackend`/`VulkanBackend` |
| [Physics/Mechanics](physics/mechanics.md) | `Body`, `Frame`, relativistic `Kinematics`, `NBodyState`, general-purpose forces (`Spring`, `Drag`, `Collision`, `Friction`, `Constraints`), rigid-body inertia diagonalization |
| [Physics/Gravity](physics/gravity.md) | `NewtonianField` (with `SphericalHarmonicsField` oblateness), `BarnesHutTree`, the 1PN correction, `RelativisticNBodySystem`, Kepler orbit elements |
| [Physics/Spacetime](physics/spacetime.md) | The metric concept, `christoffelSymbols`, `Minkowski`/`Schwarzschild`/`Kerr`/`FLRW`, `geodesicSystem` |
| [Physics/Electromagnetism](physics/electromagnetism.md) | `electricField`/`magneticField`, `lorentzForce`, `MaxwellField1D`/`MaxwellField3D` |
| [Physics/Acoustics](physics/acoustics.md) | `AcousticField1D`/`AcousticField3D`: the linear acoustic wave equation |
| [Physics/Fluids](physics/fluids.md) | `SPHParticle` and the SPH functions, `EulerianFluid1D`/`EulerianFluid3D` |
| [Physics/Continuum](physics/continuum.md) | `hookeStress`, `ElasticChain1D`: Hooke's law and a discretized elastic bar |
| [Physics/Thermodynamics](physics/thermodynamics.md) | Ideal gas law, black-body radiation, statistical mechanics, radiative transfer, `HeatEquation1D`/`HeatEquation3D` |
| [Physics/Optics](physics/optics.md) | `nullTangent`/`propagate`, `deflectionAngle`, `frequencyShift`, radiation pressure, diffraction, relativistic aberration |
| [Physics/QuantumMechanics](physics/quantummechanics.md) | `solveTimeIndependentSchrodinger`/`3D`, `TimeDependentWavefunction1D`/`3D` |
| [Renderer](renderer.md) | `Camera`, `CameraController`s, `Mesh`, `Material`, `Light`, `Texture`, `Shader`, `DebugDraw`, `Renderer`, `RayTracer` |
| [UI](ui.md) | `ImGuiLayer`, `Panel`, `TimeSeriesPlot`/`ScatterPlot`, `StatsOverlay` |

`Applications/` itself has no reference page here: it's a convention
(`Scenario.hpp`/`main.cpp` per app), not an engine API. See
[docs/applications.md](../applications.md) for that convention in full.
`Applications/Helper/README.md` is the reference for the scenario-setup
code shared between applications (`Kepler`, `Pole`, `BodyCatalog`) --
real, reusable code, but not engine content either, so it stays
documented alongside the applications that use it rather than gaining a
tier-2 page of its own here.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/README)
and let us know.
