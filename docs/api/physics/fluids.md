# Physics/Fluids API reference

Two representations of a fluid: SPH (particles carrying the fluid) and an
Eulerian grid (the fluid flowing through a fixed mesh). Start with
[docs/physics/fluids.md](../../physics/fluids.md) for which one fits which
flow; [src/Physics/README.md](../../../src/Physics/README.md) has the kernel
derivation and the Rusanov flux in full.

## `Physics/Fluids/SPH.hpp`

Smoothed Particle Hydrodynamics: the Lagrangian rung, built to reuse the
same shape as `Physics/Gravity`'s direct summation, a span of particles in,
accelerations out, ready for a `Math` integrator.

```cpp
struct SPHParticle {
    double mass = 0.0;
    Vec3 position{};
    Vec3 velocity{};
    double density = 0.0;    // output of computeDensityAndPressure, not independent state
    double pressure = 0.0;   // likewise
};

double cubicSplineKernel(double r, double smoothingLength);
Vec3 cubicSplineKernelGradient(const Vec3& separation, double smoothingLength);

void computeDensityAndPressure(std::span<SPHParticle> particles, double smoothingLength,
                               double equationOfStateK, double polytropicIndex);

std::vector<Vec3> pressureAccelerations(std::span<const SPHParticle> particles,
                                        double smoothingLength);
```

| Function | Description |
| --- | --- |
| `cubicSplineKernel`/`Gradient` | Monaghan & Lattanzio's cubic spline kernel, normalized for 3D, compact support at `2h` (a particle only feels neighbors within that radius). |
| `computeDensityAndPressure` | Updates every particle's `density` (the kernel sum over every other particle, including itself) and `pressure` (polytropic equation of state, `P = equationOfStateK * density^polytropicIndex`) **in place**. Call this before `pressureAccelerations`. |
| `pressureAccelerations` | The symmetric pressure-gradient acceleration, `a_i = -sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2) grad_i W_ij`, which conserves momentum **exactly**, the same pairwise-antisymmetry reasoning as Newtonian gravity, since `W_ij`'s gradient w.r.t. `r_i` is exactly the negative of its gradient w.r.t. `r_j`. Requires densities/pressures already current. |

**Scope: no artificial viscosity**: suited to smooth, low Mach-number flow,
not anything with a shock (`Eulerian.hpp` below is built for that regime).
**No self-gravity**: couple this to `Physics/Gravity`'s own accelerations
if a scenario needs both.

```cpp
std::vector<ysq::SPHParticle> particles = /* positions, masses, velocities */;
ysq::computeDensityAndPressure(particles, smoothingLength, equationOfStateK, polytropicIndex);
const std::vector<ysq::Vec3> accel = ysq::pressureAccelerations(particles, smoothingLength);
```

## `Physics/Fluids/Eulerian.hpp`

The compressible Euler equations (mass, momentum, energy) for an ideal gas,
in one spatial dimension, solved by a first-order finite-volume method with
the Rusanov (local Lax-Friedrichs) numerical flux.

```
d(rho)/dt   + d(rho u)/dx       = 0
d(rho u)/dt + d(rho u^2 + p)/dx = 0
d(E)/dt     + d(u (E + p))/dx   = 0
```

```cpp
class EulerianFluid1D {
public:
    EulerianFluid1D(std::size_t cellCount, double spacing, double adiabaticIndex);

    std::size_t cellCount() const noexcept;
    double spacing() const noexcept;
    double adiabaticIndex() const noexcept;

    void setState(std::size_t cell, double density, double velocity, double pressure);
    double density/velocity/pressure(std::size_t cell) const;

    void step(double dt);                              // dt must satisfy the CFL condition
    double stableTimeStep(double courantNumber) const;  // 0 < courantNumber <= 1

    double totalMass/totalMomentum/totalEnergy() const;
};
```

| Member | Description |
| --- | --- |
| `step(dt)` | One explicit finite-volume step. Use `stableTimeStep` to pick a safe `dt` rather than guessing. |
| `stableTimeStep(courantNumber)` | A CFL-safe step at the given Courant number. |
| `totalMass`/`totalMomentum`/`totalEnergy` | Conservation checks: exact under periodic boundaries, since whatever leaves one edge enters the other. |

**Scope: one spatial dimension, first-order in space and time**: robust
and simple to verify exactly, at the cost of smearing a shock or contact
discontinuity over several cells rather than resolving it sharply (the
standard first-order Godunov-type tradeoff). A higher-order reconstruction
and a sharper Riemann solver (HLLC, or an exact one) aren't implemented.
Periodic boundaries throughout, the same as `MaxwellField1D`: a domain
large enough relative to the run length keeps a wave from wrapping around
and contaminating the result. **A periodic domain split into a left/right
half creates two shock tubes, not one**, since the domain wraps at the
edges too, a real pitfall the module's own test walked into; see
`src/Physics/README.md`.

```cpp
ysq::EulerianFluid1D fluid(cellCount, spacing, adiabaticIndex);
fluid.setState(cell, density, velocity, pressure);
fluid.step(fluid.stableTimeStep(/*courantNumber=*/0.4));
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/fluids)
and let us know.
