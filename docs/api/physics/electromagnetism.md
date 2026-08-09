# Physics/Electromagnetism API reference

Coulomb and Biot-Savart superposition, the Lorentz force, and a Maxwell FDTD
solver that actually propagates. Start with
[docs/physics/electromagnetism.md](../../physics/electromagnetism.md) for
the two-rung ladder; [src/Physics/README.md](../../../src/Physics/README.md)
has the field formulas and Yee-staggering details in full.

## `Physics/Electromagnetism/Field.hpp`

Quasi-static fields, built from point charges by direct superposition: each
source's *present* position and velocity, no propagation delay. The next
rung (`Maxwell.hpp`, below) is where a field actually propagates.

```cpp
using VacuumPermeability = Quantity<dim::VacuumPermeability>;
using VacuumPermittivity = Quantity<dim::VacuumPermittivity>;
using CoulombConstant = Quantity<dim::CoulombConstant>;

namespace constants {
    inline constexpr VacuumPermeability vacuumPermeability{1.25663706212e-6};   // measured, CODATA 2018/2022
    inline constexpr VacuumPermittivity vacuumPermittivity = /* 1/(mu0 c^2), computed */;
    inline constexpr CoulombConstant coulombConstant = /* 1/(4 pi epsilon0), computed */;
}

ElectricField3 electricField(const Length3& at, std::span<const Body> sources);
MagneticFluxDensity3 magneticField(const Length3& at, std::span<const Body> sources);
```

| Function | Description |
| --- | --- |
| `electricField(at, sources)` | Coulomb's law, superposed over every source. **Undefined and skipped** at zero separation from a source (that term is simply omitted, not a division by zero propagating through). |
| `magneticField(at, sources)` | The point-charge form of Biot-Savart, superposed over every moving source. |

`constants::vacuumPermeability` lives here rather than in
`Units/Constants.hpp` for the same reason `Gravity`'s `G` does: it's
measured (not one of the SI's seven definitional constants since the 2019
redefinition tied the ampere to the elementary charge instead of fixing `mu0`
outright) and it parameterizes one specific interaction.
`vacuumPermittivity` and `coulombConstant` are **computed** from
`vacuumPermeability` and the exact speed of light rather than typed
independently, so a retyped digit can't let the three drift apart.

```cpp
const ysq::ElectricField3 e = ysq::electricField(queryPoint, sources);
const ysq::MagneticFluxDensity3 b = ysq::magneticField(queryPoint, sources);
```

## `Physics/Electromagnetism/Lorentz.hpp`

```cpp
Force3 lorentzForce(const Body& body, const ElectricField3& electric,
                    const MagneticFluxDensity3& magnetic);
// F = q (E + v x B)
```

```cpp
const ysq::Force3 force = ysq::lorentzForce(chargedBody, e, b);
```

## `Physics/Electromagnetism/Maxwell.hpp`

A one-dimensional FDTD (finite-difference time-domain) solver for the vacuum
Maxwell equations, restricted to a transverse wave along `x` with components
`Ey`, `Bz`:

```
dEy/dt = -c^2 dBz/dx
dBz/dt = -dEy/dx
```

```cpp
class MaxwellField1D {
public:
    MaxwellField1D(std::size_t cellCount, double spacing);

    std::size_t cellCount() const noexcept;
    double spacing() const noexcept;

    double electricField(std::size_t cell) const;
    void setElectricField(std::size_t cell, double value);
    double magneticField(std::size_t cell) const;    // at x_i + spacing/2 (Yee staggering)
    void setMagneticField(std::size_t cell, double value);

    void step(double dt);          // one leapfrog cycle; dt must satisfy dt <= spacing()/c
    double totalEnergy() const;    // (1/2) sum(epsilon0 Ey^2 + Bz^2/mu0) * spacing
};

double magicTimeStep(double spacing);   // spacing / c: zero numerical dispersion in 1D vacuum
```

| Member | Description |
| --- | --- |
| `step(dt)` | Leapfrog: advances `Bz` by half a step, then `Ey` by a full step using the updated `Bz`, the same structure as `Math`'s symplectic integrators, applied to a field instead of a particle. Second-order accurate and exactly energy-conserving in that sense. |
| `magicTimeStep(spacing)` | The step size at which this scheme has **zero** numerical dispersion in 1D vacuum: a wave keeps its exact shape as it propagates, not just approximately. |
| `totalEnergy()` | For a conservation check. `Bz` is averaged from its two neighbors to approximate its value at `Ey`'s grid points, since the two live half a cell apart. |

**Scope: one spatial dimension**, periodic boundaries (via `Math::Grid1D`
underneath); this rung validates against what 1D vacuum electrodynamics
predicts: a wave traveling at exactly `c`, and a closed system's energy
staying constant. The full 3D Yee-grid solver, needed for a genuinely
radiating source like a dipole, is `Maxwell3D.hpp` below.

```cpp
ysq::MaxwellField1D field(cellCount, spacing);
field.setElectricField(sourceCell, initialPulse);

field.step(ysq::magicTimeStep(spacing));   // exact propagation, no dispersion
const double energy = field.totalEnergy(); // conserved, checked over many steps
```

## `Physics/Electromagnetism/Maxwell3D.hpp`

The full 3D Yee grid, vacuum Maxwell curl equations:

```
dEx/dt = c^2 (dBz/dy - dBy/dz)     dBx/dt = -(dEz/dy - dEy/dz)
dEy/dt = c^2 (dBx/dz - dBz/dx)     dBy/dt = -(dEx/dz - dEz/dx)
dEz/dt = c^2 (dBy/dx - dBx/dy)     dBz/dt = -(dEy/dx - dEx/dy)
```

```cpp
class MaxwellField3D {
public:
    MaxwellField3D(std::size_t cellCountX, std::size_t cellCountY, std::size_t cellCountZ,
                  double spacing);

    std::size_t cellCountX/cellCountY/cellCountZ() const noexcept;
    double spacing() const noexcept;

    double electricFieldX/Y/Z(std::size_t i, std::size_t j, std::size_t k) const;
    void setElectricFieldX/Y/Z(std::size_t i, std::size_t j, std::size_t k, double value);
    double magneticFieldX/Y/Z(std::size_t i, std::size_t j, std::size_t k) const;
    void setMagneticFieldX/Y/Z(std::size_t i, std::size_t j, std::size_t k, double value);

    void step(double dt);                              // dt must satisfy the 3D CFL condition
    double stableTimeStep(double courantFactor) const;  // 0 < courantFactor <= 1

    double totalEnergy() const;
};
```

| Member | Description |
| --- | --- |
| Six field components | Each keeps its own Yee staggering (`Ex` at `(i+1/2,j,k)`, `Bx` at `(i,j+1/2,k+1/2)`, and so on), chosen so every curl term a component needs is a plain adjacent-index difference of another component already staggered to the right position. |
| `step(dt)` | Same leapfrog structure as `MaxwellField1D`: a `B` half-step reading a forward difference of `E`, then an `E` full step reading a backward difference of the just-updated `B`. |
| `stableTimeStep(courantFactor)` | The 3D CFL limit, `spacing() / (c sqrt(3))`. **No exact "magic timestep" exists in 3D** — the 1D scheme's zero-dispersion step is a coincidence of that specific 1D discretization; a Cartesian Yee grid's dispersion is direction-dependent once there's more than one dimension. |
| `totalEnergy()` | Each component bilinearly interpolated onto the grid's integer vertices before combining (`E`'s two neighbors along its own stagger axis; `B`'s four, since each `B` component is staggered along two axes at once). Bounded over many steps off the (nonexistent) magic step, not exactly conserved. |

```cpp
ysq::MaxwellField3D field(nx, ny, nz, spacing);
field.setElectricFieldY(i, j, k, initialValue);
field.step(field.stableTimeStep(/*courantFactor=*/0.9));
const double energy = field.totalEnergy();
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/electromagnetism)
and let us know.
