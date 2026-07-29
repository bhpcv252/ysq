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
underneath). A full 3D Yee-grid solver, needed for a genuinely radiating
source like a dipole, is not implemented; this rung validates against what
1D vacuum electrodynamics predicts: a wave traveling at exactly `c`, and a
closed system's energy staying constant.

```cpp
ysq::MaxwellField1D field(cellCount, spacing);
field.setElectricField(sourceCell, initialPulse);

field.step(ysq::magicTimeStep(spacing));   // exact propagation, no dispersion
const double energy = field.totalEnergy(); // conserved, checked over many steps
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/electromagnetism)
and let us know.
