# Physics/Acoustics API reference

The linear acoustic wave equation, in 1D and 3D. Start with
[docs/physics/acoustics.md](../../physics/acoustics.md) for why this is
Maxwell's own equations relabeled; [src/Physics/README.md](../../../src/Physics/README.md)
has the field correspondence and the energy formula in full.

## `Physics/Acoustics/Acoustic.hpp`

A 1D FDTD solver for the linearized acoustic wave equations in a
homogeneous medium of density `rho0` and sound speed `c`:

```
dp/dt = -rho0 c^2 du/dx
du/dt = -(1/rho0) dp/dx
```

Structurally identical to `Electromagnetism/Maxwell.hpp`'s 1D vacuum
equations (`p <-> Ey`, `u <-> Bz`, `rho0 c^2 <-> c^2`, `1/rho0 <-> 1`).

```cpp
class AcousticField1D {
public:
    AcousticField1D(std::size_t cellCount, double spacing, double mediumDensity,
                    double soundSpeed);

    std::size_t cellCount() const noexcept;
    double spacing() const noexcept;
    double mediumDensity() const noexcept;
    double soundSpeed() const noexcept;

    double pressure(std::size_t cell) const;
    void setPressure(std::size_t cell, double value);
    double velocity(std::size_t cell) const;    // at x_i + spacing/2 (Yee staggering)
    void setVelocity(std::size_t cell, double value);

    void step(double dt);          // one leapfrog cycle; dt must satisfy dt <= spacing()/soundSpeed()
    double totalEnergy() const;    // (1/2) sum(p^2/(rho0 c^2) + rho0 u^2) * spacing
};

double magicAcousticTimeStep(double spacing, double soundSpeed);  // spacing/soundSpeed
```

| Member | Description |
| --- | --- |
| `step(dt)` | Leapfrog: the same structure as `MaxwellField1D::step`, `p` and `u` in place of `Ey` and `Bz`. |
| `magicAcousticTimeStep(spacing, soundSpeed)` | The step size at which this scheme has zero numerical dispersion in a homogeneous 1D medium, the acoustic analog of `Maxwell.hpp`'s `magicTimeStep`. |
| `totalEnergy()` | Kinetic (`(1/2) rho0 u^2`) plus compressional potential (`(1/2) p^2/(rho0 c^2)`) energy density. `u` is averaged from its two neighbors to approximate its value at `p`'s grid points. |

```cpp
ysq::AcousticField1D field(cellCount, spacing, mediumDensity, soundSpeed);
field.setPressure(sourceCell, initialPulse);

field.step(ysq::magicAcousticTimeStep(spacing, soundSpeed));   // exact propagation, no dispersion
const double energy = field.totalEnergy();                    // conserved, checked over many steps
```

## `Physics/Acoustics/Acoustic3D.hpp`

The same linear acoustics in three spatial dimensions, on a staggered
(marker-and-cell) grid: `p` at cell centers, each velocity component at
its own face centers.

```
dp/dt = -rho0 c^2 (dux/dx + duy/dy + duz/dz)
dux/dt = -(1/rho0) dp/dx    duy/dt = -(1/rho0) dp/dy    duz/dt = -(1/rho0) dp/dz
```

```cpp
class AcousticField3D {
public:
    AcousticField3D(std::size_t cellCountX, std::size_t cellCountY, std::size_t cellCountZ,
                    double spacing, double mediumDensity, double soundSpeed);

    std::size_t cellCountX/cellCountY/cellCountZ() const noexcept;
    double spacing() const noexcept;
    double mediumDensity() const noexcept;
    double soundSpeed() const noexcept;

    double pressure(std::size_t i, std::size_t j, std::size_t k) const;
    void setPressure(std::size_t i, std::size_t j, std::size_t k, double value);
    double velocityX/velocityY/velocityZ(std::size_t i, std::size_t j, std::size_t k) const;
    void setVelocityX/setVelocityY/setVelocityZ(std::size_t i, std::size_t j, std::size_t k,
                                                double value);

    void step(double dt);                              // dt must satisfy the 3D CFL condition
    double stableTimeStep(double courantFactor) const;  // 0 < courantFactor <= 1

    double totalEnergy() const;
};
```

| Member | Description |
| --- | --- |
| `step(dt)` | The same leapfrog structure as `AcousticField1D::step`, generalized to three velocity components. No cross-axis coupling: divergence and gradient are separable per component, unlike `MaxwellField3D`'s curl. |
| `stableTimeStep(courantFactor)` | The 3D CFL limit, `spacing() / (soundSpeed() sqrt(3))`. **No exact "magic timestep" exists in 3D**, the same fact `Maxwell3D.hpp` documents for its own equations. |
| `totalEnergy()` | The 3D analog of `AcousticField1D::totalEnergy`, each velocity component averaged from its two staggered neighbors. |

```cpp
ysq::AcousticField3D field(nx, ny, nz, spacing, mediumDensity, soundSpeed);
field.setPressure(i, j, k, initialPulse);
field.step(field.stableTimeStep(/*courantFactor=*/0.9));
const double energy = field.totalEnergy();
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/acoustics)
and let us know.
