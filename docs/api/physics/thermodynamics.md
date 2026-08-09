# Physics/Thermodynamics API reference

The ideal gas law, the adiabatic relation, black-body radiation, the
Maxwell-Boltzmann speed distribution, radiative exchange between two
surfaces, and the heat equation (1D and 3D). Start with
[docs/physics/thermodynamics.md](../../physics/thermodynamics.md) for the
ideas; [src/Physics/README.md](../../../src/Physics/README.md) has the
Stefan-Boltzmann derivation and the FTCS stability condition in full.

## `Physics/Thermodynamics/Thermodynamics.hpp`

Gas laws and black-body radiation: no space or time dependence, a single
number in, a single number out.

```cpp
using SpecificGasConstant = Quantity<dim::SpecificGasConstant>;         // R/M: energy/(mass*temperature)
using StefanBoltzmannConstant = Quantity<dim::StefanBoltzmannConstant>; // power/(area*temperature^4)
using WienConstant = Quantity<dim::WienConstant>;                        // length*temperature

namespace constants {
    inline constexpr StefanBoltzmannConstant stefanBoltzmann = /* 2 pi^5 k^4/(15 h^3 c^2), computed, exact */;
    inline constexpr WienConstant wienDisplacementConstant{2.897771955e-3};  // exact, transcribed root
}

constexpr Pressure idealGasPressure(Density density, SpecificGasConstant specificGasConstant,
                                    Temperature temperature) noexcept;

Pressure adiabaticPressure(Pressure p1, Volume v1, Volume v2, double adiabaticIndex);

Power blackBodyLuminosity(Length radius, Temperature temperature);
constexpr Length wienPeakWavelength(Temperature temperature) noexcept;

constexpr Length isothermalScaleHeight(SpecificGasConstant specificGasConstant,
                                       Temperature temperature,
                                       Acceleration surfaceGravity) noexcept;
Density isothermalAtmosphereDensity(Density seaLevelDensity, Length altitude,
                                    Length scaleHeight);
```

| Function | Description |
| --- | --- |
| `idealGasPressure` | `p = rho * R_specific * T`. `specificGasConstant` is `R/M` for the particular gas (about 287 J/(kg K) for dry air); this module works in densities, not moles, so it takes the specific form rather than the universal gas constant. |
| `adiabaticPressure(p1, v1, v2, adiabaticIndex)` | `p V^gamma = const`: pressure after a reversible adiabatic (no heat exchanged) expansion/compression from `v1` to `v2`. `adiabaticIndex` (gamma) is `5/3` for a monatomic ideal gas, `7/5` for diatomic. |
| `blackBodyLuminosity(radius, temperature)` | Stefan-Boltzmann: total power radiated by a sphere of `radius` at uniform `temperature`, treated as an ideal black body. |
| `wienPeakWavelength(temperature)` | Wien's displacement law: `lambda_max = b / T`, the wavelength at which a black body's spectral radiance peaks. |
| `isothermalScaleHeight(specificGasConstant, temperature, surfaceGravity)` | `H = R_specific T / g`: how far an isothermal atmosphere in hydrostatic equilibrium has to rise for its density to fall by a factor of `e`. Follows from `idealGasPressure` held at constant `T` combined with hydrostatic equilibrium. |
| `isothermalAtmosphereDensity(seaLevelDensity, altitude, scaleHeight)` | `rho(h) = rho0 exp(-h / H)`: the exact consequence of `isothermalScaleHeight`'s own derivation. |

`constants::stefanBoltzmann` is **computed** from the SI-defining constants
(`h`, `k`, `c` in `Units/Constants.hpp`) rather than typed independently:
since those three are exact since the 2019 redefinition, sigma is exact
too, not measured. `wienDisplacementConstant` is exact in the same sense but
has to be transcribed rather than computed: it depends on the root of
`x = 5(1 - e^-x)`, which has no closed form.

```cpp
const ysq::Pressure p = ysq::idealGasPressure(density, specificGasConstant, temperature);
const ysq::Power luminosity = ysq::blackBodyLuminosity(starRadius, starTemperature);

const ysq::Length scaleHeight =
    ysq::isothermalScaleHeight(specificGasConstant, temperature, surfaceGravity);
const ysq::Density atThatHeight =
    ysq::isothermalAtmosphereDensity(seaLevelDensity, altitude, scaleHeight);
```

## `Physics/Thermodynamics/StatisticalMechanics.hpp`

The Maxwell-Boltzmann speed distribution: the probability density over
molecular speed for particles of a given mass in thermal equilibrium at a
given temperature. General for any ideal gas, the microscopic counterpart
to `Thermodynamics.hpp`'s bulk gas law.

```cpp
Speed maxwellBoltzmannScale(Mass particleMass, Temperature temperature);

double maxwellBoltzmannSpeedDensity(Speed v, Mass particleMass, Temperature temperature);
double maxwellBoltzmannSpeedCdf(Speed v, Mass particleMass, Temperature temperature);
double maxwellBoltzmannMoment(unsigned n, Mass particleMass, Temperature temperature);

Speed maxwellBoltzmannMostProbableSpeed(Mass particleMass, Temperature temperature);
Speed maxwellBoltzmannMeanSpeed(Mass particleMass, Temperature temperature);
Speed maxwellBoltzmannRmsSpeed(Mass particleMass, Temperature temperature);
```

| Function | Description |
| --- | --- |
| `maxwellBoltzmannScale(particleMass, temperature)` | `a = sqrt(kT/m)`: the distribution's own scale parameter, the unit every closed-form statistic below is a multiple of. |
| `maxwellBoltzmannSpeedDensity(v, particleMass, temperature)` | The probability density `f(v)`, probability per unit speed of speed `[v, v+dv)`. |
| `maxwellBoltzmannSpeedCdf(v, particleMass, temperature)` | `P(speed <= v)`, the closed form for a chi distribution with three degrees of freedom (speed is the magnitude of a 3D normally-distributed velocity). |
| `maxwellBoltzmannMoment(n, particleMass, temperature)` | The `n`-th raw moment `<v^n>`, via the gamma function. Returned as a raw `double` since a single dimension cannot parameterize a run-time exponent. |
| `maxwellBoltzmannMostProbableSpeed` | The density's own peak, `a sqrt(2)`. Distinct from the mean, since the distribution is skewed by the `v^2` factor. |
| `maxwellBoltzmannMeanSpeed` | `<v> = sqrt(8kT/(pi m))`, `maxwellBoltzmannMoment(1, ...)`. |
| `maxwellBoltzmannRmsSpeed` | `sqrt(<v^2>) = sqrt(3kT/m)`, `sqrt(maxwellBoltzmannMoment(2, ...))`; its kinetic energy `(1/2) m v_rms^2 = (3/2) kT` is the equipartition result. |

```cpp
const ysq::Speed mean = ysq::maxwellBoltzmannMeanSpeed(particleMass, temperature);
const double fraction = ysq::maxwellBoltzmannSpeedCdf(escapeSpeed, particleMass, temperature);
```

## `Physics/Thermodynamics/RadiativeTransfer.hpp`

Radiative heat exchange between two finite surfaces: the generalization of
`Thermodynamics.hpp`'s Stefan-Boltzmann law (radiation into free space, a
view factor of 1 and a sink at absolute zero) to exchange between two
bodies each at their own finite temperature, where only some of each
other's radiation reaches the other.

```cpp
constexpr Power netRadiativeExchange(Area area1, double viewFactor, Temperature temperature1,
                                     Temperature temperature2) noexcept;

double coaxialDiskViewFactor(Length radius1, Length radius2, Length distance);
```

| Function | Description |
| --- | --- |
| `netRadiativeExchange(area1, viewFactor, temperature1, temperature2)` | `Q = sigma area1 F_12 (T1^4 - T2^4)`, positive when net power flows from surface 1 to surface 2. Reduces exactly to `blackBodyLuminosity` at `viewFactor = 1`, `temperature2 = 0`. |
| `coaxialDiskViewFactor(radius1, radius2, distance)` | The exact view factor between two coaxial, parallel circular disks. Evaluated in a numerically-stable form (`2x/(S+sqrt(S^2-4x))`) that only ever divides, avoiding the textbook form's cancellation at large separations. |

```cpp
const double viewFactor = ysq::coaxialDiskViewFactor(radius1, radius2, distance);
const ysq::Power netFlow =
    ysq::netRadiativeExchange(area1, viewFactor, temperature1, temperature2);
```

## `Physics/Thermodynamics/HeatEquation.hpp`

The heat (diffusion) equation in one spatial dimension: the one quantity
here with genuine space/time dependence, needing a grid the same way
`Physics/Fluids`' Eulerian solver and `Physics/Electromagnetism`'s Maxwell
solver do.

```
dT/dt = alpha d^2T/dx^2
```

solved by explicit forward-time, centered-space (FTCS) finite differences:

```
T_i^(n+1) = T_i^n + alpha dt/dx^2 (T_(i+1)^n - 2 T_i^n + T_(i-1)^n)
```

```cpp
class HeatEquation1D {
public:
    HeatEquation1D(std::size_t cellCount, double spacing, double diffusivity);

    std::size_t cellCount() const noexcept;
    double spacing() const noexcept;
    double diffusivity() const noexcept;     // alpha = k / (rho * c_p)

    void setTemperature(std::size_t cell, double value);
    double temperature(std::size_t cell) const;

    void step(double dt);                       // must satisfy diffusivity*dt/spacing()^2 <= 0.5
    double stableTimeStep(double safetyFactor) const;  // 0 < safetyFactor <= 1

    double totalHeat() const;    // sum(T) * dx; exactly conserved under periodic boundaries
};
```

FTCS is explicit and only *conditionally* stable (unlike an implicit
scheme such as Crank-Nicolson); the tradeoff buys this scheme's simplicity
and a very direct validation against the exact spreading-Gaussian
solution, at the cost of the stability bound above. Periodic boundaries,
the same choice `Maxwell`/`Eulerian` make. `HeatEquation3D.hpp` below is
the 3D extension.

```cpp
ysq::HeatEquation1D heat(cellCount, spacing, diffusivity);
heat.setTemperature(cell, value);
heat.step(heat.stableTimeStep(/*safetyFactor=*/0.9));
```

## `Physics/Thermodynamics/HeatEquation3D.hpp`

The same heat equation in three spatial dimensions, `dT/dt = alpha
(d^2T/dx^2 + d^2T/dy^2 + d^2T/dz^2)`, the direct generalization of
`HeatEquation1D`'s FTCS scheme to the standard 7-point 3D Laplacian, still
second order.

```cpp
class HeatEquation3D {
public:
    HeatEquation3D(std::size_t cellCountX, std::size_t cellCountY, std::size_t cellCountZ,
                  double spacing, double diffusivity);

    std::size_t cellCountX/cellCountY/cellCountZ() const noexcept;
    double spacing() const noexcept;
    double diffusivity() const noexcept;

    void setTemperature(std::size_t i, std::size_t j, std::size_t k, double value);
    double temperature(std::size_t i, std::size_t j, std::size_t k) const;

    void step(double dt);                               // dt must satisfy the 3D stability condition
    double stableTimeStep(double safetyFactor) const;    // 0 < safetyFactor <= 1

    double totalHeat() const;
};
```

| Member | Description |
| --- | --- |
| `step(dt)` | One explicit FTCS step, `dt` must satisfy `diffusivity * dt / spacing()^2 <= 1/6`, three times tighter than `HeatEquation1D`'s `<= 1/2` since the 3D Laplacian has three axes' worth of neighbor terms instead of one. |
| `stableTimeStep(safetyFactor)` | A safe `dt` at the given safety factor. |
| `totalHeat()` | Exactly conserved under periodic boundaries, the same as `HeatEquation1D`. |

```cpp
ysq::HeatEquation3D heat(nx, ny, nz, spacing, diffusivity);
heat.setTemperature(i, j, k, value);
heat.step(heat.stableTimeStep(/*safetyFactor=*/0.9));
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/thermodynamics)
and let us know.
