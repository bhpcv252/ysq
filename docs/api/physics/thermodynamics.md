# Physics/Thermodynamics API reference

The ideal gas law, the adiabatic relation, black-body radiation, and the
heat equation. Start with
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
```

| Function | Description |
| --- | --- |
| `idealGasPressure` | `p = rho * R_specific * T`. `specificGasConstant` is `R/M` for the particular gas (about 287 J/(kg K) for dry air); this module works in densities, not moles, so it takes the specific form rather than the universal gas constant. |
| `adiabaticPressure(p1, v1, v2, adiabaticIndex)` | `p V^gamma = const`: pressure after a reversible adiabatic (no heat exchanged) expansion/compression from `v1` to `v2`. `adiabaticIndex` (gamma) is `5/3` for a monatomic ideal gas, `7/5` for diatomic. |
| `blackBodyLuminosity(radius, temperature)` | Stefan-Boltzmann: total power radiated by a sphere of `radius` at uniform `temperature`, treated as an ideal black body. |
| `wienPeakWavelength(temperature)` | Wien's displacement law: `lambda_max = b / T`, the wavelength at which a black body's spectral radiance peaks. |

`constants::stefanBoltzmann` is **computed** from the SI-defining constants
(`h`, `k`, `c` in `Units/Constants.hpp`) rather than typed independently:
since those three are exact since the 2019 redefinition, sigma is exact
too, not measured. `wienDisplacementConstant` is exact in the same sense but
has to be transcribed rather than computed: it depends on the root of
`x = 5(1 - e^-x)`, which has no closed form.

```cpp
const ysq::Pressure p = ysq::idealGasPressure(density, specificGasConstant, temperature);
const ysq::Power luminosity = ysq::blackBodyLuminosity(starRadius, starTemperature);
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

**Scope: one spatial dimension.** FTCS is explicit and only *conditionally*
stable (unlike an implicit scheme such as Crank-Nicolson); the tradeoff
buys this scheme's simplicity and a very direct validation against the
exact spreading-Gaussian solution, at the cost of the stability bound above.
Periodic boundaries, the same choice `Maxwell`/`Eulerian` make.

```cpp
ysq::HeatEquation1D heat(cellCount, spacing, diffusivity);
heat.setTemperature(cell, value);
heat.step(heat.stableTimeStep(/*safetyFactor=*/0.9));
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/thermodynamics)
and let us know.
