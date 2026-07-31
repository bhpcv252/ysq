# Thermodynamics

Gas laws with no space or time in them, and one that has both.

## The idea

Squeeze a gas and it heats up; heat a gas at fixed volume and its pressure
rises. The **ideal gas law** is the relationship between pressure, density,
and temperature that makes both of those true. The **adiabatic relation**
is what happens to pressure when a gas expands or compresses *without*
exchanging heat with anything around it (fast enough that heat has no time
to leak in or out).

Anything warm glows, even if not visibly: a red-hot poker and a room-
temperature rock are both radiating, just at very different wavelengths.
**Black-body radiation** is the physics of that glow: hotter objects radiate
more total power (Stefan-Boltzmann) and their glow shifts toward shorter,
bluer wavelengths (Wien's displacement law), which is why a star's color is
a direct readout of its surface temperature.

The **heat equation** is the odd one out here: everything above has no
space or time dependence, a single number in, a single number out. The heat
equation describes how temperature actually *spreads*, a hot spot cooling
into its surroundings over time, which needs a grid, the same kind
`Physics/Fluids`' Eulerian solver and `Physics/Electromagnetism`'s Maxwell
solver use.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Thermodynamics/Thermodynamics.hpp` | Ideal gas law, adiabatic relation, black-body luminosity, Wien's law, isothermal barometric profile |
| `Thermodynamics/HeatEquation.hpp` | `HeatEquation1D`: 1D diffusion, explicit finite-difference |

An atmosphere in hydrostatic equilibrium (weight balanced by pressure) with
the ideal gas law held at constant temperature gives one more closed form:
density falls off **exponentially** with altitude, `rho(h) = rho0 exp(-h /
H)`, `H` (the "scale height") set by how strong gravity is and how warm and
light the gas is. General for any planet's atmosphere; `Optics`'s
refraction and scattering laws both reuse this same shape, since a
refractive index and a scattering rate both ultimately trace back to how
dense the air actually is at a given height. See
[docs/physics/optics.md](optics.md).

## Using it

```cpp
#include <Physics/Thermodynamics/Thermodynamics.hpp>

const ysq::Pressure p =
    ysq::idealGasPressure(density, specificGasConstant, temperature);
const ysq::Power luminosity = ysq::blackBodyLuminosity(starRadius, starTemperature);

const ysq::Length scaleHeight =
    ysq::isothermalScaleHeight(specificGasConstant, temperature, surfaceGravity);
const ysq::Density atThatHeight =
    ysq::isothermalAtmosphereDensity(seaLevelDensity, altitude, scaleHeight);
```

```cpp
#include <Physics/Thermodynamics/HeatEquation.hpp>

ysq::HeatEquation1D heat(cellCount, spacing, diffusivity);
heat.setTemperature(cell, value);
heat.step(heat.stableTimeStep(/*safetyFactor=*/0.9));
```

## Go deeper

[docs/api/physics/thermodynamics.md](../api/physics/thermodynamics.md) has
every signature: the gas-law and black-body functions, and
`HeatEquation1D`'s full interface.

[src/Physics/README.md](../../src/Physics/README.md) has the exact
Stefan-Boltzmann constant and why it's computed from the SI-defining
constants rather than typed as its own measured value, the FTCS stability
condition `HeatEquation1D` enforces, and how the heat equation is validated
against its known exact solution (a Gaussian temperature profile stays
Gaussian, with its width growing predictably over time).

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/thermodynamics)
and let us know.
