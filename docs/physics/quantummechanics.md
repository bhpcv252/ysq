# Quantum mechanics

The Schrödinger equation, two ways: what states a system can be in, and how
a state evolves.

## The idea

The Schrödinger equation is quantum mechanics' equation of motion, and it
comes in two flavors that answer different questions. The
**time-independent** equation, `-hbar^2/(2m) d^2(psi)/dx^2 + V(x) psi = E
psi`, asks "what stationary states can a particle in this potential be in,
and at what energies?" Discretized on a grid, this becomes an ordinary
symmetric matrix eigenvalue problem: each eigenvector is an allowed
wavefunction, its eigenvalue the corresponding energy, ascending. The
**time-dependent** equation, `i hbar d(psi)/dt = [-hbar^2/(2m) d^2/dx^2 +
V(x)] psi`, asks "given a state right now, what does it look like a moment
later?" and is solved here by the split-step Fourier method: the kinetic
term is diagonal (a pure per-mode phase) in momentum space but expensive in
position space, so each step transforms to momentum space, applies that
phase, and transforms back.

`hbar` and `mass` are always explicit, never defaulted, in every function
here: real SI quantum mechanics has extremely small numbers (`hbar ~
1e-34`), and natural/atomic units (`hbar = 1`) are just as valid a choice a
caller might make instead, so nothing assumes one over the other.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `QuantumMechanics/Schrodinger.hpp` | `solveTimeIndependentSchrodinger`: eigenstates and energies, 1D |
| `QuantumMechanics/Schrodinger3D.hpp` | `solveTimeIndependentSchrodinger3D`: the same eigenvalue problem, 3D |
| `QuantumMechanics/WavePacket.hpp` | `TimeDependentWavefunction1D`: split-step Fourier time evolution, 1D |
| `QuantumMechanics/WavePacket3D.hpp` | `TimeDependentWavefunction3D`: the same evolution, 3D |

**This is the reason `Math/FFT.hpp` exists in this engine at all**, ahead
of whichever other consumer might have asked for it first: the
time-dependent solver's whole method depends on a fast transform between
position and momentum space every single step.

**The 3D eigenvalue problem stays a dense solver, so it only scales to
modest grids.** The 3D Hamiltonian is `N x N` for `N = nx*ny*nz`, and
`Math/Eigen.hpp` has no sparse/iterative eigensolver (Lanczos, say);
`jacobiEigenSymmetric` is `O(N^3)`, so `solveTimeIndependentSchrodinger3D`
is correct and general but only practical for small grids (validated up
to `8x8x8 = 512`), not a production-scale simulation. A sparse eigensolver
would be new, separate scope, not an extension of what's here.

Each time-dependent class's own `expectationEnergy()` is computed by the
same finite-difference Hamiltonian the corresponding eigenvalue solver
uses, deliberately *not* the FFT the propagator itself runs on: that makes
it an independent check that a time-independent potential really does
conserve energy, not a tautology that would pass even if `step` were wrong.

## Using it

```cpp
#include <Physics/QuantumMechanics/Schrodinger.hpp>

const ysq::QuantumEigenstates states =
    ysq::solveTimeIndependentSchrodinger(potential, spacing, mass, hbar);
// states.energies[0]: the ground-state energy
// states.wavefunctions[0]: the ground-state wavefunction, normalized
```

```cpp
#include <Physics/QuantumMechanics/WavePacket.hpp>

ysq::TimeDependentWavefunction1D psi(initialWavefunction, potential, spacing, mass, hbar);
psi.step(dt);
const double probability = psi.totalProbability();  // stays 1: unitarity is structural here
```

The same two problems in 3D:

```cpp
#include <Physics/QuantumMechanics/Schrodinger3D.hpp>
#include <Physics/QuantumMechanics/WavePacket3D.hpp>

const ysq::QuantumEigenstates3D states3D =
    ysq::solveTimeIndependentSchrodinger3D(potential, nx, ny, nz, spacing, mass, hbar);

ysq::TimeDependentWavefunction3D psi3D(initialWavefunction, potential, nx, ny, nz, spacing,
                                       mass, hbar);
psi3D.step(dt);
```

## Go deeper

[docs/api/physics/quantummechanics.md](../api/physics/quantummechanics.md)
has every signature: both eigenvalue solvers and both time-dependent
classes' full interfaces.

[src/Physics/README.md](../../src/Physics/README.md) has the full
derivations: the 3-point and 7-point Hamiltonian discretizations, the
Strang-split split-step Fourier scheme, and how each solver is validated,
the 1D and 3D infinite-well and harmonic-oscillator closed forms, the
dimensional-reduction identity against the 1D solvers, and unitarity/energy
conservation over many time-dependent steps.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/quantummechanics)
and let us know.
