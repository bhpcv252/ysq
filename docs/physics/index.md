# Physics

The engine's theories: what things are made of, and the laws that move them.
Start here before the seven pages below.

## Two ideas that shape everything in this section

**Gravity is a ladder, not one model.** Simulating general relativity
exactly, for many bodies interacting with each other, is intractable:
nobody can solve it that way, not YSQ, not anyone. So instead of one model,
`Physics` offers a ladder of approximations, each valid in a different
regime, and a scenario picks the rung that fits:

- **Newtonian** gravity (`F = Gm1m2/r^2`) is the weak-field, slow-motion
  limit of general relativity. It's what drives dynamical many-body systems
  like a solar system, a binary pair, or a galaxy, and it's what almost
  every everyday intuition about gravity already is.
- **Post-Newtonian** corrections add the leading relativistic effects (like
  the extra perihelion precession Mercury's orbit shows) on top of the
  Newtonian force, for a test particle orbiting one dominant mass.
- **Fixed background spacetimes** (Schwarzschild, Kerr, FLRW) are full
  general relativity, but only tractable because the geometry itself is
  fixed rather than evolving: a light ray or a test particle moves through
  a spacetime shaped by one dominant mass or by cosmic expansion, without
  that spacetime changing in response. This is exact where it applies:
  near a black hole, or at cosmological scale, not for many comparably
  massive bodies pulling on each other.

**Spacetime is the core abstraction.** A metric is a rule for measuring
distance and time at every point in space, and once you have one, "how does
anything move through it" has a single answer: a geodesic, the straightest
possible path. That single idea is why light propagation, gravitational
lensing, and every kind of frequency shift (Doppler, gravitational,
cosmological) are not three separate features in `Physics/Optics`. They're
the same computation (a null geodesic through a metric) looked at three
different ways. See [Spacetime](spacetime.md) and [Optics](optics.md).

## The seven theories

| Page | Covers |
| --- | --- |
| [Mechanics](mechanics.md) | `Body`, reference frames, proper time and the Lorentz factor, the boundary where dimensioned quantities cross into a plain integrator |
| [Gravity](gravity.md) | The ladder above, in depth: Newtonian force and softening, Barnes-Hut summation, the 1PN correction |
| [Spacetime](spacetime.md) | Metrics, Christoffel symbols, geodesics; Minkowski, Schwarzschild, Kerr, FLRW |
| [Electromagnetism](electromagnetism.md) | Coulomb and Biot-Savart fields, the Lorentz force, a Maxwell FDTD solver that actually propagates |
| [Fluids](fluids.md) | Two approaches to the same physics: particles that carry the fluid (SPH), and a fixed grid the fluid flows through |
| [Thermodynamics](thermodynamics.md) | Ideal gas law, black-body radiation, and diffusion (the heat equation) |
| [Optics](optics.md) | Light propagation, gravitational lensing, and frequency shift, unified by the spacetime idea above |

Each theory is organized as its own sibling module rather than nested
inside whichever application happens to need it: `Mechanics`, `Gravity`,
`Spacetime`, `Electromagnetism`, `Fluids`, `Thermodynamics`, and `Optics`
don't depend on one another, only on `Math` and `Units` beneath them. A
scenario in `Applications/` composes the theories and the rung of gravity
it needs; the theories themselves stay ignorant of any particular scenario.
See [docs/applications.md](../applications.md).

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/index)
and let us know.
