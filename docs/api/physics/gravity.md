# Physics/Gravity API reference

The gravity ladder: Newtonian direct summation, Barnes-Hut, and the 1PN
correction. Start with
[docs/physics/gravity.md](../../physics/gravity.md) for which rung to reach
for; [src/Physics/README.md](../../../src/Physics/README.md) has softening,
the opening-angle tradeoff, and the precession derivation in full.

## `Physics/Gravity/Newtonian.hpp`

`F = G m1 m2 / r^2`, toward the source: the weak-field, slow-motion limit
of general relativity and the rung that drives dynamical many-body systems.

```cpp
using GravitationalConstant = Quantity<dim::GravitationalConstant>;   // L^3 / (M T^2)

namespace constants {
    inline constexpr GravitationalConstant G{6.67430e-11};   // measured, CODATA 2018/2022, ~2.2e-5 relative uncertainty
}

Force3 newtonianForce(const Body& on, const Body& from);

Acceleration3 newtonianAcceleration(const Length3& at, std::span<const Body> sources,
                                    Length softening = Length::zero());

std::vector<Acceleration3> newtonianAccelerations(std::span<const Body> bodies,
                                                   Length softening = Length::zero());
// O(n^2) direct sum; see BarnesHutTree for O(n log n)

Energy newtonianPotentialEnergy(std::span<const Body> bodies,
                                Length softening = Length::zero());

class NewtonianField {
public:
    explicit NewtonianField(std::span<const Body> bodies, Length softening = Length::zero());
    NBodyState operator()(double time, const NBodyState& positions) const;
    // matches AccelerationField<NBodyState>: hand directly to VelocityVerletStepper<NBodyState> etc.
};
```

| Member | Description |
| --- | --- |
| `constants::G` | Measured, not one of `Units`'s seven SI-defining constants: it parameterizes one specific interaction, so it lives with the gravity that uses it, not in `Units/Constants.hpp`. Prefer a source's mass parameter GM directly (e.g. `constants::nominalSolarMassParameter`) where it's known: GM is what an orbit actually measures, and doesn't carry `G`'s uncertainty. |
| `softening` | Every function that takes one uses **Plummer softening**: `a = GM (r_j - r_i) / (\|r_j - r_i\|^2 + softening^2)^(3/2)`, the same force well above the softening length, finite rather than singular as bodies approach each other. Needed for direct-summation N-body integration. |
| `newtonianPotentialEnergy` | Must be called with the **same softening** the acceleration was integrated with: a conservation check is only meaningful for the softened system actually integrated, not the true `1/r` one. |
| `NewtonianField` | The `AccelerationField` to hand a `Math` stepper. Precomputes `G * mass` per body at construction. |

```cpp
ysq::NewtonianField field(bodies, softening);
ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
stepper.step(field, time, state, h, next);
```

## `Physics/Gravity/BarnesHut.hpp`

Approximate N-body gravity: a distant group of bodies is treated as one
point mass at their center of mass. Barnes & Hut, *A hierarchical O(N log N)
force-calculation algorithm*, Nature 324 (1986), 446-449.

```cpp
class BarnesHutTree {
public:
    explicit BarnesHutTree(std::span<const Body> bodies, double openingAngle = 0.5,
                           Length softening = Length::zero());
    NBodyState operator()(double time, const NBodyState& positions) const;
    // rebuilds the tree from `positions` every call; matches AccelerationField<NBodyState>
};
```

| Parameter | Description |
| --- | --- |
| `openingAngle` (theta) | A tree node of width `s` at distance `d` is accepted as a single mass when `s/d < theta`; otherwise the search recurses into its children. `theta = 0` degenerates to direct summation (every node opened); a larger `theta` accepts coarser, more distant approximations. `0.5` is the conventional default. |

The tree is **monopole only**: each internal node carries a total mass and
center of mass, not higher moments; a quadrupole correction would tighten
the error at fixed `theta` and isn't implemented. Rebuilt from scratch on
every call (the bodies have generally moved since the last one); that
rebuild cost is what buys O(N log N) per force evaluation over direct
summation's O(N^2).

```cpp
ysq::BarnesHutTree tree(bodies, 0.5, softening);
ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
stepper.step(tree, time, state, h, next);   // same AccelerationField shape as NewtonianField
```

## `Physics/Gravity/PostNewtonian.hpp`

The 1PN (first post-Newtonian) correction to the gravitational acceleration
on a test particle orbiting a dominant source: the correction that
produces perihelion precession, standard PPN form with `gamma = beta = 1`
(general relativity):

```cpp
Acceleration3 postNewtonianCorrection(const Body& testParticle, const Body& source);
```

```
a_1PN = (GM / (c^2 r^2)) * [ (4 GM/r - v^2) n + 4 (v . n) v ]
```

where `r`, `v` are the position and velocity of `testParticle` relative to
`source`, `n` is the unit vector from source to test particle, and `GM` uses
only the source's mass.

**Scope: two bodies, one a test particle.** Exact in the limit that
`source`'s mass dominates (Mercury around the Sun, not two comparable
masses), the same regime the analytic precession formula it's validated
against assumes. A full N-body correction (the Einstein-Infeld-Hoffmann
equations, with cross terms between every pair) is not implemented.

The two rungs of the ladder **compose rather than replace one another**:
add this to the Newtonian acceleration:

```cpp
const ysq::Acceleration3 total =
    ysq::newtonianAcceleration(testParticle.position, {&source, 1}) +
    ysq::postNewtonianCorrection(testParticle, source);
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/gravity)
and let us know.
