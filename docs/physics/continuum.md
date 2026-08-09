# Continuum mechanics

Hooke's law, stretched from one spring to a whole rod.

## The idea

`Physics/Mechanics/Spring.hpp`'s Hooke's law describes one spring: a force
proportional to how far it's stretched. A real elastic solid, a rod, a
beam, a cable, doesn't concentrate all its compliance at one point; it
stretches smoothly and continuously along its length. **Continuum
mechanics** is Hooke's law generalized that way: `stress = E * strain`,
where `E` (Young's modulus) plays the same role `SpringConstant` does for
a point spring, and stress/strain are the continuum's own stand-ins for
force/extension.

The practical bridge between the two: discretize a continuous rod into a
chain of short segments, and each segment behaves exactly like a spring of
stiffness `k = E A / L0` (`A` the cross-sectional area, `L0` the segment's
natural length), the standard finite-element "bar element". A whole rod is
then a chain of springs, solvable the same way any mass-spring lattice is.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Continuum/Elasticity.hpp` | Hooke's law for a continuum: stress/strain, axial extension under load, and the bar-element spring constant |
| `Continuum/ElasticChain1D.hpp` | `ElasticChain1D`: a 1D chain of elastic segments, solved for static equilibrium |

`ElasticChain1D` solves a genuinely different kind of problem from
`Mechanics/Constraints.hpp`'s sequential-impulse constraints: it's a
**static** equilibrium (no time to step through), so one exact simultaneous
linear solve (`Math/LinearSolve.hpp`) is both the natural and the cheaper
choice, where a dynamic constraint chain would have to reassemble and
re-solve its own system every single timestep instead. Node 0 is pinned at
zero displacement (the standard clamped-rod boundary condition), and the
chain is solved for every other node's displacement under an applied load.

## Using it

```cpp
#include <Physics/Continuum/Elasticity.hpp>
#include <Physics/Continuum/ElasticChain1D.hpp>

const ysq::SpringConstant segmentStiffness =
    ysq::equivalentSpringConstant(youngsModulus, crossSection, segmentLength);

std::vector<ysq::SpringConstant> stiffness(segmentCount, segmentStiffness);
ysq::ElasticChain1D chain(stiffness);

std::vector<ysq::Force> load(chain.nodeCount(), ysq::Force::zero());
load.back() = appliedLoad;

if (const auto displacements = chain.solveDisplacements(load)) {
    // displacements->at(i): node i's displacement from equilibrium
}
```

## Go deeper

[docs/api/physics/continuum.md](../api/physics/continuum.md) has every
signature: `hookeStress`, `axialExtension`, `equivalentSpringConstant`, and
`ElasticChain1D`'s full interface.

[src/Physics/README.md](../../src/Physics/README.md) has the full
derivation: the bar-element stiffness matrix `ElasticChain1D` assembles
from `equivalentSpringConstant`, and why a direct linear solve rather than
`Mechanics/Constraints.hpp`'s iterative approach is the right tool for a
static problem.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/continuum)
and let us know.
