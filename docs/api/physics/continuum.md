# Physics/Continuum API reference

Linear elasticity: Hooke's law for a continuum, and a 1D chain of elastic
segments solved for static equilibrium. Start with
[docs/physics/continuum.md](../../physics/continuum.md) for the idea;
[src/Physics/README.md](../../../src/Physics/README.md) has the bar-element
derivation in full.

## `Physics/Continuum/Elasticity.hpp`

Hooke's law for a continuum: the generalization of
`Mechanics/Spring.hpp`'s point-mass version to a distributed elastic
medium.

```cpp
constexpr Stress hookeStress(Stress youngsModulus, Strain strain) noexcept;

constexpr Length axialExtension(Force load, Length naturalLength, Area crossSection,
                                Stress youngsModulus) noexcept;

constexpr SpringConstant equivalentSpringConstant(Stress youngsModulus, Area crossSection,
                                                   Length naturalLength) noexcept;
```

| Function | Description |
| --- | --- |
| `hookeStress(youngsModulus, strain)` | `stress = E * strain`. |
| `axialExtension(load, naturalLength, crossSection, youngsModulus)` | The direct consequence of Hooke's law (`stress = F/A`, `strain = extension/L0`) solved for `extension`. |
| `equivalentSpringConstant(youngsModulus, crossSection, naturalLength)` | `k = E A / L0`, the standard finite-element "bar element" stiffness: the bridge between this header and `Mechanics/Spring.hpp`'s `SpringConstant`, one spring generalized to a whole elastic continuum divided into segments. |

```cpp
const ysq::Length extension = ysq::axialExtension(load, naturalLength, crossSection, youngsModulus);
const ysq::SpringConstant k = ysq::equivalentSpringConstant(youngsModulus, crossSection, segmentLength);
```

## `Physics/Continuum/ElasticChain1D.hpp`

A 1D chain of elastic bar elements, a discretized deformable rod or
equally a mass-spring lattice along one axis, solved for static
equilibrium via a direct linear solve rather than
`Mechanics/Constraints.hpp`'s sequential impulses: a static problem has no
time to step through, so one exact simultaneous solve is both the natural
and the cheaper choice.

```cpp
class ElasticChain1D {
public:
    explicit ElasticChain1D(std::vector<SpringConstant> segmentStiffness);

    std::size_t nodeCount() const noexcept;

    std::optional<std::vector<Length>> solveDisplacements(std::span<const Force> load) const;
};
```

| Member | Description |
| --- | --- |
| Constructor | `segmentStiffness[i]` is the spring constant of the segment connecting node `i` and node `i+1`, so there are `segmentStiffness.size() + 1` nodes in all. |
| `solveDisplacements(load)` | Solves for every node's displacement under `load` (one force per node; `load.size()` must equal `nodeCount()`, though node 0's own entry is never read), with node 0 pinned at zero displacement. `nullopt` only if the system is singular, which for this assembly happens only if some segment's stiffness is exactly zero. |

```cpp
ysq::ElasticChain1D chain(segmentStiffness);
if (const auto displacements = chain.solveDisplacements(load)) {
    const ysq::Length tipDisplacement = displacements->back();
}
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/continuum)
and let us know.
