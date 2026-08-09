#pragma once

#include <Physics/Mechanics/Spring.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace ysq {

/// A 1D chain of elastic bar elements -- a discretized deformable rod, or
/// equally a mass-spring lattice along one axis -- solved for static
/// equilibrium via a direct linear solve (`Math/LinearSolve.hpp`), not
/// `Mechanics/Constraints.hpp`'s sequential impulses: a static problem has
/// no time to step through, so one exact simultaneous solve is both the
/// natural and the cheaper choice, where a dynamic constraint chain would
/// have to reassemble and re-solve its own system every single timestep
/// instead.
class ElasticChain1D {
public:
    /// `segmentStiffness[i]` is the spring constant of the segment
    /// connecting node `i` and node `i + 1` (from
    /// `Continuum/Elasticity.hpp`'s `equivalentSpringConstant`, or any
    /// other `SpringConstant` a caller already has), so there are
    /// `segmentStiffness.size() + 1` nodes in all.
    explicit ElasticChain1D(std::vector<SpringConstant> segmentStiffness);

    [[nodiscard]] std::size_t nodeCount() const noexcept;

    /// Solves for every node's displacement under `load` (one force per
    /// node; `load.size()` must equal `nodeCount()`, though node 0's own
    /// entry is never read, since it is held fixed), with node 0 pinned at
    /// zero displacement -- the standard clamped-rod boundary condition.
    /// `nullopt` only if the resulting system is singular, which for this
    /// particular assembly happens only if some segment's own stiffness is
    /// exactly zero.
    [[nodiscard]] std::optional<std::vector<Length>>
    solveDisplacements(std::span<const Force> load) const;

private:
    std::vector<double> m_stiffness;  // raw N/m, one per segment
};

}  // namespace ysq
