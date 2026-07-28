#pragma once

#include <Physics/Body.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Units/Length.hpp>

#include <span>
#include <vector>

namespace ysq {

/// Approximate N-body gravity by treating a distant group of bodies as one
/// point mass at their center of mass: Barnes & Hut, "A hierarchical O(N log
/// N) force-calculation algorithm", Nature 324 (1986), 446-449.
///
/// A tree node of width s is accepted as a single mass when its distance d
/// from the query point satisfies s/d < openingAngle; otherwise the search
/// recurses into its children. theta = 0 degenerates to direct summation
/// (every node is opened); a large theta accepts coarse, distant
/// approximations more readily. 0.5 is the conventional default. See
/// docs/physics.md and nbody_energy's error-vs-theta check for what that
/// trades off.
///
/// The tree is monopole only: each internal node carries a total mass and a
/// center of mass, not the mass distribution's higher moments. A quadrupole
/// correction would tighten the error at fixed theta and is a documented
/// possible refinement, not implemented here.
///
/// Rebuilt from scratch on every call, since the bodies it was built from
/// have generally moved since the last one. That is the cost that buys O(N
/// log N) per force evaluation instead of the O(N^2) of direct summation.
class BarnesHutTree {
public:
    explicit BarnesHutTree(std::span<const Body> bodies, double openingAngle = 0.5,
                           Length softening = Length::zero());

    /// Rebuilds the tree from `positions` and returns the acceleration on
    /// every body, in the order the constructor's `bodies` were given in.
    /// Matches AccelerationField<NBodyState>, so a
    /// VelocityVerletStepper<NBodyState> takes a BarnesHutTree directly.
    [[nodiscard]] NBodyState operator()(double time, const NBodyState& positions) const;

private:
    std::vector<double> m_gravitationalParameters;  // G * mass, per body, m^3/s^2
    double m_openingAngle;
    double m_softeningSquared;
};

}  // namespace ysq
