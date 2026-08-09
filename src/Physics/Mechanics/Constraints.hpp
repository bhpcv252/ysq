#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Units/Length.hpp>
#include <Units/Time.hpp>

namespace ysq {

/// Sequential-impulse constraint solving: each function applies one
/// instantaneous impulse that drives the constrained quantity (a
/// separation, a position) toward satisfying the constraint at the
/// velocity level, plus a Baumgarte stabilization term
/// (`baumgarteFactor * error / dt`) that also corrects whatever
/// positional error has already accumulated. Calling one of these once per
/// constraint per timestep is a single Gauss-Seidel sweep; a chain of
/// several constraints (a rope, a linkage) converges toward simultaneous
/// satisfaction over a handful of sweeps rather than needing an exact
/// simultaneous solve, the same trade real-time constraint solvers
/// (Box2D and similar) make and the reason this is not built on
/// `Math/LinearSolve.hpp`'s exact solve instead.
///
/// `baumgarteFactor` trades correction speed against stability: 0 removes
/// positional correction entirely (drift accumulates and is never
/// corrected, only velocity is kept consistent going forward), 1 corrects
/// all of it in one step (which overshoots and can oscillate); the
/// conventional 0.1-0.3 range corrects most drift within a handful of
/// steps without either problem.

/// Keeps `a` and `b` at `targetDistance` apart: a rigid rod, a pendulum's
/// own two ends, one link of a chain.
void solveDistanceConstraint(Body& a, Body& b, Length targetDistance, Time dt,
                             double baumgarteFactor = 0.2);

/// The same constraint against a fixed point `anchor` (the limit of the
/// two-body form above as the second body's mass goes to infinity): a
/// pendulum's own pivot, a rope's fixed end.
void solveDistanceConstraint(Body& body, const Length3& anchor, Length targetDistance,
                             Time dt, double baumgarteFactor = 0.2);

/// Pins `body`'s position to a fixed point `anchor` exactly (all three
/// components at once, not merely their separation): a hinge with zero
/// length, a weld. The vector generalization of `solveDistanceConstraint`
/// at `targetDistance = 0`, but constraining every axis simultaneously
/// rather than only the radial one, since "zero distance" alone does not
/// pin a direction.
void solvePointConstraint(Body& body, const Length3& anchor, Time dt,
                          double baumgarteFactor = 0.2);

}  // namespace ysq
