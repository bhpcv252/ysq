#pragma once

#include <Physics/Body.hpp>
#include <Units/Length.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

/// An inertial reference frame: an origin and a velocity, both measured in
/// whatever frame a Body handed to transformTo/transformFrom is already
/// expressed in.
///
/// Only a Galilean transform lives here. Frame's job is bookkeeping for the
/// regime Newtonian gravity and mechanics already assume, v << c. The
/// relativistic transform of a velocity between frames related by more than a
/// small offset needs the Lorentz factor, and that machinery lives in
/// Mechanics/Kinematics.hpp instead, alongside everything else relativity
/// touches.
struct Frame {
    Length3 origin{};
    Velocity3 velocity{};

    [[nodiscard]] static constexpr Frame lab() noexcept { return Frame{}; }
};

/// `body`, as seen from `frame`. Position shifts by the frame's origin;
/// momentum shifts by the body's mass times the frame's velocity, since a
/// Galilean boost changes velocity by a constant offset and momentum follows.
[[nodiscard]] constexpr Body transformTo(const Frame& frame, const Body& body) noexcept {
    Body result = body;
    result.position = body.position - frame.origin;
    result.momentum = body.momentum - body.mass * frame.velocity;
    return result;
}

/// The inverse of transformTo: `body`, given as seen from `frame`, expressed
/// back in the frame `frame` itself is measured against.
[[nodiscard]] constexpr Body transformFrom(const Frame& frame,
                                           const Body& body) noexcept {
    Body result = body;
    result.position = body.position + frame.origin;
    result.momentum = body.momentum + body.mass * frame.velocity;
    return result;
}

}  // namespace ysq
