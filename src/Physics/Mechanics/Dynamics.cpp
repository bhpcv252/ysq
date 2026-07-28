#include <Physics/Mechanics/Dynamics.hpp>

#include <Units/Length.hpp>
#include <Units/Mass.hpp>

#include <cassert>
#include <cstddef>

namespace ysq {

NBodyState positionsOf(std::span<const Body> bodies) {
    NBodyState result(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        result[i] = bodies[i].position.value();
    }
    return result;
}

NBodyState velocitiesOf(std::span<const Body> bodies) {
    NBodyState result(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        result[i] = bodies[i].velocity().value();
    }
    return result;
}

void applyState(std::span<Body> bodies, const NBodyState& positions,
                const NBodyState& velocities) {
    assert(bodies.size() == positions.size());
    assert(bodies.size() == velocities.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].position = Length3{positions[i]};
        bodies[i].momentum = Momentum3{velocities[i] * bodies[i].mass.value()};
    }
}

}  // namespace ysq
