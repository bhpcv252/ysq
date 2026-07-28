#pragma once

#include <Units/Constants.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

/// Matter state: what a physical body is made of and where it is.
///
/// The dynamical variable is momentum, not velocity. In the Newtonian limit
/// velocity is just momentum over mass, but momentum is the primitive that
/// stays correct once something moves fast enough that Mechanics/Kinematics's
/// relativistic momentum, p = gamma m v, applies: velocity alone cannot
/// reconstruct that, while momentum already is that quantity.
struct Body {
    Mass mass{};
    ElectricCharge charge{};
    Length3 position{};
    Momentum3 momentum{};

    /// Non-relativistic velocity, p / m. Exact only while v << c; see
    /// Mechanics/Kinematics.hpp for the relativistic relation between the two.
    [[nodiscard]] constexpr Velocity3 velocity() const noexcept {
        return momentum / mass;
    }
};

}  // namespace ysq
