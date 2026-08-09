#pragma once

#include <Math/Vector3.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>

#include <cstddef>
#include <vector>

namespace ysq {

/// A general spherical-harmonics gravity field: the generalization of
/// `Gravity/Newtonian.hpp`'s single J2 term to a body whose real shape
/// needs more than one oblateness coefficient to describe (a tumbling
/// asteroid, a body with north-south or longitudinal mass asymmetry J2
/// alone cannot represent).
///
/// `cosine[i][m]` and `sine[i][m]` are the standard geodesy `C_nm`/`S_nm`
/// coefficients (unnormalized convention) for degree `n = i + 2` and order
/// `m = 0..n`, so `cosine[0]` holds degree 2's `m = 0, 1, 2` and so on.
/// `C_2,0 = -J2`, every other coefficient zero, reproduces
/// `Gravity/Newtonian.hpp`'s oblateness term exactly.
///
/// Coefficients are expressed in the source body's own frame -- the frame
/// its own shape was measured or modeled in, which generally rotates in
/// the inertial frame. A caller with a rotating source rotates `position`
/// into that body frame before calling either function below, the same
/// boundary `Gravity/Newtonian.hpp`'s own J2 term draws with its source's
/// spin axis.
struct SphericalHarmonicsField {
    Mass mass;
    Length referenceRadius;
    std::vector<std::vector<double>> cosine;
    std::vector<std::vector<double>> sine;
};

/// The geopotential, in the geodesy sign convention (positive, increasing
/// toward the source, unlike a potential *energy*):
///
/// ```
/// U(r) = (GM/r) [1 + sum_n (Re/r)^n sum_m P_n^m(sin(phi))
///                      (C_nm cos(m lambda) + S_nm sin(m lambda))]
/// ```
///
/// `phi` and `lambda` are `position`'s own latitude and longitude in the
/// field's body frame; `position` is relative to the source's center.
[[nodiscard]] double sphericalHarmonicsPotential(const SphericalHarmonicsField& field,
                                                 const Vec3& position);

/// The gravitational acceleration at `position`: `+grad(U)` above, the sign
/// this geodesy convention needs (U increases toward the source, the same
/// way `GM/r` alone does, so its gradient already points the right way
/// with no extra sign to track).
///
/// Computed by central-difference numerical differentiation
/// (`Math/Calculus.hpp`'s `numericalGradient`) rather than an analytic
/// partial derivative: the associated Legendre functions' own derivatives
/// need a separate recurrence relation with singularities at the poles to
/// get right, and a finite-difference gradient of the perfectly ordinary,
/// pole-free Cartesian potential above sidesteps needing it, at the cost of
/// the usual finite-difference precision (about eight digits) rather than
/// machine precision -- adequate for a general force law, and worth
/// revisiting with an exact analytic gradient only once a real consumer
/// needs more.
[[nodiscard]] Acceleration3
sphericalHarmonicsAcceleration(const SphericalHarmonicsField& field,
                               const Length3& position);

}  // namespace ysq
