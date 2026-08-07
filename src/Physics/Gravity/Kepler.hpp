#pragma once

#include <Math/Vector3.hpp>

namespace ysq {

/// Classical orbital elements: the standard six-number description of an
/// unperturbed (two-body) ellipse.
///
/// Used only to build a physically sensible *initial* state vector for a
/// real n-body integrator to take over from -- the same "osculating
/// elements" convention real astrodynamics uses: nothing here is
/// re-consulted once a simulation starts.
struct OrbitalElements {
    double semiMajorAxis;
    double eccentricity;
    double inclination;               // radians, from the reference plane
    double longitudeOfAscendingNode;  // radians
    double argumentOfPeriapsis;       // radians
    double trueAnomaly;               // radians, the starting point on the ellipse
};

struct KeplerStateVector {
    Vec3 position;
    Vec3 velocity;
};

/// The standard perifocal-to-reference-frame rotation (Rz(Omega) Rx(i)
/// Rz(omega)) applied to both position and velocity in one pass, and the
/// standard perifocal-plane position/velocity formulas ahead of it: this is
/// the one, well-known way classical orbital elements become a Cartesian
/// state vector. `gm` is the *central* body's own gravitational parameter,
/// G times its mass, not the combined system's; the caller adds the central
/// body's own position/velocity afterward.
[[nodiscard]] KeplerStateVector stateVectorFromElements(const OrbitalElements& elements,
                                                         double gm);

/// Published orbital elements (JPL's included) give the mean anomaly `M`,
/// the angle a body *would* have if it moved at a constant rate around the
/// ellipse -- not the true anomaly `stateVectorFromElements` needs, the
/// body's actual angle. The two are related by Kepler's equation,
/// `M = E - e sin(E)`, for the eccentric anomaly `E`, solved here by
/// Newton-Raphson (a handful of iterations reach double precision for any
/// bound orbit, `e` in `[0, 1)`) and then converted to true anomaly by the
/// standard half-angle relation
/// `tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2)`.
[[nodiscard]] double trueAnomalyFromMeanAnomaly(double meanAnomaly, double eccentricity);

/// The two-body mean motion, `n = sqrt(gm / a^3)`: the constant rate a body
/// on an unperturbed ellipse of semi-major axis `a` around a source of
/// gravitational parameter `gm` would sweep its mean anomaly at.
[[nodiscard]] double keplerMeanMotion(double gm, double semiMajorAxis);

/// The orbital period, `2 pi / n`, for the same two-body ellipse
/// `keplerMeanMotion` describes.
[[nodiscard]] double keplerOrbitalPeriod(double gm, double semiMajorAxis);

/// Classical orbital elements anchored at an epoch, for repeated
/// re-evaluation at any later time rather than the single initial state
/// `OrbitalElements` above is used for. The orbit's shape (`semiMajorAxis`,
/// `eccentricity`, `inclination`, `longitudeOfAscendingNode`) does not
/// change with time; `meanAnomalyAtEpoch` is this orbit's position at
/// `elapsedSeconds = 0`, propagated forward at `keplerMeanMotion` by
/// `stateVectorAtTime` below.
///
/// `precessionRatePerSecond` rotates `argumentOfPeriapsis` at a constant
/// rate: zero reproduces a fixed, non-precessing ellipse exactly (every
/// existing consumer of plain `OrbitalElements` is a special case of this
/// with rate zero), nonzero is how a caller not running a real integrator
/// still shows a real effect like relativistic perihelion advance --
/// `Physics/Gravity/PostNewtonian.hpp`'s `perihelionPrecessionPerOrbit`
/// gives the physical rate to convert into this field (divide by
/// `keplerOrbitalPeriod` to get radians per second from radians per orbit).
struct OrbitalElementsAtEpoch {
    double semiMajorAxis;
    double eccentricity;
    double inclination;
    double longitudeOfAscendingNode;
    double argumentOfPeriapsis;
    double meanAnomalyAtEpoch;
    double precessionRatePerSecond = 0.0;
};

/// `stateVectorFromElements`, at `elapsedSeconds` after the epoch
/// `elements` is anchored at, instead of at the epoch itself: advances the
/// mean anomaly at the two-body mean motion and, if `precessionRatePerSecond`
/// is nonzero, the argument of periapsis, then solves Kepler's equation and
/// converts exactly as `stateVectorFromElements` already does. Cost is the
/// same regardless of how large `elapsedSeconds` is -- a handful of
/// Newton-Raphson iterations and a fixed rotation, not a number of state
/// evaluations proportional to the elapsed time the way stepping a real
/// integrator forward would be.
[[nodiscard]] KeplerStateVector stateVectorAtTime(const OrbitalElementsAtEpoch& elements,
                                                   double gm, double elapsedSeconds);

}  // namespace ysq
