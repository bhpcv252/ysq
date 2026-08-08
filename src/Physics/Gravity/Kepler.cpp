#include <Physics/Gravity/Kepler.hpp>

#include <Math/Scalar.hpp>

#include <cmath>

namespace ysq {

namespace {

/// Into `[0, tau)`: keeps every angle this file works with small and
/// precise regardless of how large `elapsedSeconds` is, rather than letting
/// mean anomaly grow without bound and lose precision (or feed
/// `trueAnomalyFromMeanAnomaly`'s Newton-Raphson a residual computed from
/// two nearly-equal huge numbers) the longer a caller propagates forward.
[[nodiscard]] double wrapAngle(double angle) {
    double wrapped = std::fmod(angle, kTau<double>);
    if (wrapped < 0.0) {
        wrapped += kTau<double>;
    }
    return wrapped;
}

}  // namespace

KeplerStateVector stateVectorFromElements(const OrbitalElements& elements, double gm) {
    const double a = elements.semiMajorAxis;
    const double e = elements.eccentricity;
    const double nu = elements.trueAnomaly;

    const double r = a * (1.0 - e * e) / (1.0 + e * std::cos(nu));
    const double h = std::sqrt(gm * a * (1.0 - e * e));

    const double xPf = r * std::cos(nu);
    const double yPf = r * std::sin(nu);
    const double vxPf = -(gm / h) * std::sin(nu);
    const double vyPf = (gm / h) * (e + std::cos(nu));

    const double cosO = std::cos(elements.longitudeOfAscendingNode);
    const double sinO = std::sin(elements.longitudeOfAscendingNode);
    const double cosI = std::cos(elements.inclination);
    const double sinI = std::sin(elements.inclination);
    const double cosW = std::cos(elements.argumentOfPeriapsis);
    const double sinW = std::sin(elements.argumentOfPeriapsis);

    const double r11 = cosO * cosW - sinO * sinW * cosI;
    const double r12 = -cosO * sinW - sinO * cosW * cosI;
    const double r21 = sinO * cosW + cosO * sinW * cosI;
    const double r22 = -sinO * sinW + cosO * cosW * cosI;
    const double r31 = sinW * sinI;
    const double r32 = cosW * sinI;

    return KeplerStateVector{
        Vec3{r11 * xPf + r12 * yPf, r21 * xPf + r22 * yPf, r31 * xPf + r32 * yPf},
        Vec3{r11 * vxPf + r12 * vyPf, r21 * vxPf + r22 * vyPf, r31 * vxPf + r32 * vyPf}};
}

double trueAnomalyFromMeanAnomaly(double meanAnomaly, double eccentricity) {
    const double e = eccentricity;

    // E0 = M + e sin(M): the standard better-than-M starting guess: it
    // reaches double-precision convergence in noticeably fewer iterations
    // than starting from E0 = M, though either converges for any bound
    // orbit (e in [0, 1)).
    double eccentricAnomaly = meanAnomaly + e * std::sin(meanAnomaly);

    for (int iteration = 0; iteration < 50; ++iteration) {
        const double f = eccentricAnomaly - e * std::sin(eccentricAnomaly) - meanAnomaly;
        const double fPrime = 1.0 - e * std::cos(eccentricAnomaly);
        const double step = f / fPrime;
        eccentricAnomaly -= step;
        if (std::abs(step) < 1.0e-14) {
            break;
        }
    }

    // Not needed for correctness (the doubled atan2 below already returns
    // the same physical angle for eccentricAnomaly shifted by any whole
    // number of turns, cos/sin only ever seeing it mod tau), only to keep
    // the value the half-angle formula actually works with bounded: a
    // caller that has propagated meanAnomaly over a very large elapsed
    // time hands this a correspondingly large eccentricAnomaly, and
    // wrapping it here keeps sin/cos evaluated on a small argument instead
    // of however many turns Newton-Raphson's own input accumulated.
    const double wrappedEccentricAnomaly = std::fmod(eccentricAnomaly, kTau<double>);

    // The atan2 form of the half-angle relation
    // tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2): numerically robust across all
    // quadrants, unlike the raw tan/atan version.
    return 2.0 * std::atan2(std::sqrt(1.0 + e) * std::sin(wrappedEccentricAnomaly / 2.0),
                            std::sqrt(1.0 - e) * std::cos(wrappedEccentricAnomaly / 2.0));
}

double keplerMeanMotion(double gm, double semiMajorAxis) {
    return std::sqrt(gm / (semiMajorAxis * semiMajorAxis * semiMajorAxis));
}

double keplerOrbitalPeriod(double gm, double semiMajorAxis) {
    return kTau<double> / keplerMeanMotion(gm, semiMajorAxis);
}

KeplerStateVector stateVectorAtTime(const OrbitalElementsAtEpoch& elements, double gm,
                                    double elapsedSeconds) {
    const double a = elements.semiMajorAxis;
    const double meanMotion = keplerMeanMotion(gm, a);

    const double meanAnomaly =
        wrapAngle(elements.meanAnomalyAtEpoch + meanMotion * elapsedSeconds);
    const double argumentOfPeriapsis = wrapAngle(
        elements.argumentOfPeriapsis + elements.precessionRatePerSecond * elapsedSeconds);

    OrbitalElements atTime{};
    atTime.semiMajorAxis = a;
    atTime.eccentricity = elements.eccentricity;
    atTime.inclination = elements.inclination;
    atTime.longitudeOfAscendingNode = elements.longitudeOfAscendingNode;
    atTime.argumentOfPeriapsis = argumentOfPeriapsis;
    atTime.trueAnomaly = trueAnomalyFromMeanAnomaly(meanAnomaly, elements.eccentricity);

    return stateVectorFromElements(atTime, gm);
}

}  // namespace ysq
