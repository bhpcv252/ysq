#include <Physics/Fluids/Eulerian3D.hpp>

#include <Compute/ComputeBackend.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ysq {

namespace {

// Measured on the development machine (Apple Silicon, Metal backend) by
// benchmarks/compute_thresholds.cpp: a real observed crossover, and lower
// than the rest of the grid-stencil family (HeatEquation3D.cpp,
// Acoustic3D.cpp, Maxwell3D.cpp, Math/Multigrid.hpp, all measured floors)
// -- this kernel reads five conserved-quantity arrays and writes five
// more per cell, and (unlike the others) does real equation-of-state and
// Rusanov-flux work per face, not just neighbor averaging/differencing,
// so its higher per-cell cost crosses over to GPU-favorable at a smaller
// grid. Also keeps every existing grid in
// tests/unit/fluids_eulerian3d.cpp (at most 16^3) on the exact
// double-precision CPU path. Re-run the benchmark and update this if the
// reference machine or backend ever changes.
constexpr std::size_t kGpuDispatchThreshold = 262144;

struct Conserved {
    double density;
    double momentumNormal;
    double momentumTangent1;
    double momentumTangent2;
    double energy;
};

struct Flux {
    double density;
    double momentumNormal;
    double momentumTangent1;
    double momentumTangent2;
    double energy;
};

[[nodiscard]] double pressureFrom(const Conserved& state, double gamma) {
    const double u = state.momentumNormal / state.density;
    const double v = state.momentumTangent1 / state.density;
    const double w = state.momentumTangent2 / state.density;
    const double kinetic = 0.5 * state.density * (u * u + v * v + w * w);
    return (gamma - 1.0) * (state.energy - kinetic);
}

[[nodiscard]] double soundSpeedOf(double density, double pressure, double gamma) {
    return std::sqrt(gamma * pressure / density);
}

[[nodiscard]] Flux fluxOf(const Conserved& state, double gamma) {
    const double u = state.momentumNormal / state.density;
    const double v = state.momentumTangent1 / state.density;
    const double w = state.momentumTangent2 / state.density;
    const double pressure = pressureFrom(state, gamma);
    return {state.momentumNormal, state.momentumNormal * u + pressure,
            state.momentumNormal * v, state.momentumNormal * w,
            u * (state.energy + pressure)};
}

/// The Rusanov flux generalized to a state with two passive transverse
/// momentum components: `fluxOf` gives each transverse component's flux
/// as pure advection by the normal velocity (no pressure term, since a
/// 1D-normal Riemann problem along one axis has nothing to say about the
/// other two), while the dissipation term below still applies uniformly
/// to all five conserved quantities, `EulerianFluid1D::rusanovFlux`'s own
/// stabilization idea unchanged.
[[nodiscard]] Flux rusanovFlux(const Conserved& left, const Conserved& right,
                               double gamma) {
    const Flux fluxLeft = fluxOf(left, gamma);
    const Flux fluxRight = fluxOf(right, gamma);

    const double velocityLeft = left.momentumNormal / left.density;
    const double velocityRight = right.momentumNormal / right.density;
    const double soundLeft = soundSpeedOf(left.density, pressureFrom(left, gamma), gamma);
    const double soundRight =
        soundSpeedOf(right.density, pressureFrom(right, gamma), gamma);
    const double maxSpeed = std::max(std::abs(velocityLeft) + soundLeft,
                                     std::abs(velocityRight) + soundRight);

    return {0.5 * (fluxLeft.density + fluxRight.density) -
                0.5 * maxSpeed * (right.density - left.density),
            0.5 * (fluxLeft.momentumNormal + fluxRight.momentumNormal) -
                0.5 * maxSpeed * (right.momentumNormal - left.momentumNormal),
            0.5 * (fluxLeft.momentumTangent1 + fluxRight.momentumTangent1) -
                0.5 * maxSpeed * (right.momentumTangent1 - left.momentumTangent1),
            0.5 * (fluxLeft.momentumTangent2 + fluxRight.momentumTangent2) -
                0.5 * maxSpeed * (right.momentumTangent2 - left.momentumTangent2),
            0.5 * (fluxLeft.energy + fluxRight.energy) -
                0.5 * maxSpeed * (right.energy - left.energy)};
}

/// One dimensional-split sweep along `axis` (0 = x, 1 = y, 2 = z):
/// `momentumA` holds momentum along `axis`, the "normal" component whose
/// flux carries pressure; `momentumB`/`momentumC` are the other two,
/// carried along passively. Indexing follows
/// `Math/FiniteDifference.hpp`'s `axisOffset` convention: `otherA`/`otherB`
/// are whichever two axis indices are not `axis`.
void sweep(Grid3D<double>& density, Grid3D<double>& momentumA, Grid3D<double>& momentumB,
           Grid3D<double>& momentumC, Grid3D<double>& energy, double gamma, double dt,
           double dx, int axis) {
    const std::size_t nxCount = density.cellCountX();
    const std::size_t nyCount = density.cellCountY();
    const std::size_t nzCount = density.cellCountZ();

    if (nxCount * nyCount * nzCount >= kGpuDispatchThreshold) {
        const std::size_t total = nxCount * nyCount * nzCount;
        std::vector<float> densityData(total);
        std::vector<float> momentumAData(total);
        std::vector<float> momentumBData(total);
        std::vector<float> momentumCData(total);
        std::vector<float> energyData(total);
        for (std::size_t i = 0; i < nxCount; ++i) {
            for (std::size_t j = 0; j < nyCount; ++j) {
                for (std::size_t k = 0; k < nzCount; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    const std::size_t flat = (i * nyCount + j) * nzCount + k;
                    densityData[flat] = static_cast<float>(density(pi, pj, pk));
                    momentumAData[flat] = static_cast<float>(momentumA(pi, pj, pk));
                    momentumBData[flat] = static_cast<float>(momentumB(pi, pj, pk));
                    momentumCData[flat] = static_cast<float>(momentumC(pi, pj, pk));
                    energyData[flat] = static_cast<float>(energy(pi, pj, pk));
                }
            }
        }

        std::vector<float> nextDensity(total);
        std::vector<float> nextMomentumA(total);
        std::vector<float> nextMomentumB(total);
        std::vector<float> nextMomentumC(total);
        std::vector<float> nextEnergy(total);
        defaultBackend().eulerianFluid3DSweep(
            densityData, momentumAData, momentumBData, momentumCData, energyData, nxCount,
            nyCount, nzCount, axis, static_cast<float>(gamma),
            static_cast<float>(dt / dx), nextDensity, nextMomentumA, nextMomentumB,
            nextMomentumC, nextEnergy);

        for (std::size_t i = 0; i < nxCount; ++i) {
            for (std::size_t j = 0; j < nyCount; ++j) {
                for (std::size_t k = 0; k < nzCount; ++k) {
                    const auto pi = static_cast<std::ptrdiff_t>(i);
                    const auto pj = static_cast<std::ptrdiff_t>(j);
                    const auto pk = static_cast<std::ptrdiff_t>(k);
                    const std::size_t flat = (i * nyCount + j) * nzCount + k;
                    density(pi, pj, pk) = static_cast<double>(nextDensity[flat]);
                    momentumA(pi, pj, pk) = static_cast<double>(nextMomentumA[flat]);
                    momentumB(pi, pj, pk) = static_cast<double>(nextMomentumB[flat]);
                    momentumC(pi, pj, pk) = static_cast<double>(nextMomentumC[flat]);
                    energy(pi, pj, pk) = static_cast<double>(nextEnergy[flat]);
                }
            }
        }
        return;
    }

    density.applyPeriodicBoundary();
    momentumA.applyPeriodicBoundary();
    momentumB.applyPeriodicBoundary();
    momentumC.applyPeriodicBoundary();
    energy.applyPeriodicBoundary();

    const std::array<std::ptrdiff_t, 3> n{
        static_cast<std::ptrdiff_t>(density.cellCountX()),
        static_cast<std::ptrdiff_t>(density.cellCountY()),
        static_cast<std::ptrdiff_t>(density.cellCountZ())};
    const int otherA = (axis == 0) ? 1 : 0;
    const int otherB = (axis == 2) ? 1 : 2;

    const auto at = [&](std::ptrdiff_t sweepIndex, std::ptrdiff_t a, std::ptrdiff_t b) {
        std::array<std::ptrdiff_t, 3> idx{};
        idx[static_cast<std::size_t>(axis)] = sweepIndex;
        idx[static_cast<std::size_t>(otherA)] = a;
        idx[static_cast<std::size_t>(otherB)] = b;
        return idx;
    };
    const auto stateAt = [&](const std::array<std::ptrdiff_t, 3>& idx) {
        return Conserved{
            density(idx[0], idx[1], idx[2]), momentumA(idx[0], idx[1], idx[2]),
            momentumB(idx[0], idx[1], idx[2]), momentumC(idx[0], idx[1], idx[2]),
            energy(idx[0], idx[1], idx[2])};
    };

    const std::ptrdiff_t extentA = n[static_cast<std::size_t>(otherA)];
    const std::ptrdiff_t extentB = n[static_cast<std::size_t>(otherB)];
    const std::ptrdiff_t sweepCount = n[static_cast<std::size_t>(axis)];

    for (std::ptrdiff_t a = 0; a < extentA; ++a) {
        for (std::ptrdiff_t b = 0; b < extentB; ++b) {
            std::vector<Flux> fluxes(static_cast<std::size_t>(sweepCount) + 1);
            for (std::ptrdiff_t s = 0; s <= sweepCount; ++s) {
                const Conserved left = stateAt(at(s - 1, a, b));
                const Conserved right = stateAt(at(s, a, b));
                fluxes[static_cast<std::size_t>(s)] = rusanovFlux(left, right, gamma);
            }
            for (std::ptrdiff_t s = 0; s < sweepCount; ++s) {
                const std::array<std::ptrdiff_t, 3> idx = at(s, a, b);
                const Flux& fluxLeft = fluxes[static_cast<std::size_t>(s)];
                const Flux& fluxRight = fluxes[static_cast<std::size_t>(s + 1)];
                density(idx[0], idx[1], idx[2]) -=
                    (dt / dx) * (fluxRight.density - fluxLeft.density);
                momentumA(idx[0], idx[1], idx[2]) -=
                    (dt / dx) * (fluxRight.momentumNormal - fluxLeft.momentumNormal);
                momentumB(idx[0], idx[1], idx[2]) -=
                    (dt / dx) * (fluxRight.momentumTangent1 - fluxLeft.momentumTangent1);
                momentumC(idx[0], idx[1], idx[2]) -=
                    (dt / dx) * (fluxRight.momentumTangent2 - fluxLeft.momentumTangent2);
                energy(idx[0], idx[1], idx[2]) -=
                    (dt / dx) * (fluxRight.energy - fluxLeft.energy);
            }
        }
    }
}

}  // namespace

EulerianFluid3D::EulerianFluid3D(std::size_t cellCountX, std::size_t cellCountY,
                                 std::size_t cellCountZ, double spacing,
                                 double adiabaticIndex)
    : m_gamma(adiabaticIndex),
      m_density(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_momentumX(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_momentumY(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_momentumZ(cellCountX, cellCountY, cellCountZ, spacing, 1),
      m_energy(cellCountX, cellCountY, cellCountZ, spacing, 1) {}

std::size_t EulerianFluid3D::cellCountX() const noexcept {
    return m_density.cellCountX();
}
std::size_t EulerianFluid3D::cellCountY() const noexcept {
    return m_density.cellCountY();
}
std::size_t EulerianFluid3D::cellCountZ() const noexcept {
    return m_density.cellCountZ();
}
double EulerianFluid3D::spacing() const noexcept {
    return m_density.spacing();
}
double EulerianFluid3D::adiabaticIndex() const noexcept {
    return m_gamma;
}

void EulerianFluid3D::setState(std::size_t i, std::size_t j, std::size_t k,
                               double density, double velocityX, double velocityY,
                               double velocityZ, double pressure) {
    const auto ii = static_cast<std::ptrdiff_t>(i);
    const auto jj = static_cast<std::ptrdiff_t>(j);
    const auto kk = static_cast<std::ptrdiff_t>(k);
    const double kinetic =
        0.5 * density *
        (velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ);
    const double internal = pressure / (m_gamma - 1.0);

    m_density(ii, jj, kk) = density;
    m_momentumX(ii, jj, kk) = density * velocityX;
    m_momentumY(ii, jj, kk) = density * velocityY;
    m_momentumZ(ii, jj, kk) = density * velocityZ;
    m_energy(ii, jj, kk) = internal + kinetic;
}

double EulerianFluid3D::density(std::size_t i, std::size_t j, std::size_t k) const {
    return m_density(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                     static_cast<std::ptrdiff_t>(k));
}
double EulerianFluid3D::velocityX(std::size_t i, std::size_t j, std::size_t k) const {
    const auto ii = static_cast<std::ptrdiff_t>(i);
    const auto jj = static_cast<std::ptrdiff_t>(j);
    const auto kk = static_cast<std::ptrdiff_t>(k);
    return m_momentumX(ii, jj, kk) / m_density(ii, jj, kk);
}
double EulerianFluid3D::velocityY(std::size_t i, std::size_t j, std::size_t k) const {
    const auto ii = static_cast<std::ptrdiff_t>(i);
    const auto jj = static_cast<std::ptrdiff_t>(j);
    const auto kk = static_cast<std::ptrdiff_t>(k);
    return m_momentumY(ii, jj, kk) / m_density(ii, jj, kk);
}
double EulerianFluid3D::velocityZ(std::size_t i, std::size_t j, std::size_t k) const {
    const auto ii = static_cast<std::ptrdiff_t>(i);
    const auto jj = static_cast<std::ptrdiff_t>(j);
    const auto kk = static_cast<std::ptrdiff_t>(k);
    return m_momentumZ(ii, jj, kk) / m_density(ii, jj, kk);
}
double EulerianFluid3D::pressure(std::size_t i, std::size_t j, std::size_t k) const {
    const auto ii = static_cast<std::ptrdiff_t>(i);
    const auto jj = static_cast<std::ptrdiff_t>(j);
    const auto kk = static_cast<std::ptrdiff_t>(k);
    const Conserved state{m_density(ii, jj, kk), m_momentumX(ii, jj, kk),
                          m_momentumY(ii, jj, kk), m_momentumZ(ii, jj, kk),
                          m_energy(ii, jj, kk)};
    return pressureFrom(state, m_gamma);
}

double EulerianFluid3D::stableTimeStep(double courantNumber) const {
    const auto nx = static_cast<std::ptrdiff_t>(m_density.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_density.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_density.cellCountZ());
    double maxSpeed = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                const Conserved state{m_density(i, j, k), m_momentumX(i, j, k),
                                      m_momentumY(i, j, k), m_momentumZ(i, j, k),
                                      m_energy(i, j, k)};
                const double u = state.momentumNormal / state.density;
                const double v = m_momentumY(i, j, k) / state.density;
                const double w = m_momentumZ(i, j, k) / state.density;
                const double sound =
                    soundSpeedOf(state.density, pressureFrom(state, m_gamma), m_gamma);
                maxSpeed = std::max({maxSpeed, std::abs(u) + sound, std::abs(v) + sound,
                                     std::abs(w) + sound});
            }
        }
    }
    return courantNumber * m_density.spacing() / maxSpeed;
}

void EulerianFluid3D::step(double dt) {
    const double dx = m_density.spacing();
    sweep(m_density, m_momentumX, m_momentumY, m_momentumZ, m_energy, m_gamma, dt, dx, 0);
    sweep(m_density, m_momentumY, m_momentumX, m_momentumZ, m_energy, m_gamma, dt, dx, 1);
    sweep(m_density, m_momentumZ, m_momentumX, m_momentumY, m_energy, m_gamma, dt, dx, 2);
}

double EulerianFluid3D::totalMass() const {
    const auto nx = static_cast<std::ptrdiff_t>(m_density.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_density.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_density.cellCountZ());
    const double h = m_density.spacing();
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                total += m_density(i, j, k);
            }
        }
    }
    return total * h * h * h;
}

double EulerianFluid3D::totalMomentumX() const {
    const auto nx = static_cast<std::ptrdiff_t>(m_momentumX.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_momentumX.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_momentumX.cellCountZ());
    const double h = m_momentumX.spacing();
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                total += m_momentumX(i, j, k);
            }
        }
    }
    return total * h * h * h;
}

double EulerianFluid3D::totalMomentumY() const {
    const auto nx = static_cast<std::ptrdiff_t>(m_momentumY.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_momentumY.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_momentumY.cellCountZ());
    const double h = m_momentumY.spacing();
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                total += m_momentumY(i, j, k);
            }
        }
    }
    return total * h * h * h;
}

double EulerianFluid3D::totalMomentumZ() const {
    const auto nx = static_cast<std::ptrdiff_t>(m_momentumZ.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_momentumZ.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_momentumZ.cellCountZ());
    const double h = m_momentumZ.spacing();
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                total += m_momentumZ(i, j, k);
            }
        }
    }
    return total * h * h * h;
}

double EulerianFluid3D::totalEnergy() const {
    const auto nx = static_cast<std::ptrdiff_t>(m_energy.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_energy.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_energy.cellCountZ());
    const double h = m_energy.spacing();
    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                total += m_energy(i, j, k);
            }
        }
    }
    return total * h * h * h;
}

}  // namespace ysq
