#include <Compute/CPU/CpuBackend.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ysq {

namespace {

// Neumaier-compensated summation, the same idea as Statistics::sum in Math
// (not reused directly: Compute has no dependency on Math by design, so this
// is a few duplicated lines rather than an unwanted one). Accumulates as Acc
// while reading elements of type In, which is what lets the float-facing
// sum() still accumulate in double internally.
template <class Acc, class In>
Acc compensatedSum(std::span<const In> values) {
    Acc total{0};
    Acc compensation{0};
    for (const In element : values) {
        const Acc value = static_cast<Acc>(element);
        const Acc t = total + value;
        if (std::abs(total) >= std::abs(value)) {
            compensation += (total - t) + value;
        } else {
            compensation += (value - t) + total;
        }
        total = t;
    }
    return total + compensation;
}

}  // namespace

std::unique_ptr<ComputeBackend> CpuBackend::create() {
    return std::make_unique<CpuBackend>();
}

void CpuBackend::saxpy(std::span<const float> x, std::span<float> y, float a) const {
    assert(x.size() == y.size() && "saxpy needs matching spans");
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = a * x[i] + y[i];
    }
}

float CpuBackend::sum(std::span<const float> x) const {
    // float in, float out at the interface; accumulated in double internally,
    // because the reference has to be trustworthy and a naive float
    // accumulator loses far more over a long run than the double form costs.
    return static_cast<float>(compensatedSum<double>(x));
}

void CpuBackend::linearCombine(std::span<const std::span<const float>> terms,
                               std::span<const float> coefficients,
                               std::span<float> y) const {
    assert(terms.size() == coefficients.size() && "one coefficient per term");
    assert(terms.size() >= 1 && terms.size() <= 4 &&
           "linearCombine takes one to four terms");
    for (std::size_t k = 0; k < terms.size(); ++k) {
        assert(terms[k].size() == y.size() && "every term must match y's length");
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        // Accumulated in double for the same reason sum() does: the
        // reference has to be trustworthy, and this is the same weighted
        // reduction shape, just per-element instead of over the whole array.
        double total = 0.0;
        for (std::size_t k = 0; k < terms.size(); ++k) {
            total +=
                static_cast<double>(coefficients[k]) * static_cast<double>(terms[k][i]);
        }
        y[i] = static_cast<float>(total);
    }
}

void CpuBackend::gravitationalNBody(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> gm, float softeningSquared,
                                    std::span<float> accelerationsX,
                                    std::span<float> accelerationsY,
                                    std::span<float> accelerationsZ) const {
    const std::size_t n = positionsX.size();
    assert(positionsY.size() == n && positionsZ.size() == n && gm.size() == n);
    assert(accelerationsX.size() == n && accelerationsY.size() == n &&
           accelerationsZ.size() == n);

    for (std::size_t i = 0; i < n; ++i) {
        // Accumulated in double: this is the reference every GPU backend is
        // checked against, the same reasoning sum()'s internal accumulator
        // has, applied to a per-body reduction over up to n-1 terms instead
        // of one over the whole array.
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const double dx =
                static_cast<double>(positionsX[j]) - static_cast<double>(positionsX[i]);
            const double dy =
                static_cast<double>(positionsY[j]) - static_cast<double>(positionsY[i]);
            const double dz =
                static_cast<double>(positionsZ[j]) - static_cast<double>(positionsZ[i]);
            const double r2 =
                dx * dx + dy * dy + dz * dz + static_cast<double>(softeningSquared);
            const double invR3 = 1.0 / (r2 * std::sqrt(r2));
            const double factor = static_cast<double>(gm[j]) * invR3;
            ax += factor * dx;
            ay += factor * dy;
            az += factor * dz;
        }
        accelerationsX[i] = static_cast<float>(ax);
        accelerationsY[i] = static_cast<float>(ay);
        accelerationsZ[i] = static_cast<float>(az);
    }
}

void CpuBackend::electricFieldNBody(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> charge, float coulombConstant,
                                    std::span<float> fieldX, std::span<float> fieldY,
                                    std::span<float> fieldZ) const {
    const std::size_t n = positionsX.size();
    assert(positionsY.size() == n && positionsZ.size() == n && charge.size() == n);
    assert(fieldX.size() == n && fieldY.size() == n && fieldZ.size() == n);

    for (std::size_t i = 0; i < n; ++i) {
        double ex = 0.0;
        double ey = 0.0;
        double ez = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const double dx =
                static_cast<double>(positionsX[i]) - static_cast<double>(positionsX[j]);
            const double dy =
                static_cast<double>(positionsY[i]) - static_cast<double>(positionsY[j]);
            const double dz =
                static_cast<double>(positionsZ[i]) - static_cast<double>(positionsZ[j]);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 0.0) {
                continue;
            }
            const double invR3 = 1.0 / (r2 * std::sqrt(r2));
            const double factor = static_cast<double>(charge[j]) * invR3;
            ex += factor * dx;
            ey += factor * dy;
            ez += factor * dz;
        }
        const double scale = static_cast<double>(coulombConstant);
        fieldX[i] = static_cast<float>(scale * ex);
        fieldY[i] = static_cast<float>(scale * ey);
        fieldZ[i] = static_cast<float>(scale * ez);
    }
}

void CpuBackend::magneticFieldNBody(
    std::span<const float> positionsX, std::span<const float> positionsY,
    std::span<const float> positionsZ, std::span<const float> velocitiesX,
    std::span<const float> velocitiesY, std::span<const float> velocitiesZ,
    std::span<const float> charge, float permeabilityOver4Pi, std::span<float> fieldX,
    std::span<float> fieldY, std::span<float> fieldZ) const {
    const std::size_t n = positionsX.size();
    assert(positionsY.size() == n && positionsZ.size() == n);
    assert(velocitiesX.size() == n && velocitiesY.size() == n && velocitiesZ.size() == n);
    assert(charge.size() == n);
    assert(fieldX.size() == n && fieldY.size() == n && fieldZ.size() == n);

    for (std::size_t i = 0; i < n; ++i) {
        double bx = 0.0;
        double by = 0.0;
        double bz = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const double dx =
                static_cast<double>(positionsX[i]) - static_cast<double>(positionsX[j]);
            const double dy =
                static_cast<double>(positionsY[i]) - static_cast<double>(positionsY[j]);
            const double dz =
                static_cast<double>(positionsZ[i]) - static_cast<double>(positionsZ[j]);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 0.0) {
                continue;
            }
            const double invR3 = 1.0 / (r2 * std::sqrt(r2));
            const double factor = static_cast<double>(charge[j]) * invR3;
            const double vx = static_cast<double>(velocitiesX[j]);
            const double vy = static_cast<double>(velocitiesY[j]);
            const double vz = static_cast<double>(velocitiesZ[j]);
            // v x delta
            bx += factor * (vy * dz - vz * dy);
            by += factor * (vz * dx - vx * dz);
            bz += factor * (vx * dy - vy * dx);
        }
        const double scale = static_cast<double>(permeabilityOver4Pi);
        fieldX[i] = static_cast<float>(scale * bx);
        fieldY[i] = static_cast<float>(scale * by);
        fieldZ[i] = static_cast<float>(scale * bz);
    }
}

namespace {

// The cubic spline kernel (Monaghan & Lattanzio 1985), normalized for 3D;
// matches Physics/Fluids/SPH.hpp's cubicSplineKernel exactly. Kept as a
// free function here rather than shared with Physics, the same
// Compute-has-no-dependency-on-Physics reasoning every other kernel in
// this file already follows.
float cubicSplineKernel(float r, float smoothingLength) {
    constexpr float kNormalization3D = 0.3183098861837907f;  // 1 / pi
    const float q = r / smoothingLength;
    const float sigma =
        kNormalization3D / (smoothingLength * smoothingLength * smoothingLength);
    if (q < 1.0f) {
        return sigma * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    }
    if (q < 2.0f) {
        const float t = 2.0f - q;
        return sigma * 0.25f * t * t * t;
    }
    return 0.0f;
}

// dW/dr / r, so the caller multiplies by the raw separation vector rather
// than normalizing it separately; matches cubicSplineKernelGradient's
// direction * (dWdq / smoothingLength) / r.
float cubicSplineKernelGradientOverR(float r, float smoothingLength) {
    if (r <= 0.0f) {
        return 0.0f;
    }
    constexpr float kNormalization3D = 0.3183098861837907f;
    const float q = r / smoothingLength;
    const float sigma =
        kNormalization3D / (smoothingLength * smoothingLength * smoothingLength);
    float dWdq;
    if (q < 1.0f) {
        dWdq = sigma * (-3.0f * q + 2.25f * q * q);
    } else if (q < 2.0f) {
        const float t = 2.0f - q;
        dWdq = -0.75f * sigma * t * t;
    } else {
        dWdq = 0.0f;
    }
    return (dWdq / smoothingLength) / r;
}

}  // namespace

void CpuBackend::sphDensityPressure(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> mass, float smoothingLength,
                                    float equationOfStateK, float polytropicIndex,
                                    std::span<float> density,
                                    std::span<float> pressure) const {
    const std::size_t n = positionsX.size();
    const float supportRadius = 2.0f * smoothingLength;

    for (std::size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j) {
            const float dx = positionsX[i] - positionsX[j];
            const float dy = positionsY[i] - positionsY[j];
            const float dz = positionsZ[i] - positionsZ[j];
            const float r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r >= supportRadius) {
                continue;
            }
            sum += mass[j] * cubicSplineKernel(r, smoothingLength);
        }
        density[i] = sum;
        pressure[i] = equationOfStateK * std::pow(sum, polytropicIndex);
    }
}

void CpuBackend::sphPressureAcceleration(
    std::span<const float> positionsX, std::span<const float> positionsY,
    std::span<const float> positionsZ, std::span<const float> mass,
    std::span<const float> density, std::span<const float> pressure,
    float smoothingLength, std::span<float> accelerationsX,
    std::span<float> accelerationsY, std::span<float> accelerationsZ) const {
    const std::size_t n = positionsX.size();
    const float supportRadius = 2.0f * smoothingLength;

    for (std::size_t i = 0; i < n; ++i) {
        const float targetTerm = pressure[i] / (density[i] * density[i]);
        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const float dx = positionsX[i] - positionsX[j];
            const float dy = positionsY[i] - positionsY[j];
            const float dz = positionsZ[i] - positionsZ[j];
            const float r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r >= supportRadius) {
                continue;
            }
            const float sourceTerm = pressure[j] / (density[j] * density[j]);
            const float gradOverR = cubicSplineKernelGradientOverR(r, smoothingLength);
            const float factor = gradOverR * mass[j] * (targetTerm + sourceTerm);
            ax += factor * dx;
            ay += factor * dy;
            az += factor * dz;
        }
        accelerationsX[i] = -ax;
        accelerationsY[i] = -ay;
        accelerationsZ[i] = -az;
    }
}

void CpuBackend::heatEquation3DStep(std::span<const float> temperature, std::size_t nx,
                                    std::size_t ny, std::size_t nz, float factor,
                                    std::span<float> next) const {
    const auto idx = [ny, nz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * ny + j) * nz + k;
    };
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t xm = (i + nx - 1) % nx;
                const std::size_t xp = (i + 1) % nx;
                const std::size_t ym = (j + ny - 1) % ny;
                const std::size_t yp = (j + 1) % ny;
                const std::size_t zm = (k + nz - 1) % nz;
                const std::size_t zp = (k + 1) % nz;
                const float here = temperature[idx(i, j, k)];
                const float laplacian =
                    temperature[idx(xp, j, k)] + temperature[idx(xm, j, k)] +
                    temperature[idx(i, yp, k)] + temperature[idx(i, ym, k)] +
                    temperature[idx(i, j, zp)] + temperature[idx(i, j, zm)] - 6.0f * here;
                next[idx(i, j, k)] = here + factor * laplacian;
            }
        }
    }
}

void CpuBackend::acoustic3DStep(
    std::span<const float> pressure, std::span<const float> velocityX,
    std::span<const float> velocityY, std::span<const float> velocityZ, std::size_t nx,
    std::size_t ny, std::size_t nz, float velocityFactor, float pressureFactor,
    std::span<float> nextPressure, std::span<float> nextVelocityX,
    std::span<float> nextVelocityY, std::span<float> nextVelocityZ) const {
    const auto idx = [ny, nz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * ny + j) * nz + k;
    };

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t xp = (i + 1) % nx;
                const std::size_t yp = (j + 1) % ny;
                const std::size_t zp = (k + 1) % nz;
                const std::size_t here = idx(i, j, k);
                const float p = pressure[here];
                nextVelocityX[here] =
                    velocityX[here] - velocityFactor * (pressure[idx(xp, j, k)] - p);
                nextVelocityY[here] =
                    velocityY[here] - velocityFactor * (pressure[idx(i, yp, k)] - p);
                nextVelocityZ[here] =
                    velocityZ[here] - velocityFactor * (pressure[idx(i, j, zp)] - p);
            }
        }
    }

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t xm = (i + nx - 1) % nx;
                const std::size_t ym = (j + ny - 1) % ny;
                const std::size_t zm = (k + nz - 1) % nz;
                const std::size_t here = idx(i, j, k);
                const float divergence =
                    (nextVelocityX[here] - nextVelocityX[idx(xm, j, k)]) +
                    (nextVelocityY[here] - nextVelocityY[idx(i, ym, k)]) +
                    (nextVelocityZ[here] - nextVelocityZ[idx(i, j, zm)]);
                nextPressure[here] = pressure[here] - pressureFactor * divergence;
            }
        }
    }
}

void CpuBackend::maxwell3DStep(std::span<const float> ex, std::span<const float> ey,
                               std::span<const float> ez, std::span<const float> bx,
                               std::span<const float> by, std::span<const float> bz,
                               std::size_t nx, std::size_t ny, std::size_t nz,
                               float bFactor, float eFactor, std::span<float> nextEx,
                               std::span<float> nextEy, std::span<float> nextEz,
                               std::span<float> nextBx, std::span<float> nextBy,
                               std::span<float> nextBz) const {
    const auto idx = [ny, nz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * ny + j) * nz + k;
    };

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t ip = (i + 1) % nx;
                const std::size_t jp = (j + 1) % ny;
                const std::size_t kp = (k + 1) % nz;
                const std::size_t here = idx(i, j, k);
                nextBx[here] = bx[here] - bFactor * ((ez[idx(i, jp, k)] - ez[here]) -
                                                     (ey[idx(i, j, kp)] - ey[here]));
                nextBy[here] = by[here] - bFactor * ((ex[idx(i, j, kp)] - ex[here]) -
                                                     (ez[idx(ip, j, k)] - ez[here]));
                nextBz[here] = bz[here] - bFactor * ((ey[idx(ip, j, k)] - ey[here]) -
                                                     (ex[idx(i, jp, k)] - ex[here]));
            }
        }
    }

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t im = (i + nx - 1) % nx;
                const std::size_t jm = (j + ny - 1) % ny;
                const std::size_t km = (k + nz - 1) % nz;
                const std::size_t here = idx(i, j, k);
                nextEx[here] =
                    ex[here] + eFactor * ((nextBz[here] - nextBz[idx(i, jm, k)]) -
                                          (nextBy[here] - nextBy[idx(i, j, km)]));
                nextEy[here] =
                    ey[here] + eFactor * ((nextBx[here] - nextBx[idx(i, j, km)]) -
                                          (nextBz[here] - nextBz[idx(im, j, k)]));
                nextEz[here] =
                    ez[here] + eFactor * ((nextBy[here] - nextBy[idx(im, j, k)]) -
                                          (nextBx[here] - nextBx[idx(i, jm, k)]));
            }
        }
    }
}

namespace {

// The compressible Euler equations' gas-dynamics relations, float
// throughout to match the GPU kernels bit-for-bit in shape (not reused from
// Physics/Fluids/Eulerian3D.cpp's own double-precision copies: Compute has
// no dependency on Physics by design, the same duplicated-deliberately
// convention cubicSplineKernel above uses for SPH).
struct EulerState {
    float density;
    float momentumNormal;
    float momentumTangent1;
    float momentumTangent2;
    float energy;
};

struct EulerFlux {
    float density;
    float momentumNormal;
    float momentumTangent1;
    float momentumTangent2;
    float energy;
};

float eulerPressureOf(const EulerState& state, float gamma) {
    const float u = state.momentumNormal / state.density;
    const float v = state.momentumTangent1 / state.density;
    const float w = state.momentumTangent2 / state.density;
    const float kinetic = 0.5f * state.density * (u * u + v * v + w * w);
    return (gamma - 1.0f) * (state.energy - kinetic);
}

float eulerSoundSpeedOf(float density, float pressure, float gamma) {
    return std::sqrt(gamma * pressure / density);
}

EulerFlux eulerFluxOf(const EulerState& state, float gamma) {
    const float u = state.momentumNormal / state.density;
    const float v = state.momentumTangent1 / state.density;
    const float w = state.momentumTangent2 / state.density;
    const float pressure = eulerPressureOf(state, gamma);
    return {state.momentumNormal, state.momentumNormal * u + pressure,
            state.momentumNormal * v, state.momentumNormal * w,
            u * (state.energy + pressure)};
}

EulerFlux eulerRusanovFlux(const EulerState& left, const EulerState& right, float gamma) {
    const EulerFlux fluxLeft = eulerFluxOf(left, gamma);
    const EulerFlux fluxRight = eulerFluxOf(right, gamma);

    const float velocityLeft = left.momentumNormal / left.density;
    const float velocityRight = right.momentumNormal / right.density;
    const float soundLeft =
        eulerSoundSpeedOf(left.density, eulerPressureOf(left, gamma), gamma);
    const float soundRight =
        eulerSoundSpeedOf(right.density, eulerPressureOf(right, gamma), gamma);
    const float maxSpeed = std::max(std::abs(velocityLeft) + soundLeft,
                                    std::abs(velocityRight) + soundRight);

    return {0.5f * (fluxLeft.density + fluxRight.density) -
                0.5f * maxSpeed * (right.density - left.density),
            0.5f * (fluxLeft.momentumNormal + fluxRight.momentumNormal) -
                0.5f * maxSpeed * (right.momentumNormal - left.momentumNormal),
            0.5f * (fluxLeft.momentumTangent1 + fluxRight.momentumTangent1) -
                0.5f * maxSpeed * (right.momentumTangent1 - left.momentumTangent1),
            0.5f * (fluxLeft.momentumTangent2 + fluxRight.momentumTangent2) -
                0.5f * maxSpeed * (right.momentumTangent2 - left.momentumTangent2),
            0.5f * (fluxLeft.energy + fluxRight.energy) -
                0.5f * maxSpeed * (right.energy - left.energy)};
}

}  // namespace

void CpuBackend::eulerianFluid3DSweep(
    std::span<const float> density, std::span<const float> momentumNormal,
    std::span<const float> momentumTangent1, std::span<const float> momentumTangent2,
    std::span<const float> energy, std::size_t nx, std::size_t ny, std::size_t nz,
    int axis, float gamma, float dtOverSpacing, std::span<float> nextDensity,
    std::span<float> nextMomentumNormal, std::span<float> nextMomentumTangent1,
    std::span<float> nextMomentumTangent2, std::span<float> nextEnergy) const {
    const auto idx = [ny, nz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * ny + j) * nz + k;
    };
    const auto stateAt = [&](std::size_t p) {
        return EulerState{density[p], momentumNormal[p], momentumTangent1[p],
                          momentumTangent2[p], energy[p]};
    };

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                std::size_t im = i, ip = i, jm = j, jp = j, km = k, kp = k;
                if (axis == 0) {
                    im = (i + nx - 1) % nx;
                    ip = (i + 1) % nx;
                } else if (axis == 1) {
                    jm = (j + ny - 1) % ny;
                    jp = (j + 1) % ny;
                } else {
                    km = (k + nz - 1) % nz;
                    kp = (k + 1) % nz;
                }

                const std::size_t here = idx(i, j, k);
                const EulerFlux fluxLeft =
                    eulerRusanovFlux(stateAt(idx(im, jm, km)), stateAt(here), gamma);
                const EulerFlux fluxRight =
                    eulerRusanovFlux(stateAt(here), stateAt(idx(ip, jp, kp)), gamma);

                nextDensity[here] = density[here] - dtOverSpacing * (fluxRight.density -
                                                                     fluxLeft.density);
                nextMomentumNormal[here] =
                    momentumNormal[here] -
                    dtOverSpacing * (fluxRight.momentumNormal - fluxLeft.momentumNormal);
                nextMomentumTangent1[here] =
                    momentumTangent1[here] - dtOverSpacing * (fluxRight.momentumTangent1 -
                                                              fluxLeft.momentumTangent1);
                nextMomentumTangent2[here] =
                    momentumTangent2[here] - dtOverSpacing * (fluxRight.momentumTangent2 -
                                                              fluxLeft.momentumTangent2);
                nextEnergy[here] =
                    energy[here] - dtOverSpacing * (fluxRight.energy - fluxLeft.energy);
            }
        }
    }
}

namespace {

// 2*pi, duplicated rather than taken from Math::kTau since Compute has no
// dependency on Math by design (see the file-level comment on
// EulerState/EulerFlux above for the same convention).
constexpr float kTauFloat = 6.283185307179586f;

// The number of bits needed to index length-1 (length itself a power of
// two): log2(length), computed by counting rather than assuming a
// standard-library bit-counting function, to keep this file's only
// dependency std::span/cmath/cassert.
std::size_t log2OfPowerOfTwo(std::size_t length) {
    std::size_t bits = 0;
    while ((std::size_t{1} << bits) < length) {
        ++bits;
    }
    return bits;
}

std::size_t reverseBits(std::size_t value, std::size_t bitCount) {
    std::size_t result = 0;
    for (std::size_t b = 0; b < bitCount; ++b) {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    return result;
}

}  // namespace

void CpuBackend::fftBatched(std::span<const float> real, std::span<const float> imag,
                            std::size_t length, std::size_t batchCount, bool inverse,
                            std::span<float> nextReal, std::span<float> nextImag) const {
    if (length == 0 || batchCount == 0) {
        return;
    }
    const std::size_t bitCount = log2OfPowerOfTwo(length);

    // Bit-reversal permutation, out of place: unlike Math/FFT.hpp's
    // in-place version (which only needs to swap when i < j), a separate
    // output buffer lets every index write its own destination directly.
    std::vector<float> bufferReal(length * batchCount);
    std::vector<float> bufferImag(length * batchCount);
    for (std::size_t batch = 0; batch < batchCount; ++batch) {
        for (std::size_t i = 0; i < length; ++i) {
            const std::size_t reversed = reverseBits(i, bitCount);
            bufferReal[batch * length + reversed] = real[batch * length + i];
            bufferImag[batch * length + reversed] = imag[batch * length + i];
        }
    }

    // One butterfly pass per stage, ping-ponging between bufferX and
    // scratchX so each stage reads the previous stage's full output.
    // Matches Math/FFT.hpp's detail::fftImpl stage loop exactly, per batch.
    std::vector<float> scratchReal(length * batchCount);
    std::vector<float> scratchImag(length * batchCount);
    std::vector<float>* currentReal = &bufferReal;
    std::vector<float>* currentImag = &bufferImag;
    std::vector<float>* nextStageReal = &scratchReal;
    std::vector<float>* nextStageImag = &scratchImag;

    for (std::size_t len = 2; len <= length; len <<= 1) {
        const auto angle = (inverse ? 1.0f : -1.0f) * kTauFloat / static_cast<float>(len);
        const std::size_t half = len / 2;
        for (std::size_t batch = 0; batch < batchCount; ++batch) {
            const std::size_t base = batch * length;
            for (std::size_t blockStart = 0; blockStart < length; blockStart += len) {
                for (std::size_t k = 0; k < half; ++k) {
                    const std::size_t i0 = base + blockStart + k;
                    const std::size_t i1 = i0 + half;
                    const float twiddleReal = std::cos(static_cast<float>(k) * angle);
                    const float twiddleImag = std::sin(static_cast<float>(k) * angle);
                    const float evenReal = (*currentReal)[i0];
                    const float evenImag = (*currentImag)[i0];
                    const float oddReal0 = (*currentReal)[i1];
                    const float oddImag0 = (*currentImag)[i1];
                    const float oddReal = oddReal0 * twiddleReal - oddImag0 * twiddleImag;
                    const float oddImag = oddReal0 * twiddleImag + oddImag0 * twiddleReal;
                    (*nextStageReal)[i0] = evenReal + oddReal;
                    (*nextStageImag)[i0] = evenImag + oddImag;
                    (*nextStageReal)[i1] = evenReal - oddReal;
                    (*nextStageImag)[i1] = evenImag - oddImag;
                }
            }
        }
        std::swap(currentReal, nextStageReal);
        std::swap(currentImag, nextStageImag);
    }

    const float scale = inverse ? 1.0f / static_cast<float>(length) : 1.0f;
    for (std::size_t i = 0; i < length * batchCount; ++i) {
        nextReal[i] = (*currentReal)[i] * scale;
        nextImag[i] = (*currentImag)[i] * scale;
    }
}

void CpuBackend::matVec(std::span<const float> matrix, std::size_t rows, std::size_t cols,
                        std::span<const float> vector, std::span<float> result) const {
    for (std::size_t r = 0; r < rows; ++r) {
        float total = 0.0f;
        for (std::size_t c = 0; c < cols; ++c) {
            total += matrix[r * cols + c] * vector[c];
        }
        result[r] = total;
    }
}

void CpuBackend::matMul(std::span<const float> a, std::size_t aRows, std::size_t aCols,
                        std::span<const float> b, std::size_t bCols,
                        std::span<float> result) const {
    for (std::size_t r = 0; r < aRows; ++r) {
        for (std::size_t c = 0; c < bCols; ++c) {
            float total = 0.0f;
            for (std::size_t k = 0; k < aCols; ++k) {
                total += a[r * aCols + k] * b[k * bCols + c];
            }
            result[r * bCols + c] = total;
        }
    }
}

bool CpuBackend::luDecomposeGpu(std::span<const float> matrix, std::size_t n,
                                std::span<float> lu,
                                std::span<std::uint32_t> pivot) const {
    for (std::size_t i = 0; i < n * n; ++i) {
        lu[i] = matrix[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        pivot[i] = static_cast<std::uint32_t>(i);
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivotRow = k;
        float best = std::abs(lu[k * n + k]);
        for (std::size_t row = k + 1; row < n; ++row) {
            const float candidate = std::abs(lu[row * n + k]);
            if (best < candidate) {
                best = candidate;
                pivotRow = row;
            }
        }

        if (!(0.0f < best) || !std::isfinite(best)) {
            return false;
        }

        if (pivotRow != k) {
            for (std::size_t col = 0; col < n; ++col) {
                std::swap(lu[k * n + col], lu[pivotRow * n + col]);
            }
            std::swap(pivot[k], pivot[pivotRow]);
        }

        for (std::size_t row = k + 1; row < n; ++row) {
            const float factor = lu[row * n + k] / lu[k * n + k];
            lu[row * n + k] = factor;
            for (std::size_t col = k + 1; col < n; ++col) {
                lu[row * n + col] -= factor * lu[k * n + col];
            }
        }
    }
    return true;
}

bool CpuBackend::choleskyDecomposeGpu(std::span<const float> a, std::size_t n,
                                      std::span<float> l) const {
    for (float& value : l) {
        value = 0.0f;
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            float total = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                total -= l[i * n + k] * l[j * n + k];
            }

            if (i == j) {
                if (!(0.0f < total) || !std::isfinite(total)) {
                    return false;
                }
                l[i * n + j] = std::sqrt(total);
            } else {
                l[i * n + j] = total / l[j * n + j];
            }
        }
    }
    return true;
}

void CpuBackend::qrDecomposeGpu(std::span<const float> matrix, std::size_t rows,
                                std::size_t cols, std::span<float> q,
                                std::span<float> r) const {
    for (std::size_t i = 0; i < rows * cols; ++i) {
        r[i] = matrix[i];
    }
    for (std::size_t i = 0; i < rows * rows; ++i) {
        q[i] = 0.0f;
    }
    for (std::size_t i = 0; i < rows; ++i) {
        q[i * rows + i] = 1.0f;
    }

    const std::size_t steps = std::min(rows, cols);
    std::vector<float> v(rows);

    for (std::size_t k = 0; k < steps; ++k) {
        float normX = 0.0f;
        for (std::size_t i = k; i < rows; ++i) {
            normX += r[i * cols + k] * r[i * cols + k];
        }
        normX = std::sqrt(normX);
        if (normX == 0.0f) {
            continue;
        }

        const float alpha = (r[k * cols + k] < 0.0f) ? normX : -normX;
        for (std::size_t i = k; i < rows; ++i) {
            v[i - k] = r[i * cols + k];
        }
        v[0] -= alpha;

        float vNormSquared = 0.0f;
        for (std::size_t i = 0; i < rows - k; ++i) {
            vNormSquared += v[i] * v[i];
        }
        if (vNormSquared == 0.0f) {
            continue;
        }

        for (std::size_t col = k; col < cols; ++col) {
            float dotProduct = 0.0f;
            for (std::size_t i = k; i < rows; ++i) {
                dotProduct += v[i - k] * r[i * cols + col];
            }
            const float factor = 2.0f * dotProduct / vNormSquared;
            for (std::size_t i = k; i < rows; ++i) {
                r[i * cols + col] -= factor * v[i - k];
            }
        }

        for (std::size_t row = 0; row < rows; ++row) {
            float dotProduct = 0.0f;
            for (std::size_t i = k; i < rows; ++i) {
                dotProduct += q[row * rows + i] * v[i - k];
            }
            const float factor = 2.0f * dotProduct / vNormSquared;
            for (std::size_t i = k; i < rows; ++i) {
                q[row * rows + i] -= factor * v[i - k];
            }
        }
    }
}

void CpuBackend::jacobiEigenSymmetricGpu(std::span<const float> matrix, std::size_t n,
                                         int maxSweeps, float tolerance,
                                         std::span<float> resultDiagonal,
                                         std::span<float> resultEigenvectors) const {
    // The reference implementation uses the same simple cyclic order
    // Math/Eigen.hpp's own CPU jacobiEigenSymmetric does (not the
    // round-robin scheme the GPU backends use internally): both converge
    // to the same diagonalization, and the cyclic order is the one this
    // engine's own CPU tests already validate independently.
    for (std::size_t i = 0; i < n * n; ++i) {
        resultDiagonal[i] = matrix[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            resultDiagonal[j * n + i] = resultDiagonal[i * n + j];
        }
    }
    for (float& value : resultEigenvectors) {
        value = 0.0f;
    }
    for (std::size_t i = 0; i < n; ++i) {
        resultEigenvectors[i * n + i] = 1.0f;
    }

    auto& a = resultDiagonal;
    auto& v = resultEigenvectors;
    const float effectiveTolerance =
        (tolerance > 0.0f) ? tolerance : std::numeric_limits<float>::epsilon() * 100.0f;

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        float offDiagonalSquared = 0.0f;
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                offDiagonalSquared += a[p * n + q] * a[p * n + q];
            }
        }
        if (offDiagonalSquared < effectiveTolerance * effectiveTolerance) {
            break;
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const float apq = a[p * n + q];
                if (std::abs(apq) <= std::numeric_limits<float>::epsilon()) {
                    continue;
                }

                const float app = a[p * n + p];
                const float aqq = a[q * n + q];
                const float theta = (aqq - app) / (2.0f * apq);
                const float sign = (theta < 0.0f) ? -1.0f : 1.0f;
                const float t =
                    sign / (std::abs(theta) + std::sqrt(theta * theta + 1.0f));
                const float c = 1.0f / std::sqrt(t * t + 1.0f);
                const float s = t * c;

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0f;
                a[q * n + p] = 0.0f;

                for (std::size_t i = 0; i < n; ++i) {
                    if (i == p || i == q) {
                        continue;
                    }
                    const float aip = a[i * n + p];
                    const float aiq = a[i * n + q];
                    a[i * n + p] = c * aip - s * aiq;
                    a[p * n + i] = a[i * n + p];
                    a[i * n + q] = s * aip + c * aiq;
                    a[q * n + i] = a[i * n + q];
                }

                for (std::size_t i = 0; i < n; ++i) {
                    const float vip = v[i * n + p];
                    const float viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
}

void CpuBackend::jacobiSvdGpu(std::span<const float> matrix, std::size_t rows,
                              std::size_t cols, int maxSweeps, float tolerance,
                              std::span<float> resultA, std::span<float> resultV) const {
    // Same cyclic-order-for-the-reference rationale as
    // jacobiEigenSymmetricGpu above.
    for (std::size_t i = 0; i < rows * cols; ++i) {
        resultA[i] = matrix[i];
    }
    for (float& value : resultV) {
        value = 0.0f;
    }
    for (std::size_t i = 0; i < cols; ++i) {
        resultV[i * cols + i] = 1.0f;
    }

    auto& a = resultA;
    auto& v = resultV;
    const float effectiveTolerance =
        (tolerance > 0.0f) ? tolerance : std::numeric_limits<float>::epsilon() * 100.0f;

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        float offDiagonalSquared = 0.0f;
        for (std::size_t p = 0; p < cols; ++p) {
            for (std::size_t q = p + 1; q < cols; ++q) {
                float gamma = 0.0f;
                for (std::size_t i = 0; i < rows; ++i) {
                    gamma += a[i * cols + p] * a[i * cols + q];
                }
                offDiagonalSquared += gamma * gamma;
            }
        }
        if (offDiagonalSquared < effectiveTolerance * effectiveTolerance) {
            break;
        }

        for (std::size_t p = 0; p < cols; ++p) {
            for (std::size_t q = p + 1; q < cols; ++q) {
                float alpha = 0.0f;
                float beta = 0.0f;
                float gamma = 0.0f;
                for (std::size_t i = 0; i < rows; ++i) {
                    alpha += a[i * cols + p] * a[i * cols + p];
                    beta += a[i * cols + q] * a[i * cols + q];
                    gamma += a[i * cols + p] * a[i * cols + q];
                }

                const float threshold = effectiveTolerance * std::sqrt(alpha * beta);
                if (std::abs(gamma) <= threshold) {
                    continue;
                }

                const float zeta = (beta - alpha) / (2.0f * gamma);
                const float sign = (zeta < 0.0f) ? -1.0f : 1.0f;
                const float t = sign / (std::abs(zeta) + std::sqrt(1.0f + zeta * zeta));
                const float c = 1.0f / std::sqrt(1.0f + t * t);
                const float s = c * t;

                for (std::size_t i = 0; i < rows; ++i) {
                    const float ap = a[i * cols + p];
                    const float aq = a[i * cols + q];
                    a[i * cols + p] = c * ap - s * aq;
                    a[i * cols + q] = s * ap + c * aq;
                }
                for (std::size_t i = 0; i < cols; ++i) {
                    const float vp = v[i * cols + p];
                    const float vq = v[i * cols + q];
                    v[i * cols + p] = c * vp - s * vq;
                    v[i * cols + q] = s * vp + c * vq;
                }
            }
        }
    }
}

void CpuBackend::batchErf(std::span<const float> x, std::span<float> result) const {
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = std::erf(x[i]);
    }
}

void CpuBackend::batchErfc(std::span<const float> x, std::span<float> result) const {
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = std::erfc(x[i]);
    }
}

void CpuBackend::batchGamma(std::span<const float> x, std::span<float> result) const {
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = std::tgamma(x[i]);
    }
}

void CpuBackend::batchLogGamma(std::span<const float> x, std::span<float> result) const {
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = std::lgamma(x[i]);
    }
}

void CpuBackend::batchLegendreP(unsigned n, unsigned m, std::span<const float> x,
                                std::span<float> result) const {
    for (std::size_t idx = 0; idx < x.size(); ++idx) {
        const float value = x[idx];
        float pmm = 1.0f;
        if (m > 0) {
            const float oneMinusX2 = (1.0f - value) * (1.0f + value);
            const float somx2 = std::sqrt(oneMinusX2);
            float fact = 1.0f;
            for (unsigned i = 1; i <= m; ++i) {
                pmm *= -fact * somx2;
                fact += 2.0f;
            }
        }
        if (n == m) {
            result[idx] = pmm;
            continue;
        }

        float pmmp1 = value * static_cast<float>(2 * m + 1) * pmm;
        if (n == m + 1) {
            result[idx] = pmmp1;
            continue;
        }

        float pll = 0.0f;
        for (unsigned ll = m + 2; ll <= n; ++ll) {
            pll = (value * static_cast<float>(2 * ll - 1) * pmmp1 -
                   static_cast<float>(ll + m - 1) * pmm) /
                  static_cast<float>(ll - m);
            pmm = pmmp1;
            pmmp1 = pll;
        }
        result[idx] = pll;
    }
}

void CpuBackend::batchPolynomialEval(std::span<const float> coefficients,
                                     std::span<const float> x,
                                     std::span<float> result) const {
    for (std::size_t idx = 0; idx < x.size(); ++idx) {
        float value = coefficients.back();
        for (std::size_t i = coefficients.size() - 1; i-- > 0;) {
            value = value * x[idx] + coefficients[i];
        }
        result[idx] = value;
    }
}

void CpuBackend::batchCubicSplineEval(std::span<const float> knotsX,
                                      std::span<const float> knotsY,
                                      std::span<const float> secondDerivatives,
                                      std::span<const float> queryX,
                                      std::span<float> result) const {
    for (std::size_t idx = 0; idx < queryX.size(); ++idx) {
        const float at = queryX[idx];
        if (at <= knotsX.front()) {
            result[idx] = knotsY.front();
            continue;
        }
        if (knotsX.back() <= at) {
            result[idx] = knotsY.back();
            continue;
        }

        const auto upper = std::upper_bound(knotsX.begin(), knotsX.end(), at);
        auto i = static_cast<std::size_t>(upper - knotsX.begin());
        i = (i == 0) ? 0 : i - 1;
        i = std::min(i, knotsX.size() - 2);

        const float width = knotsX[i + 1] - knotsX[i];
        const float left = (knotsX[i + 1] - at) / width;
        const float right = (at - knotsX[i]) / width;

        result[idx] = left * knotsY[i] + right * knotsY[i + 1] +
                      ((left * left * left - left) * secondDerivatives[i] +
                       (right * right * right - right) * secondDerivatives[i + 1]) *
                          (width * width) / 6.0f;
    }
}

namespace {

// Philox4x32-10 (Salmon, Moraes, Hadjidoukas & Schulten 2011): counter and
// key each split into two 32-bit words, ten rounds of a Feistel-style
// multiply-and-permute. Pure bit arithmetic, no floating-point cancellation
// possible, so unlike the batch-evaluation family this needs no float32
// suitability analysis at all -- the same algorithm at the same precision
// runs correctly in double, float, or a GPU shader alike.
void mulhilo32(std::uint32_t a, std::uint32_t b, std::uint32_t& hi, std::uint32_t& lo) {
    const std::uint64_t product =
        static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    hi = static_cast<std::uint32_t>(product >> 32);
    lo = static_cast<std::uint32_t>(product);
}

std::array<std::uint32_t, 4> philox4x32_10(std::uint32_t c0, std::uint32_t c1,
                                           std::uint32_t c2, std::uint32_t c3,
                                           std::uint32_t k0, std::uint32_t k1) {
    constexpr std::uint32_t kM0 = 0xD2511F53u;
    constexpr std::uint32_t kM1 = 0xCD9E8D57u;
    constexpr std::uint32_t kW0 = 0x9E3779B9u;
    constexpr std::uint32_t kW1 = 0xBB67AE85u;
    for (int round = 0; round < 10; ++round) {
        std::uint32_t hi0{}, lo0{}, hi1{}, lo1{};
        mulhilo32(kM0, c0, hi0, lo0);
        mulhilo32(kM1, c2, hi1, lo1);
        const std::uint32_t nextC0 = hi1 ^ c1 ^ k0;
        const std::uint32_t nextC1 = lo1;
        const std::uint32_t nextC2 = hi0 ^ c3 ^ k1;
        const std::uint32_t nextC3 = lo0;
        c0 = nextC0;
        c1 = nextC1;
        c2 = nextC2;
        c3 = nextC3;
        k0 += kW0;
        k1 += kW1;
    }
    return {c0, c1, c2, c3};
}

constexpr float kOneOverTwoToThe32 = 2.3283064365386963e-10f;
constexpr float kTwoPiF = 6.28318530717958647692f;

}  // namespace

void CpuBackend::batchUniformReal(std::uint64_t seed, std::uint64_t offset,
                                  std::span<float> result) const {
    const auto k0 = static_cast<std::uint32_t>(seed);
    const auto k1 = static_cast<std::uint32_t>(seed >> 32);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const std::uint64_t counter = offset + i;
        const auto c0 = static_cast<std::uint32_t>(counter);
        const auto c1 = static_cast<std::uint32_t>(counter >> 32);
        const std::array<std::uint32_t, 4> out = philox4x32_10(c0, c1, 0u, 0u, k0, k1);
        result[i] = static_cast<float>(out[0]) * kOneOverTwoToThe32;
    }
}

void CpuBackend::batchNormal(std::uint64_t seed, std::uint64_t offset,
                             std::span<float> result) const {
    const auto k0 = static_cast<std::uint32_t>(seed);
    const auto k1 = static_cast<std::uint32_t>(seed >> 32);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const std::uint64_t counter = offset + i;
        const auto c0 = static_cast<std::uint32_t>(counter);
        const auto c1 = static_cast<std::uint32_t>(counter >> 32);
        const std::array<std::uint32_t, 4> out = philox4x32_10(c0, c1, 0u, 0u, k0, k1);
        // u1 in (0, 1], never exactly 0, so log(u1) below is always finite:
        // offsetting the 32-bit word up by one before scaling avoids a
        // special case for the all-zero output word instead.
        const float u1 = (static_cast<float>(out[0]) + 1.0f) * kOneOverTwoToThe32;
        const float u2 = static_cast<float>(out[1]) * kOneOverTwoToThe32;
        result[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(kTwoPiF * u2);
    }
}

void CpuBackend::sortAscending(std::span<float> values) const {
    std::sort(values.begin(), values.end());
}

void CpuBackend::multigridRestrict3D(std::span<const float> fine, std::size_t nx,
                                     std::size_t ny, std::size_t nz,
                                     std::span<float> coarse) const {
    const std::size_t cnx = nx / 2;
    const std::size_t cny = ny / 2;
    const std::size_t cnz = nz / 2;
    const auto fineIdx = [ny, nz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * ny + j) * nz + k;
    };
    const auto coarseIdx = [cny, cnz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * cny + j) * cnz + k;
    };

    for (std::size_t i = 0; i < cnx; ++i) {
        for (std::size_t j = 0; j < cny; ++j) {
            for (std::size_t k = 0; k < cnz; ++k) {
                double sum = 0.0;
                for (std::size_t di = 0; di < 2; ++di) {
                    for (std::size_t dj = 0; dj < 2; ++dj) {
                        for (std::size_t dk = 0; dk < 2; ++dk) {
                            sum += static_cast<double>(
                                fine[fineIdx(2 * i + di, 2 * j + dj, 2 * k + dk)]);
                        }
                    }
                }
                coarse[coarseIdx(i, j, k)] = static_cast<float>(sum / 8.0);
            }
        }
    }
}

void CpuBackend::multigridProlongateAndAdd3D(std::span<const float> fine,
                                             std::span<const float> coarseCorrection,
                                             std::size_t nx, std::size_t ny,
                                             std::size_t nz,
                                             std::span<float> nextFine) const {
    const std::size_t cny = ny / 2;
    const std::size_t cnz = nz / 2;
    const auto fineIdx = [ny, nz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * ny + j) * nz + k;
    };
    const auto coarseIdx = [cny, cnz](std::size_t i, std::size_t j, std::size_t k) {
        return (i * cny + j) * cnz + k;
    };

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t here = fineIdx(i, j, k);
                nextFine[here] =
                    fine[here] + coarseCorrection[coarseIdx(i / 2, j / 2, k / 2)];
            }
        }
    }
}

std::size_t CpuBackend::minIndex(std::span<const float> x) const {
    if (x.empty()) {
        return x.size();
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] < x[best]) {
            best = i;
        }
    }
    return best;
}

void CpuBackend::saxpyD(std::span<const double> x, std::span<double> y, double a) const {
    assert(x.size() == y.size() && "saxpy needs matching spans");
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = a * x[i] + y[i];
    }
}

double CpuBackend::sumD(std::span<const double> x) const {
    return compensatedSum<double>(x);
}

}  // namespace ysq
