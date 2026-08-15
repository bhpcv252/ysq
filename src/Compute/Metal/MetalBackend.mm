#include <Compute/Metal/MetalBackend.hpp>

#include <Core/Logger.hpp>

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace ysq {

namespace {

constexpr NSUInteger kThreadgroupSize = 256;

// All three reference kernels in one library, compiled from source at
// runtime rather than a precompiled .metallib: there is no shader-embedding
// build step in this project yet (see src/Compute/README.md's OpenGL
// backend section, which made the same call for GLSL), and three kernels
// this small do not earn one. saxpy and sum mirror the OpenGL backend's own
// kernels exactly (same bounds-check convention, same two-pass tree
// reduction for sum); linear_combine is new, see ComputeBackend.hpp.
constexpr const char* kKernelSourcePart1 = R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void saxpy_kernel(device const float* x [[buffer(0)]],
                         device float* y [[buffer(1)]],
                         constant float& a [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    y[i] = a * x[i] + y[i];
}

kernel void sum_kernel(device const float* data [[buffer(0)]],
                       device float* partials [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint i [[thread_position_in_grid]],
                       uint local [[thread_position_in_threadgroup]],
                       uint groupId [[threadgroup_position_in_grid]],
                       threadgroup float* scratch [[threadgroup(0)]]) {
    scratch[local] = (i < count) ? data[i] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local < stride) {
            scratch[local] += scratch[local + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local == 0) {
        partials[groupId] = scratch[0];
    }
}

kernel void linear_combine_kernel(device const float* term0 [[buffer(0)]],
                                  device const float* term1 [[buffer(1)]],
                                  device const float* term2 [[buffer(2)]],
                                  device const float* term3 [[buffer(3)]],
                                  device float* y [[buffer(4)]],
                                  constant uint& count [[buffer(5)]],
                                  constant int& termCount [[buffer(6)]],
                                  constant float4& coefficients [[buffer(7)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float total = 0.0f;
    if (termCount > 0) total += coefficients.x * term0[i];
    if (termCount > 1) total += coefficients.y * term1[i];
    if (termCount > 2) total += coefficients.z * term2[i];
    if (termCount > 3) total += coefficients.w * term3[i];
    y[i] = total;
}

// Direct-sum (O(n^2)) Newtonian gravity, Plummer-softened: one thread per
// target body, looping over every other body. Matches
// Physics/Gravity/Newtonian.cpp's point-mass term exactly (no J2: see
// ComputeBackend.hpp on why that needs a different, pairwise shape).
kernel void gravitational_nbody_kernel(device const float* posX [[buffer(0)]],
                                       device const float* posY [[buffer(1)]],
                                       device const float* posZ [[buffer(2)]],
                                       device const float* gm [[buffer(3)]],
                                       device float* accX [[buffer(4)]],
                                       device float* accY [[buffer(5)]],
                                       device float* accZ [[buffer(6)]],
                                       constant uint& count [[buffer(7)]],
                                       constant float& softeningSquared [[buffer(8)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    for (uint j = 0; j < count; ++j) {
        if (j == i) continue;
        float dx = posX[j] - px;
        float dy = posY[j] - py;
        float dz = posZ[j] - pz;
        float r2 = dx * dx + dy * dy + dz * dz + softeningSquared;
        float invR3 = 1.0f / (r2 * sqrt(r2));
        float factor = gm[j] * invR3;
        ax += factor * dx;
        ay += factor * dy;
        az += factor * dz;
    }
    accX[i] = ax;
    accY[i] = ay;
    accZ[i] = az;
}

// Direct-sum Coulomb field, one thread per query charge. Matches
// Physics/Electromagnetism/Field.cpp's electricField exactly: unsoftened,
// skipping a source at zero separation.
kernel void electric_field_nbody_kernel(device const float* posX [[buffer(0)]],
                                        device const float* posY [[buffer(1)]],
                                        device const float* posZ [[buffer(2)]],
                                        device const float* charge [[buffer(3)]],
                                        device float* fieldX [[buffer(4)]],
                                        device float* fieldY [[buffer(5)]],
                                        device float* fieldZ [[buffer(6)]],
                                        constant uint& count [[buffer(7)]],
                                        constant float& coulombConstant [[buffer(8)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float ex = 0.0f;
    float ey = 0.0f;
    float ez = 0.0f;
    for (uint j = 0; j < count; ++j) {
        if (j == i) continue;
        float dx = px - posX[j];
        float dy = py - posY[j];
        float dz = pz - posZ[j];
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= 0.0f) continue;
        float invR3 = 1.0f / (r2 * sqrt(r2));
        float factor = charge[j] * invR3;
        ex += factor * dx;
        ey += factor * dy;
        ez += factor * dz;
    }
    fieldX[i] = coulombConstant * ex;
    fieldY[i] = coulombConstant * ey;
    fieldZ[i] = coulombConstant * ez;
}

// Point-charge Biot-Savart. Matches Field.cpp's magneticField exactly.
kernel void magnetic_field_nbody_kernel(device const float* posX [[buffer(0)]],
                                        device const float* posY [[buffer(1)]],
                                        device const float* posZ [[buffer(2)]],
                                        device const float* velX [[buffer(3)]],
                                        device const float* velY [[buffer(4)]],
                                        device const float* velZ [[buffer(5)]],
                                        device const float* charge [[buffer(6)]],
                                        device float* fieldX [[buffer(7)]],
                                        device float* fieldY [[buffer(8)]],
                                        device float* fieldZ [[buffer(9)]],
                                        constant uint& count [[buffer(10)]],
                                        constant float& permeabilityOver4Pi [[buffer(11)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float bx = 0.0f;
    float by = 0.0f;
    float bz = 0.0f;
    for (uint j = 0; j < count; ++j) {
        if (j == i) continue;
        float dx = px - posX[j];
        float dy = py - posY[j];
        float dz = pz - posZ[j];
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= 0.0f) continue;
        float invR3 = 1.0f / (r2 * sqrt(r2));
        float factor = charge[j] * invR3;
        float vx = velX[j];
        float vy = velY[j];
        float vz = velZ[j];
        bx += factor * (vy * dz - vz * dy);
        by += factor * (vz * dx - vx * dz);
        bz += factor * (vx * dy - vy * dx);
    }
    fieldX[i] = permeabilityOver4Pi * bx;
    fieldY[i] = permeabilityOver4Pi * by;
    fieldZ[i] = permeabilityOver4Pi * bz;
}

// Matches Physics/Fluids/SPH.hpp's cubicSplineKernel and
// cubicSplineKernelGradient exactly (Monaghan & Lattanzio 1985, normalized
// for 3D). Ordinary device functions, visible to every kernel below in the
// same compiled library, unlike GLSL's per-shader-program source strings.
inline float cubic_spline_kernel(float r, float h) {
    float q = r / h;
    float sigma = 0.3183098861837907f / (h * h * h);
    if (q < 1.0f) return sigma * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    if (q < 2.0f) { float t = 2.0f - q; return sigma * 0.25f * t * t * t; }
    return 0.0f;
}
inline float cubic_spline_kernel_gradient_over_r(float r, float h) {
    if (r <= 0.0f) return 0.0f;
    float q = r / h;
    float sigma = 0.3183098861837907f / (h * h * h);
    float dWdq;
    if (q < 1.0f) dWdq = sigma * (-3.0f * q + 2.25f * q * q);
    else if (q < 2.0f) { float t = 2.0f - q; dWdq = -0.75f * sigma * t * t; }
    else dWdq = 0.0f;
    return (dWdq / h) / r;
}

// Direct O(n^2) with a distance cutoff at the kernel's compact support
// (2h), not a spatial acceleration structure; see src/Compute/README.md.
// Matches Physics/Fluids/SPH.hpp's computeDensityAndPressure exactly,
// including the self term.
kernel void sph_density_pressure_kernel(device const float* posX [[buffer(0)]],
                                        device const float* posY [[buffer(1)]],
                                        device const float* posZ [[buffer(2)]],
                                        device const float* mass [[buffer(3)]],
                                        device float* density [[buffer(4)]],
                                        device float* pressure [[buffer(5)]],
                                        constant uint& count [[buffer(6)]],
                                        constant float& smoothingLength [[buffer(7)]],
                                        constant float& equationOfStateK [[buffer(8)]],
                                        constant float& polytropicIndex [[buffer(9)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float supportRadius = 2.0f * smoothingLength;
    float sum = 0.0f;
    for (uint j = 0; j < count; ++j) {
        float dx = px - posX[j];
        float dy = py - posY[j];
        float dz = pz - posZ[j];
        float r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r >= supportRadius) continue;
        sum += mass[j] * cubic_spline_kernel(r, smoothingLength);
    }
    density[i] = sum;
    pressure[i] = equationOfStateK * pow(sum, polytropicIndex);
}

// Matches Physics/Fluids/SPH.hpp's pressureAccelerations exactly. density
// and pressure are inputs here (already computed by sph_density_pressure_kernel
// or its CPU equivalent), not recomputed.
kernel void sph_pressure_acceleration_kernel(
    device const float* posX [[buffer(0)]], device const float* posY [[buffer(1)]],
    device const float* posZ [[buffer(2)]], device const float* mass [[buffer(3)]],
    device const float* density [[buffer(4)]], device const float* pressure [[buffer(5)]],
    device float* accX [[buffer(6)]], device float* accY [[buffer(7)]],
    device float* accZ [[buffer(8)]], constant uint& count [[buffer(9)]],
    constant float& smoothingLength [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float supportRadius = 2.0f * smoothingLength;
    float targetTerm = pressure[i] / (density[i] * density[i]);
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    for (uint j = 0; j < count; ++j) {
        if (j == i) continue;
        float dx = px - posX[j];
        float dy = py - posY[j];
        float dz = pz - posZ[j];
        float r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r >= supportRadius) continue;
        float sourceTerm = pressure[j] / (density[j] * density[j]);
        float gradOverR = cubic_spline_kernel_gradient_over_r(r, smoothingLength);
        float factor = gradOverR * mass[j] * (targetTerm + sourceTerm);
        ax += factor * dx;
        ay += factor * dy;
        az += factor * dz;
    }
    accX[i] = -ax;
    accY[i] = -ay;
    accZ[i] = -az;
}

// One explicit-Euler diffusion step over a periodic grid, flat row-major,
// interior cells only, periodic wraparound via modular index arithmetic.
// Matches Physics/Thermodynamics/HeatEquation3D.cpp's step() exactly.
kernel void heat_equation_3d_step_kernel(device const float* temperature [[buffer(0)]],
                                         device float* next [[buffer(1)]],
                                         constant int& nx [[buffer(2)]],
                                         constant int& ny [[buffer(3)]],
                                         constant int& nz [[buffer(4)]],
                                         constant float& factor [[buffer(5)]],
                                         uint tid [[thread_position_in_grid]]) {
    int idx = int(tid);
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    int xm = (i + nx - 1) % nx;
    int xp = (i + 1) % nx;
    int ym = (j + ny - 1) % ny;
    int yp = (j + 1) % ny;
    int zm = (k + nz - 1) % nz;
    int zp = (k + 1) % nz;
    float here = temperature[idx];
    float laplacian = temperature[(xp * ny + j) * nz + k] + temperature[(xm * ny + j) * nz + k]
                     + temperature[(i * ny + yp) * nz + k] + temperature[(i * ny + ym) * nz + k]
                     + temperature[(i * ny + j) * nz + zp] + temperature[(i * ny + j) * nz + zm]
                     - 6.0f * here;
    next[idx] = here + factor * laplacian;
}

// Two dispatches, run in sequence by MetalBackend::acoustic3DStep. Matches
// Physics/Acoustics/Acoustic3D.cpp's step() exactly.
kernel void acoustic_3d_velocity_kernel(device const float* pressure [[buffer(0)]],
                                        device const float* velocityXIn [[buffer(1)]],
                                        device const float* velocityYIn [[buffer(2)]],
                                        device const float* velocityZIn [[buffer(3)]],
                                        device float* velocityXOut [[buffer(4)]],
                                        device float* velocityYOut [[buffer(5)]],
                                        device float* velocityZOut [[buffer(6)]],
                                        constant int& nx [[buffer(7)]],
                                        constant int& ny [[buffer(8)]],
                                        constant int& nz [[buffer(9)]],
                                        constant float& velocityFactor [[buffer(10)]],
                                        uint tid [[thread_position_in_grid]]) {
    int idx = int(tid);
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    int xp = (i + 1) % nx;
    int yp = (j + 1) % ny;
    int zp = (k + 1) % nz;
    float p = pressure[idx];
    velocityXOut[idx] = velocityXIn[idx] - velocityFactor * (pressure[(xp * ny + j) * nz + k] - p);
    velocityYOut[idx] = velocityYIn[idx] - velocityFactor * (pressure[(i * ny + yp) * nz + k] - p);
    velocityZOut[idx] = velocityZIn[idx] - velocityFactor * (pressure[(i * ny + j) * nz + zp] - p);
}

kernel void acoustic_3d_pressure_kernel(device const float* pressureIn [[buffer(0)]],
                                        device const float* velocityX [[buffer(1)]],
                                        device const float* velocityY [[buffer(2)]],
                                        device const float* velocityZ [[buffer(3)]],
                                        device float* pressureOut [[buffer(4)]],
                                        constant int& nx [[buffer(5)]],
                                        constant int& ny [[buffer(6)]],
                                        constant int& nz [[buffer(7)]],
                                        constant float& pressureFactor [[buffer(8)]],
                                        uint tid [[thread_position_in_grid]]) {
    int idx = int(tid);
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    int xm = (i + nx - 1) % nx;
    int ym = (j + ny - 1) % ny;
    int zm = (k + nz - 1) % nz;
    float divergence = (velocityX[idx] - velocityX[(xm * ny + j) * nz + k])
                      + (velocityY[idx] - velocityY[(i * ny + ym) * nz + k])
                      + (velocityZ[idx] - velocityZ[(i * ny + j) * nz + zm]);
    pressureOut[idx] = pressureIn[idx] - pressureFactor * divergence;
}

// Two dispatches, run in sequence by MetalBackend::maxwell3DStep. Matches
// Physics/Electromagnetism/Maxwell3D.cpp's step() exactly.
kernel void maxwell_3d_b_kernel(device const float* ex [[buffer(0)]],
                                device const float* ey [[buffer(1)]],
                                device const float* ez [[buffer(2)]],
                                device const float* bxIn [[buffer(3)]],
                                device const float* byIn [[buffer(4)]],
                                device const float* bzIn [[buffer(5)]],
                                device float* bxOut [[buffer(6)]],
                                device float* byOut [[buffer(7)]],
                                device float* bzOut [[buffer(8)]],
                                constant int& nx [[buffer(9)]], constant int& ny [[buffer(10)]],
                                constant int& nz [[buffer(11)]],
                                constant float& bFactor [[buffer(12)]],
                                uint tid [[thread_position_in_grid]]) {
    int idx = int(tid);
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    int ip = (i + 1) % nx;
    int jp = (j + 1) % ny;
    int kp = (k + 1) % nz;
    bxOut[idx] = bxIn[idx] - bFactor * ((ez[(i * ny + jp) * nz + k] - ez[idx]) -
                                        (ey[(i * ny + j) * nz + kp] - ey[idx]));
    byOut[idx] = byIn[idx] - bFactor * ((ex[(i * ny + j) * nz + kp] - ex[idx]) -
                                        (ez[(ip * ny + j) * nz + k] - ez[idx]));
    bzOut[idx] = bzIn[idx] - bFactor * ((ey[(ip * ny + j) * nz + k] - ey[idx]) -
                                        (ex[(i * ny + jp) * nz + k] - ex[idx]));
}

kernel void maxwell_3d_e_kernel(device const float* exIn [[buffer(0)]],
                                device const float* eyIn [[buffer(1)]],
                                device const float* ezIn [[buffer(2)]],
                                device const float* bx [[buffer(3)]],
                                device const float* by [[buffer(4)]],
                                device const float* bz [[buffer(5)]],
                                device float* exOut [[buffer(6)]],
                                device float* eyOut [[buffer(7)]],
                                device float* ezOut [[buffer(8)]],
                                constant int& nx [[buffer(9)]], constant int& ny [[buffer(10)]],
                                constant int& nz [[buffer(11)]],
                                constant float& eFactor [[buffer(12)]],
                                uint tid [[thread_position_in_grid]]) {
    int idx = int(tid);
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    int im = (i + nx - 1) % nx;
    int jm = (j + ny - 1) % ny;
    int km = (k + nz - 1) % nz;
    exOut[idx] = exIn[idx] + eFactor * ((bz[idx] - bz[(i * ny + jm) * nz + k]) -
                                        (by[idx] - by[(i * ny + j) * nz + km]));
    eyOut[idx] = eyIn[idx] + eFactor * ((bx[idx] - bx[(i * ny + j) * nz + km]) -
                                        (bz[idx] - bz[(im * ny + j) * nz + k]));
    ezOut[idx] = ezIn[idx] + eFactor * ((by[idx] - by[(im * ny + j) * nz + k]) -
                                        (bx[idx] - bx[(i * ny + jm) * nz + k]));
}

// The compressible Euler equations' gas-dynamics relations, as device
// functions shared by eulerian_fluid_3d_sweep_kernel below. Matches
// Physics/Fluids/Eulerian3D.cpp's own (double-precision) copies exactly,
// float throughout here to match the rest of this file's kernels.
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

float euler_pressure_of(EulerState state, float gamma) {
    float u = state.momentumNormal / state.density;
    float v = state.momentumTangent1 / state.density;
    float w = state.momentumTangent2 / state.density;
    float kinetic = 0.5f * state.density * (u * u + v * v + w * w);
    return (gamma - 1.0f) * (state.energy - kinetic);
}

float euler_sound_speed_of(float density, float pressure, float gamma) {
    return sqrt(gamma * pressure / density);
}

EulerFlux euler_flux_of(EulerState state, float gamma) {
    float u = state.momentumNormal / state.density;
    float v = state.momentumTangent1 / state.density;
    float w = state.momentumTangent2 / state.density;
    float pressure = euler_pressure_of(state, gamma);
    EulerFlux result;
    result.density = state.momentumNormal;
    result.momentumNormal = state.momentumNormal * u + pressure;
    result.momentumTangent1 = state.momentumNormal * v;
    result.momentumTangent2 = state.momentumNormal * w;
    result.energy = u * (state.energy + pressure);
    return result;
}

EulerFlux euler_rusanov_flux(EulerState left, EulerState right, float gamma) {
    EulerFlux fluxLeft = euler_flux_of(left, gamma);
    EulerFlux fluxRight = euler_flux_of(right, gamma);

    float velocityLeft = left.momentumNormal / left.density;
    float velocityRight = right.momentumNormal / right.density;
    float soundLeft = euler_sound_speed_of(left.density, euler_pressure_of(left, gamma), gamma);
    float soundRight =
        euler_sound_speed_of(right.density, euler_pressure_of(right, gamma), gamma);
    float maxSpeed = max(fabs(velocityLeft) + soundLeft, fabs(velocityRight) + soundRight);

    EulerFlux result;
    result.density = 0.5f * (fluxLeft.density + fluxRight.density) -
                     0.5f * maxSpeed * (right.density - left.density);
    result.momentumNormal =
        0.5f * (fluxLeft.momentumNormal + fluxRight.momentumNormal) -
        0.5f * maxSpeed * (right.momentumNormal - left.momentumNormal);
    result.momentumTangent1 =
        0.5f * (fluxLeft.momentumTangent1 + fluxRight.momentumTangent1) -
        0.5f * maxSpeed * (right.momentumTangent1 - left.momentumTangent1);
    result.momentumTangent2 =
        0.5f * (fluxLeft.momentumTangent2 + fluxRight.momentumTangent2) -
        0.5f * maxSpeed * (right.momentumTangent2 - left.momentumTangent2);
    result.energy = 0.5f * (fluxLeft.energy + fluxRight.energy) -
                    0.5f * maxSpeed * (right.energy - left.energy);
    return result;
}

// One dimensional-split sweep along `axis` (0/1/2), first-order Rusanov
// flux, periodic wraparound via modular index arithmetic. Matches
// Physics/Fluids/Eulerian3D.cpp's sweep() exactly for one axis.
kernel void eulerian_fluid_3d_sweep_kernel(
        device const float* density [[buffer(0)]],
        device const float* momentumNormal [[buffer(1)]],
        device const float* momentumTangent1 [[buffer(2)]],
        device const float* momentumTangent2 [[buffer(3)]],
        device const float* energy [[buffer(4)]],
        device float* nextDensity [[buffer(5)]],
        device float* nextMomentumNormal [[buffer(6)]],
        device float* nextMomentumTangent1 [[buffer(7)]],
        device float* nextMomentumTangent2 [[buffer(8)]],
        device float* nextEnergy [[buffer(9)]],
        constant int& nx [[buffer(10)]],
        constant int& ny [[buffer(11)]],
        constant int& nz [[buffer(12)]],
        constant int& axis [[buffer(13)]],
        constant float& gamma [[buffer(14)]],
        constant float& dtOverSpacing [[buffer(15)]],
        uint tid [[thread_position_in_grid]]) {
    int idx = int(tid);
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    int im = i, ip = i, jm = j, jp = j, km = k, kp = k;
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

    int leftIdx = (im * ny + jm) * nz + km;
    int rightIdx = (ip * ny + jp) * nz + kp;

    EulerState hereState;
    hereState.density = density[idx];
    hereState.momentumNormal = momentumNormal[idx];
    hereState.momentumTangent1 = momentumTangent1[idx];
    hereState.momentumTangent2 = momentumTangent2[idx];
    hereState.energy = energy[idx];

    EulerState leftState;
    leftState.density = density[leftIdx];
    leftState.momentumNormal = momentumNormal[leftIdx];
    leftState.momentumTangent1 = momentumTangent1[leftIdx];
    leftState.momentumTangent2 = momentumTangent2[leftIdx];
    leftState.energy = energy[leftIdx];

    EulerState rightState;
    rightState.density = density[rightIdx];
    rightState.momentumNormal = momentumNormal[rightIdx];
    rightState.momentumTangent1 = momentumTangent1[rightIdx];
    rightState.momentumTangent2 = momentumTangent2[rightIdx];
    rightState.energy = energy[rightIdx];

    EulerFlux fluxLeft = euler_rusanov_flux(leftState, hereState, gamma);
    EulerFlux fluxRight = euler_rusanov_flux(hereState, rightState, gamma);

    nextDensity[idx] = density[idx] - dtOverSpacing * (fluxRight.density - fluxLeft.density);
    nextMomentumNormal[idx] =
        momentumNormal[idx] -
        dtOverSpacing * (fluxRight.momentumNormal - fluxLeft.momentumNormal);
    nextMomentumTangent1[idx] =
        momentumTangent1[idx] -
        dtOverSpacing * (fluxRight.momentumTangent1 - fluxLeft.momentumTangent1);
    nextMomentumTangent2[idx] =
        momentumTangent2[idx] -
        dtOverSpacing * (fluxRight.momentumTangent2 - fluxLeft.momentumTangent2);
    nextEnergy[idx] = energy[idx] - dtOverSpacing * (fluxRight.energy - fluxLeft.energy);
}

// A batched, power-of-two, radix-2 Cooley-Tukey FFT: two kernels dispatched
// in a host-side loop (fft_bit_reversal_permute_kernel once, then
// fft_butterfly_stage_kernel once per stage), matching Math/FFT.hpp's
// detail::fftImpl exactly per batch. Out-of-place throughout: unlike
// fftImpl's in-place swap-if-i<j permutation, every thread here writes its
// own destination directly, and each butterfly stage reads the previous
// stage's separate output buffer (MetalBackend::fftBatched ping-pongs
// between two buffers on the host side).
kernel void fft_bit_reversal_permute_kernel(device const float* real [[buffer(0)]],
                                            device const float* imag [[buffer(1)]],
                                            device float* nextReal [[buffer(2)]],
                                            device float* nextImag [[buffer(3)]],
                                            constant uint& length [[buffer(4)]],
                                            constant uint& bitCount [[buffer(5)]],
                                            constant uint& total [[buffer(6)]],
                                            uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    uint idx = tid;
    uint batch = idx / length;
    uint i = idx % length;

    uint reversed = 0;
    uint value = i;
    for (uint b = 0; b < bitCount; ++b) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }

    uint destination = batch * length + reversed;
    nextReal[destination] = real[idx];
    nextImag[destination] = imag[idx];
}

kernel void fft_butterfly_stage_kernel(device const float* real [[buffer(0)]],
                                       device const float* imag [[buffer(1)]],
                                       device float* nextReal [[buffer(2)]],
                                       device float* nextImag [[buffer(3)]],
                                       constant uint& length [[buffer(4)]],
                                       constant uint& len [[buffer(5)]],
                                       constant float& angle [[buffer(6)]],
                                       constant uint& total [[buffer(7)]],
                                       uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    uint halfLen = len / 2;
    uint pairsPerBatch = length / 2;
    uint batch = tid / pairsPerBatch;
    uint pairInBatch = tid % pairsPerBatch;
    uint blockIndex = pairInBatch / halfLen;
    uint k = pairInBatch % halfLen;
    uint blockStart = blockIndex * len;
    uint base = batch * length;
    uint i0 = base + blockStart + k;
    uint i1 = i0 + halfLen;

    float twiddleReal = cos(float(k) * angle);
    float twiddleImag = sin(float(k) * angle);
    float evenReal = real[i0];
    float evenImag = imag[i0];
    float oddReal0 = real[i1];
    float oddImag0 = imag[i1];
    float oddReal = oddReal0 * twiddleReal - oddImag0 * twiddleImag;
    float oddImag = oddReal0 * twiddleImag + oddImag0 * twiddleReal;

    nextReal[i0] = evenReal + oddReal;
    nextImag[i0] = evenImag + oddImag;
    nextReal[i1] = evenReal - oddReal;
    nextImag[i1] = evenImag - oddImag;
}

kernel void fft_scale_kernel(device const float* real [[buffer(0)]],
                             device const float* imag [[buffer(1)]],
                             device float* nextReal [[buffer(2)]],
                             device float* nextImag [[buffer(3)]],
                             constant float& scale [[buffer(4)]],
                             constant uint& total [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    nextReal[tid] = real[tid] * scale;
    nextImag[tid] = imag[tid] * scale;
}

// Dense matrix-vector and matrix-matrix multiply, row-major, one thread
// per output element -- the same shape as every other kernel here, not a
// tiled/shared-memory GEMM: a reference kernel, not a peak-throughput one
// (see MetalBackend's own doc comment on MTLResourceStorageModeShared for
// the same tradeoff elsewhere in this file).
kernel void mat_vec_kernel(device const float* mat [[buffer(0)]],
                           device const float* vec [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           constant uint& rows [[buffer(3)]],
                           constant uint& cols [[buffer(4)]],
                           uint tid [[thread_position_in_grid]]) {
    if (tid >= rows) return;
    float total = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        total += mat[tid * cols + c] * vec[c];
    }
    result[tid] = total;
}

kernel void mat_mul_kernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           constant uint& aRows [[buffer(3)]],
                           constant uint& aCols [[buffer(4)]],
                           constant uint& bCols [[buffer(5)]],
                           uint tid [[thread_position_in_grid]]) {
    uint total = aRows * bCols;
    if (tid >= total) return;
    uint r = tid / bCols;
    uint c = tid % bCols;
    float sum = 0.0f;
    for (uint k = 0; k < aCols; ++k) {
        sum += a[r * aCols + k] * b[k * bCols + c];
    }
    result[tid] = sum;
}

// Dense LU decomposition with partial pivoting: three kernels dispatched
// in a host-side loop, one iteration per pivot column (MetalBackend::
// luDecomposeGpu), matching Math/LinearSolve.hpp's luDecompose exactly.
// arg_max_abs_column_kernel is min_index_kernel's own two-pass
// reduce-then-finish-on-CPU shape, adapted to a strided column read and a
// max (not min) comparison; its finish step also recovers the winning
// value (not just the index), since luDecomposeGpu needs it for the
// singularity check luDecompose itself makes with the same value.
kernel void arg_max_abs_column_kernel(device const float* mat [[buffer(0)]],
                                      device float* partialValues [[buffer(1)]],
                                      device float* partialIndices [[buffer(2)]],
                                      constant uint& n [[buffer(3)]],
                                      constant uint& column [[buffer(4)]],
                                      constant uint& startRow [[buffer(5)]],
                                      constant uint& activeRows [[buffer(6)]],
                                      uint i [[thread_position_in_grid]],
                                      uint local [[thread_position_in_threadgroup]],
                                      uint groupId [[threadgroup_position_in_grid]],
                                      threadgroup float* scratchValue [[threadgroup(0)]],
                                      threadgroup uint* scratchIndex [[threadgroup(1)]]) {
    if (i < activeRows) {
        uint row = startRow + i;
        scratchValue[local] = fabs(mat[row * n + column]);
        scratchIndex[local] = row;
    } else {
        scratchValue[local] = -1.0f;
        scratchIndex[local] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local < stride && scratchValue[local + stride] > scratchValue[local]) {
            scratchValue[local] = scratchValue[local + stride];
            scratchIndex[local] = scratchIndex[local + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local == 0) {
        partialValues[groupId] = scratchValue[0];
        partialIndices[groupId] = float(scratchIndex[0]);
    }
}
)msl";

// Split in two: a single raw string this large exceeds the C++ standard's
// minimum guaranteed string-literal length (65536 characters after
// concatenation), which most compilers accept beyond but is not something
// to rely on. Adjacent literal concatenation would not help (the standard's
// limit applies to the already-concatenated result), so these are two
// separate variables, joined at runtime in create() instead, right at a
// clean kernel boundary.
constexpr const char* kKernelSourcePart2 = R"msl(
// Copies the whole matrix through, swapping rowA and rowB (a no-op copy
// when they're equal): out-of-place, like every other kernel here, even
// though an in-place swap would also be race-free (each thread owns
// exactly one column across both rows).
kernel void swap_rows_kernel(device const float* mat [[buffer(0)]],
                             device float* nextMat [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             constant uint& rowA [[buffer(3)]],
                             constant uint& rowB [[buffer(4)]],
                             uint tid [[thread_position_in_grid]]) {
    uint total = n * n;
    if (tid >= total) return;
    uint row = tid / n;
    uint col = tid % n;
    uint sourceRow = row;
    if (row == rowA) {
        sourceRow = rowB;
    } else if (row == rowB) {
        sourceRow = rowA;
    }
    nextMat[tid] = mat[sourceRow * n + col];
}

// The trailing-submatrix rank-1 update for pivot column k: rows/columns
// already finalized (row <= k or col < k) pass through unchanged; row > k,
// col == k stores the elimination factor (L's entry, Doolittle packing);
// row > k, col > k applies the elimination itself. Matches
// Math/LinearSolve.hpp's luDecompose inner loop exactly, just dispatched
// over the whole matrix at once (with the passthrough guard) rather than
// only the active submatrix, since the output buffer is separate from the
// input.
kernel void lu_elimination_step_kernel(device const float* mat [[buffer(0)]],
                                       device float* nextMat [[buffer(1)]],
                                       constant uint& n [[buffer(2)]],
                                       constant uint& k [[buffer(3)]],
                                       uint tid [[thread_position_in_grid]]) {
    uint total = n * n;
    if (tid >= total) return;
    uint row = tid / n;
    uint col = tid % n;
    if (row <= k || col < k) {
        nextMat[tid] = mat[tid];
        return;
    }
    float factor = mat[row * n + k] / mat[k * n + k];
    if (col == k) {
        nextMat[tid] = factor;
    } else {
        nextMat[tid] = mat[row * n + col] - factor * mat[k * n + col];
    }
}

// Dense Cholesky factorization: two kernels dispatched in a host-side
// loop, one iteration per column (MetalBackend::choleskyDecomposeGpu),
// matching Math/LinearSolve.hpp's choleskyDecompose exactly. Unlike LU,
// every column's writes land in memory no later step ever overwrites, so
// both kernels accumulate into one resident `l` buffer in place across
// the whole decomposition rather than ping-ponging between two.
// cholesky_diagonal_kernel writes a -1 sentinel (impossible for a genuine
// diagonal entry, which is always non-negative) when `a` isn't actually
// positive-definite; choleskyDecomposeGpu reads back that one value each
// step to detect it and stop early, the same small-per-step-readback
// shape luDecomposeGpu's pivot search already uses.
kernel void cholesky_diagonal_kernel(device const float* a [[buffer(0)]],
                                     device float* l [[buffer(1)]],
                                     constant uint& n [[buffer(2)]],
                                     constant uint& j [[buffer(3)]],
                                     uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float total = a[j * n + j];
    for (uint k = 0; k < j; ++k) {
        float ljk = l[j * n + k];
        total -= ljk * ljk;
    }
    if (!(total > 0.0f) || !isfinite(total)) {
        l[j * n + j] = -1.0f;
        return;
    }
    l[j * n + j] = sqrt(total);
}

kernel void cholesky_column_kernel(device const float* a [[buffer(0)]],
                                   device float* l [[buffer(1)]],
                                   constant uint& n [[buffer(2)]],
                                   constant uint& j [[buffer(3)]],
                                   uint tid [[thread_position_in_grid]]) {
    uint i = j + 1 + tid;
    if (i >= n) return;
    float total = a[i * n + j];
    for (uint k = 0; k < j; ++k) {
        total -= l[i * n + k] * l[j * n + k];
    }
    l[i * n + j] = total / l[j * n + j];
}

// Dense QR decomposition via Householder reflections: three kernels
// dispatched in a host-side loop, one iteration per column
// (MetalBackend::qrDecomposeGpu), matching Math/Eigen.hpp's qrDecompose
// exactly. qr_column_norm_squared_kernel is a single-thread reduction (the
// same "one thread loops, no parallel tree reduction needed" shape
// cholesky_diagonal_kernel already uses, since this is a tiny fraction of
// a step's total work next to the two per-column/per-row kernels below);
// it also returns r(k,k) itself, so the host can compute alpha and
// vNormSquared without a second GPU round trip. qr_apply_left_kernel and
// qr_accumulate_q_kernel both read the reflection vector implicitly from
// r's own (pre-this-step) column k rather than taking a separate v buffer.
kernel void qr_column_norm_squared_kernel(device const float* r [[buffer(0)]],
                                          device float* output [[buffer(1)]],
                                          constant uint& rows [[buffer(2)]],
                                          constant uint& cols [[buffer(3)]],
                                          constant uint& k [[buffer(4)]],
                                          uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float sumSquares = 0.0f;
    for (uint i = k; i < rows; ++i) {
        float value = r[i * cols + k];
        sumSquares += value * value;
    }
    output[0] = sumSquares;
    output[1] = r[k * cols + k];
}

// One thread per column c in [0, cols): passthrough for c < k; for c >= k,
// applies the Householder reflection to that column alone (the same
// per-column-loop shape cholesky_column_kernel uses, here because each
// column's update is independent of every other column's, not because of
// a triangular dependency).
kernel void qr_apply_left_kernel(device const float* r [[buffer(0)]],
                                 device float* nextR [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]],
                                 constant uint& cols [[buffer(3)]],
                                 constant uint& k [[buffer(4)]],
                                 constant float& alpha [[buffer(5)]],
                                 constant float& vNormSquared [[buffer(6)]],
                                 uint tid [[thread_position_in_grid]]) {
    if (tid >= cols) return;
    if (tid < k) {
        for (uint i = 0; i < rows; ++i) {
            nextR[i * cols + tid] = r[i * cols + tid];
        }
        return;
    }
    for (uint i = 0; i < k; ++i) {
        nextR[i * cols + tid] = r[i * cols + tid];
    }
    float dotProduct = 0.0f;
    for (uint i = k; i < rows; ++i) {
        float vi = (i == k) ? (r[k * cols + k] - alpha) : r[i * cols + k];
        dotProduct += vi * r[i * cols + tid];
    }
    float factor = 2.0f * dotProduct / vNormSquared;
    for (uint i = k; i < rows; ++i) {
        float vi = (i == k) ? (r[k * cols + k] - alpha) : r[i * cols + k];
        nextR[i * cols + tid] = r[i * cols + tid] - factor * vi;
    }
}

// One thread per row in [0, rows): accumulates the same reflection
// (recovered from r, not q) into q, matching qrDecompose's own order of
// operations (the reflection is captured once per step, applied to both
// the working r and the accumulating q).
kernel void qr_accumulate_q_kernel(device const float* r [[buffer(0)]],
                                   device const float* q [[buffer(1)]],
                                   device float* nextQ [[buffer(2)]],
                                   constant uint& rows [[buffer(3)]],
                                   constant uint& cols [[buffer(4)]],
                                   constant uint& k [[buffer(5)]],
                                   constant float& alpha [[buffer(6)]],
                                   constant float& vNormSquared [[buffer(7)]],
                                   uint tid [[thread_position_in_grid]]) {
    if (tid >= rows) return;
    for (uint i = 0; i < k; ++i) {
        nextQ[tid * rows + i] = q[tid * rows + i];
    }
    float dotProduct = 0.0f;
    for (uint i = k; i < rows; ++i) {
        float vi = (i == k) ? (r[k * cols + k] - alpha) : r[i * cols + k];
        dotProduct += vi * q[tid * rows + i];
    }
    float factor = 2.0f * dotProduct / vNormSquared;
    for (uint i = k; i < rows; ++i) {
        float vi = (i == k) ? (r[k * cols + k] - alpha) : r[i * cols + k];
        nextQ[tid * rows + i] = q[tid * rows + i] - factor * vi;
    }
}

// The cyclic Jacobi eigenvalue algorithm's round-robin/tournament-ordered
// parallel form: two kernels dispatched per round (MetalBackend::
// jacobiEigenSymmetricGpu), `pairOf[i]` giving index i's partner this
// round (or i itself, unpaired). Both kernels independently recompute the
// same rotation angle from the pair's *original*, pre-round a(p,p)/a(q,q)/
// a(p,q) -- deterministic given the same three inputs, so the two threads
// owning one pair (or the column-mix and row-mix kernels touching the
// same pair) never need to communicate it explicitly. Applying a two-sided
// similarity for several simultaneous disjoint pairs is genuinely two
// passes, not one: jacobi_eigen_column_mix_kernel computes `b = a * q`
// (and mixes eigenvectors `v` the same way, a pure right-multiply too),
// then jacobi_eigen_row_mix_kernel computes `a' = q^T * b` from that
// pass's complete output -- recomputing the rotation from the original
// `a` again (not from `b`, whose diagonal the column-mix pass has already
// disturbed).
inline void jacobiRotationOf(float app, float aqq, float apq, thread float& cosOut,
                             thread float& sinOut) {
    float theta = (aqq - app) / (2.0f * apq);
    float sign = (theta < 0.0f) ? -1.0f : 1.0f;
    float t = sign / (fabs(theta) + sqrt(theta * theta + 1.0f));
    cosOut = 1.0f / sqrt(t * t + 1.0f);
    sinOut = t * cosOut;
}

kernel void jacobi_eigen_column_mix_kernel(device const float* a [[buffer(0)]],
                                           device const float* v [[buffer(1)]],
                                           device const uint* pairOf [[buffer(2)]],
                                           device float* nextB [[buffer(3)]],
                                           device float* nextV [[buffer(4)]],
                                           constant uint& n [[buffer(5)]],
                                           constant float& epsilon [[buffer(6)]],
                                           uint c [[thread_position_in_grid]]) {
    if (c >= n) return;
    uint partner = pairOf[c];
    if (partner == c) {
        for (uint i = 0; i < n; ++i) {
            nextB[i * n + c] = a[i * n + c];
            nextV[i * n + c] = v[i * n + c];
        }
        return;
    }
    uint p = min(c, partner);
    uint q = max(c, partner);
    float apq = a[p * n + q];
    if (fabs(apq) <= epsilon) {
        for (uint i = 0; i < n; ++i) {
            nextB[i * n + c] = a[i * n + c];
            nextV[i * n + c] = v[i * n + c];
        }
        return;
    }
    float cosT;
    float sinT;
    jacobiRotationOf(a[p * n + p], a[q * n + q], apq, cosT, sinT);
    if (c == p) {
        for (uint i = 0; i < n; ++i) {
            nextB[i * n + p] = cosT * a[i * n + p] - sinT * a[i * n + q];
            nextV[i * n + p] = cosT * v[i * n + p] - sinT * v[i * n + q];
        }
    } else {
        for (uint i = 0; i < n; ++i) {
            nextB[i * n + q] = sinT * a[i * n + p] + cosT * a[i * n + q];
            nextV[i * n + q] = sinT * v[i * n + p] + cosT * v[i * n + q];
        }
    }
}

kernel void jacobi_eigen_row_mix_kernel(device const float* a [[buffer(0)]],
                                        device const float* b [[buffer(1)]],
                                        device const uint* pairOf [[buffer(2)]],
                                        device float* nextA [[buffer(3)]],
                                        constant uint& n [[buffer(4)]],
                                        constant float& epsilon [[buffer(5)]],
                                        uint r [[thread_position_in_grid]]) {
    if (r >= n) return;
    uint partner = pairOf[r];
    if (partner == r) {
        for (uint j = 0; j < n; ++j) {
            nextA[r * n + j] = b[r * n + j];
        }
        return;
    }
    uint p = min(r, partner);
    uint q = max(r, partner);
    float apq = a[p * n + q];
    if (fabs(apq) <= epsilon) {
        for (uint j = 0; j < n; ++j) {
            nextA[r * n + j] = b[r * n + j];
        }
        return;
    }
    float cosT;
    float sinT;
    jacobiRotationOf(a[p * n + p], a[q * n + q], apq, cosT, sinT);
    if (r == p) {
        for (uint j = 0; j < n; ++j) {
            nextA[p * n + j] = cosT * b[p * n + j] - sinT * b[q * n + j];
        }
    } else {
        for (uint j = 0; j < n; ++j) {
            nextA[q * n + j] = sinT * b[p * n + j] + cosT * b[q * n + j];
        }
    }
}

// One-sided Jacobi SVD's round-robin form: one dispatch per round (unlike
// the symmetric case, this rotation is one-sided -- a <- a*q, no
// corresponding left multiplication -- so disjoint pairs never share a
// cross term and every round genuinely is independent column pairs read
// and written directly, matching MetalBackend::jacobiSvdGpu).
kernel void jacobi_svd_round_kernel(device const float* a [[buffer(0)]],
                                    device const float* v [[buffer(1)]],
                                    device const uint* pairOf [[buffer(2)]],
                                    device float* nextA [[buffer(3)]],
                                    device float* nextV [[buffer(4)]],
                                    constant uint& rows [[buffer(5)]],
                                    constant uint& cols [[buffer(6)]],
                                    constant float& epsilon [[buffer(7)]],
                                    uint c [[thread_position_in_grid]]) {
    if (c >= cols) return;
    uint partner = pairOf[c];
    if (partner == c) {
        for (uint i = 0; i < rows; ++i) {
            nextA[i * cols + c] = a[i * cols + c];
        }
        for (uint i = 0; i < cols; ++i) {
            nextV[i * cols + c] = v[i * cols + c];
        }
        return;
    }
    uint p = min(c, partner);
    uint q = max(c, partner);
    float alpha = 0.0f;
    float beta = 0.0f;
    float gamma = 0.0f;
    for (uint i = 0; i < rows; ++i) {
        float ap = a[i * cols + p];
        float aq = a[i * cols + q];
        alpha += ap * ap;
        beta += aq * aq;
        gamma += ap * aq;
    }
    float threshold = epsilon * sqrt(alpha * beta);
    if (fabs(gamma) <= threshold) {
        for (uint i = 0; i < rows; ++i) {
            nextA[i * cols + c] = a[i * cols + c];
        }
        for (uint i = 0; i < cols; ++i) {
            nextV[i * cols + c] = v[i * cols + c];
        }
        return;
    }
    float zeta = (beta - alpha) / (2.0f * gamma);
    float sign = (zeta < 0.0f) ? -1.0f : 1.0f;
    float t = sign / (fabs(zeta) + sqrt(1.0f + zeta * zeta));
    float cosT = 1.0f / sqrt(1.0f + t * t);
    float sinT = cosT * t;
    if (c == p) {
        for (uint i = 0; i < rows; ++i) {
            nextA[i * cols + p] = cosT * a[i * cols + p] - sinT * a[i * cols + q];
        }
        for (uint i = 0; i < cols; ++i) {
            nextV[i * cols + p] = cosT * v[i * cols + p] - sinT * v[i * cols + q];
        }
    } else {
        for (uint i = 0; i < rows; ++i) {
            nextA[i * cols + q] = sinT * a[i * cols + p] + cosT * a[i * cols + q];
        }
        for (uint i = 0; i < cols; ++i) {
            nextV[i * cols + q] = sinT * v[i * cols + p] + cosT * v[i * cols + q];
        }
    }
}

// Batched, independent, pointwise evaluation: one thread per element of
// `x`, no reduction, no multi-pass structure -- the simplest kernel shape
// in this file. See ComputeBackend.hpp's own comment on this family for
// why erf/erfc/gamma/logGamma use different (but well-established, and
// float32-safe) approximations than Math/SpecialFunctions.hpp's CPU
// `std::erf`/`std::tgamma`/`std::lgamma` calls, and why besselJ/besselY
// have no batched kernel here at all.
constant float kPi = 3.14159265358979323846f;

// Abramowitz & Stegun 7.1.26, computed directly as erfc(|x|) = poly *
// exp(-x^2) rather than via 1 - erf(|x|), avoiding the catastrophic
// cancellation "1 minus something near 1" would cost for large |x|.
float erfcOf(float x) {
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = t * (0.254829592f +
                      t * (-0.284496736f +
                           t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    float y = poly * exp(-ax * ax);
    return (x >= 0.0f) ? y : (2.0f - y);
}

float erfOf(float x) {
    return 1.0f - erfcOf(x);
}

// The Lanczos approximation (g = 5, Numerical Recipes' own coefficients),
// valid for x > 0; reflected via gamma(x) = pi / (sin(pi x) gamma(1 - x))
// for x <= 0, matching std::tgamma/std::lgamma's own domain.
float lgammaPositiveOf(float x) {
    float y = x;
    float tmp = x + 5.5f;
    tmp -= (x + 0.5f) * log(tmp);
    float ser = 1.000000000190015f;
    y += 1.0f;
    ser += 76.18009172947146f / y;
    y += 1.0f;
    ser += -86.50532032941677f / y;
    y += 1.0f;
    ser += 24.01409824083091f / y;
    y += 1.0f;
    ser += -1.231739572450155f / y;
    y += 1.0f;
    ser += 0.1208650973866179e-2f / y;
    y += 1.0f;
    ser += -0.5395239384953e-5f / y;
    return -tmp + log(2.5066282746310005f * ser / x);
}

float lgammaOf(float x) {
    if (x > 0.0f) {
        return lgammaPositiveOf(x);
    }
    return log(kPi) - log(fabs(sin(kPi * x))) - lgammaPositiveOf(1.0f - x);
}

float gammaOf(float x) {
    if (x > 0.0f) {
        return exp(lgammaPositiveOf(x));
    }
    return kPi / (sin(kPi * x) * exp(lgammaPositiveOf(1.0f - x)));
}

kernel void batch_erf_kernel(device const float* x [[buffer(0)]],
                             device float* result [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    result[tid] = erfOf(x[tid]);
}

kernel void batch_erfc_kernel(device const float* x [[buffer(0)]],
                              device float* result [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    result[tid] = erfcOf(x[tid]);
}

kernel void batch_gamma_kernel(device const float* x [[buffer(0)]],
                               device float* result [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    result[tid] = gammaOf(x[tid]);
}

kernel void batch_log_gamma_kernel(device const float* x [[buffer(0)]],
                                   device float* result [[buffer(1)]],
                                   constant uint& count [[buffer(2)]],
                                   uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    result[tid] = lgammaOf(x[tid]);
}

// The same upward recurrence Math/SpecialFunctions.hpp's legendreP uses
// (already validated in float32 by that header's own strict-warnings
// test, unlike the Bessel rational approximation), one thread per point,
// `n`/`m` fixed for the whole batch.
kernel void batch_legendre_p_kernel(device const float* x [[buffer(0)]],
                                    device float* result [[buffer(1)]],
                                    constant uint& count [[buffer(2)]],
                                    constant uint& n [[buffer(3)]],
                                    constant uint& m [[buffer(4)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    float value = x[tid];

    float pmm = 1.0f;
    if (m > 0) {
        float oneMinusX2 = (1.0f - value) * (1.0f + value);
        float somx2 = sqrt(oneMinusX2);
        float fact = 1.0f;
        for (uint i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0f;
        }
    }
    if (n == m) {
        result[tid] = pmm;
        return;
    }

    float pmmp1 = value * float(2 * m + 1) * pmm;
    if (n == m + 1) {
        result[tid] = pmmp1;
        return;
    }

    float pll = 0.0f;
    for (uint ll = m + 2; ll <= n; ++ll) {
        pll = (value * float(2 * ll - 1) * pmmp1 - float(ll + m - 1) * pmm) / float(ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    result[tid] = pll;
}

// Horner's method, one thread per point, the same fixed coefficients
// evaluated at every point.
kernel void batch_polynomial_eval_kernel(device const float* coefficients [[buffer(0)]],
                                         device const float* x [[buffer(1)]],
                                         device float* result [[buffer(2)]],
                                         constant uint& coefficientCount [[buffer(3)]],
                                         constant uint& count [[buffer(4)]],
                                         uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    float value = coefficients[coefficientCount - 1];
    for (uint i = coefficientCount - 1; i-- > 0;) {
        value = value * x[tid] + coefficients[i];
    }
    result[tid] = value;
}

// Held flat outside the table and a binary search per point, matching
// CubicSpline::operator() exactly; the fixed knots/second derivatives are
// the same for every point in the batch.
kernel void batch_cubic_spline_eval_kernel(device const float* knotsX [[buffer(0)]],
                                           device const float* knotsY [[buffer(1)]],
                                           device const float* secondDerivatives [[buffer(2)]],
                                           device const float* queryX [[buffer(3)]],
                                           device float* result [[buffer(4)]],
                                           constant uint& knotCount [[buffer(5)]],
                                           constant uint& count [[buffer(6)]],
                                           uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    float at = queryX[tid];
    if (at <= knotsX[0]) {
        result[tid] = knotsY[0];
        return;
    }
    if (knotsX[knotCount - 1] <= at) {
        result[tid] = knotsY[knotCount - 1];
        return;
    }

    uint lo = 0;
    uint hi = knotCount;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (knotsX[mid] <= at) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    uint i = (lo == 0) ? 0 : lo - 1;
    i = min(i, knotCount - 2);

    float width = knotsX[i + 1] - knotsX[i];
    float left = (knotsX[i + 1] - at) / width;
    float right = (at - knotsX[i]) / width;

    result[tid] = left * knotsY[i] + right * knotsY[i + 1] +
                 ((left * left * left - left) * secondDerivatives[i] +
                  (right * right * right - right) * secondDerivatives[i + 1]) *
                     (width * width) / 6.0f;
}

// Philox4x32-10 (Salmon, Moraes, Hadjidoukas & Schulten 2011): a
// counter-based RNG, embarrassingly parallel by construction (thread tid's
// output depends only on (seed, offset + tid), no shared state, no
// sequential dependency between threads). mulhi32 computes the high 32
// bits of an unsigned 32x32 multiply using only 16-bit-limb schoolbook
// long multiplication rather than a 64-bit intermediate, so the identical
// source works unchanged in GLSL, which has no portable 64-bit integer
// type at the GLSL 430 this engine targets; the low 32 bits are just an
// ordinary `*`, which wraps to the right value on its own.
uint mulhi32(uint a, uint b) {
    uint a0 = a & 0xffffu;
    uint a1 = a >> 16u;
    uint b0 = b & 0xffffu;
    uint b1 = b >> 16u;

    uint t = a0 * b0;
    uint k = t >> 16u;

    t = a1 * b0 + k;
    uint w1 = t & 0xffffu;
    uint w2 = t >> 16u;

    t = a0 * b1 + w1;
    k = t >> 16u;

    return a1 * b1 + w2 + k;
}

void philox4x32_10(thread uint& c0, thread uint& c1, thread uint& c2, thread uint& c3,
                   uint seedLo, uint seedHi) {
    uint k0 = seedLo;
    uint k1 = seedHi;
    for (int round = 0; round < 10; ++round) {
        uint m0 = 0xD2511F53u;
        uint m1 = 0xCD9E8D57u;
        uint hi0 = mulhi32(m0, c0);
        uint lo0 = m0 * c0;
        uint hi1 = mulhi32(m1, c2);
        uint lo1 = m1 * c2;
        uint nextC0 = hi1 ^ c1 ^ k0;
        uint nextC1 = lo1;
        uint nextC2 = hi0 ^ c3 ^ k1;
        uint nextC3 = lo0;
        c0 = nextC0;
        c1 = nextC1;
        c2 = nextC2;
        c3 = nextC3;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
}

constant float kOneOverTwoToThe32 = 2.3283064365386963e-10f;
constant float kTwoPi = 6.28318530717958647692f;

// counter = offset + tid, as a 64-bit value split into two 32-bit words
// with an explicit carry, matching CpuBackend's std::uint64_t addition
// exactly (the standard unsigned-overflow idiom: the sum wrapped iff it
// came out smaller than either original operand).
kernel void batch_uniform_real_kernel(device float* result [[buffer(0)]],
                                      constant uint& count [[buffer(1)]],
                                      constant uint& seedLo [[buffer(2)]],
                                      constant uint& seedHi [[buffer(3)]],
                                      constant uint& offsetLo [[buffer(4)]],
                                      constant uint& offsetHi [[buffer(5)]],
                                      uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    uint c0 = offsetLo + tid;
    uint carry = (c0 < offsetLo) ? 1u : 0u;
    uint c1 = offsetHi + carry;
    uint c2 = 0u;
    uint c3 = 0u;
    philox4x32_10(c0, c1, c2, c3, seedLo, seedHi);
    result[tid] = float(c0) * kOneOverTwoToThe32;
}

kernel void batch_normal_kernel(device float* result [[buffer(0)]],
                                constant uint& count [[buffer(1)]],
                                constant uint& seedLo [[buffer(2)]],
                                constant uint& seedHi [[buffer(3)]],
                                constant uint& offsetLo [[buffer(4)]],
                                constant uint& offsetHi [[buffer(5)]],
                                uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    uint c0 = offsetLo + tid;
    uint carry = (c0 < offsetLo) ? 1u : 0u;
    uint c1 = offsetHi + carry;
    uint c2 = 0u;
    uint c3 = 0u;
    philox4x32_10(c0, c1, c2, c3, seedLo, seedHi);
    // u1 in (0, 1], never exactly 0, so log(u1) is always finite: offsetting
    // the 32-bit word up by one before scaling avoids a special case for the
    // all-zero output word instead, matching CpuBackend::batchNormal exactly.
    float u1 = (float(c0) + 1.0f) * kOneOverTwoToThe32;
    float u2 = float(c1) * kOneOverTwoToThe32;
    result[tid] = sqrt(-2.0f * log(u1)) * cos(kTwoPi * u2);
}

// Bitonic sort (Batcher 1968): the standard data-parallel sorting network.
// One compare-exchange pass per dispatch, `log2(n) * (log2(n) + 1) / 2`
// dispatches total (MetalBackend::sortAscending's own host loop), `n` a
// power of two -- the caller pads with +infinity and truncates back, see
// ComputeBackend.hpp's own comment on sortAscending for why that's exact
// rather than approximate, unlike zero-padding an FFT. `stageSize` is the
// size of the bitonic sequence the current major stage is building
// (2, 4, 8, ..., n); `stepSize` is the current compare distance within
// that stage (stageSize/2, stageSize/4, ..., 1). Thread tid only acts when
// tid is the lower of its compare-exchange pair (tid < partner), and
// ascending/descending direction alternates in blocks of stageSize,
// exactly Batcher's construction.
kernel void bitonic_compare_exchange_kernel(device float* values [[buffer(0)]],
                                            constant uint& n [[buffer(1)]],
                                            constant uint& stageSize [[buffer(2)]],
                                            constant uint& stepSize [[buffer(3)]],
                                            uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    uint partner = tid ^ stepSize;
    if (partner <= tid || partner >= n) return;
    bool ascending = (tid & stageSize) == 0u;
    float a = values[tid];
    float b = values[partner];
    if ((a > b) == ascending) {
        values[tid] = b;
        values[partner] = a;
    }
}

// Geometric-multigrid transfer operators. Equation-independent: matches
// Math/Multigrid.hpp's detail::restrictGrid/prolongateAndAdd exactly, one
// thread per coarse (restrict) or fine (prolongate) cell.
kernel void multigrid_restrict_3d_kernel(device const float* fine [[buffer(0)]],
                                         device float* coarse [[buffer(1)]],
                                         constant int& nx [[buffer(2)]],
                                         constant int& ny [[buffer(3)]],
                                         constant int& nz [[buffer(4)]],
                                         uint tid [[thread_position_in_grid]]) {
    int cnx = nx / 2;
    int cny = ny / 2;
    int cnz = nz / 2;
    int idx = int(tid);
    int total = cnx * cny * cnz;
    if (idx >= total) return;
    int ck = idx % cnz;
    int cj = (idx / cnz) % cny;
    int ci = idx / (cny * cnz);
    float sum = 0.0f;
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                int fi = 2 * ci + di;
                int fj = 2 * cj + dj;
                int fk = 2 * ck + dk;
                sum += fine[(fi * ny + fj) * nz + fk];
            }
        }
    }
    coarse[idx] = sum / 8.0f;
}

kernel void multigrid_prolongate_and_add_3d_kernel(
        device const float* fine [[buffer(0)]],
        device const float* coarseCorrection [[buffer(1)]],
        device float* nextFine [[buffer(2)]],
        constant int& nx [[buffer(3)]],
        constant int& ny [[buffer(4)]],
        constant int& nz [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
    int idx = int(tid);
    int total = nx * ny * nz;
    if (idx >= total) return;
    int cny = ny / 2;
    int cnz = nz / 2;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    int coarseIdx = ((i / 2) * cny + (j / 2)) * cnz + (k / 2);
    nextFine[idx] = fine[idx] + coarseCorrection[coarseIdx];
}

// Same two-pass shape as sum_kernel, tracking (value, index) pairs: a tie
// within a threadgroup keeps whichever came from the lower local index
// (strict '<' never replaces on equality), and MetalBackend::minIndex
// applies the same rule across threadgroups when finishing on the CPU.
kernel void min_index_kernel(device const float* data [[buffer(0)]],
                             device float* partialValues [[buffer(1)]],
                             device float* partialIndices [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             uint i [[thread_position_in_grid]],
                             uint local [[thread_position_in_threadgroup]],
                             uint groupId [[threadgroup_position_in_grid]],
                             threadgroup float* scratchValue [[threadgroup(0)]],
                             threadgroup uint* scratchIndex [[threadgroup(1)]]) {
    if (i < count) {
        scratchValue[local] = data[i];
        scratchIndex[local] = i;
    } else {
        scratchValue[local] = INFINITY;
        scratchIndex[local] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local < stride && scratchValue[local + stride] < scratchValue[local]) {
            scratchValue[local] = scratchValue[local + stride];
            scratchIndex[local] = scratchIndex[local + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local == 0) {
        partialValues[groupId] = scratchValue[0];
        partialIndices[groupId] = float(scratchIndex[0]);
    }
}
)msl";

NSUInteger groupCountFor(std::size_t elementCount) {
    return (static_cast<NSUInteger>(elementCount) + kThreadgroupSize - 1) /
           kThreadgroupSize;
}

}  // namespace

struct MetalBackend::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> saxpyPipeline = nil;
    id<MTLComputePipelineState> sumPipeline = nil;
    id<MTLComputePipelineState> linearCombinePipeline = nil;
    id<MTLComputePipelineState> gravitationalNBodyPipeline = nil;
    id<MTLComputePipelineState> electricFieldNBodyPipeline = nil;
    id<MTLComputePipelineState> magneticFieldNBodyPipeline = nil;
    id<MTLComputePipelineState> sphDensityPressurePipeline = nil;
    id<MTLComputePipelineState> sphPressureAccelerationPipeline = nil;
    id<MTLComputePipelineState> heatEquation3DStepPipeline = nil;
    id<MTLComputePipelineState> acoustic3DVelocityPipeline = nil;
    id<MTLComputePipelineState> acoustic3DPressurePipeline = nil;
    id<MTLComputePipelineState> maxwell3DBPipeline = nil;
    id<MTLComputePipelineState> maxwell3DEPipeline = nil;
    id<MTLComputePipelineState> eulerianFluid3DSweepPipeline = nil;
    id<MTLComputePipelineState> fftBitReversalPermutePipeline = nil;
    id<MTLComputePipelineState> fftButterflyStagePipeline = nil;
    id<MTLComputePipelineState> fftScalePipeline = nil;
    id<MTLComputePipelineState> matVecPipeline = nil;
    id<MTLComputePipelineState> matMulPipeline = nil;
    id<MTLComputePipelineState> argMaxAbsColumnPipeline = nil;
    id<MTLComputePipelineState> swapRowsPipeline = nil;
    id<MTLComputePipelineState> luEliminationStepPipeline = nil;
    id<MTLComputePipelineState> choleskyDiagonalPipeline = nil;
    id<MTLComputePipelineState> choleskyColumnPipeline = nil;
    id<MTLComputePipelineState> qrColumnNormSquaredPipeline = nil;
    id<MTLComputePipelineState> qrApplyLeftPipeline = nil;
    id<MTLComputePipelineState> qrAccumulateQPipeline = nil;
    id<MTLComputePipelineState> jacobiEigenColumnMixPipeline = nil;
    id<MTLComputePipelineState> jacobiEigenRowMixPipeline = nil;
    id<MTLComputePipelineState> jacobiSvdRoundPipeline = nil;
    id<MTLComputePipelineState> batchErfPipeline = nil;
    id<MTLComputePipelineState> batchErfcPipeline = nil;
    id<MTLComputePipelineState> batchGammaPipeline = nil;
    id<MTLComputePipelineState> batchLogGammaPipeline = nil;
    id<MTLComputePipelineState> batchLegendrePPipeline = nil;
    id<MTLComputePipelineState> batchPolynomialEvalPipeline = nil;
    id<MTLComputePipelineState> batchCubicSplineEvalPipeline = nil;
    id<MTLComputePipelineState> batchUniformRealPipeline = nil;
    id<MTLComputePipelineState> batchNormalPipeline = nil;
    id<MTLComputePipelineState> bitonicCompareExchangePipeline = nil;
    id<MTLComputePipelineState> multigridRestrict3DPipeline = nil;
    id<MTLComputePipelineState> multigridProlongateAndAdd3DPipeline = nil;
    id<MTLComputePipelineState> minIndexPipeline = nil;
};

std::unique_ptr<ComputeBackend> MetalBackend::create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            logging::debug("Metal compute backend unavailable: no default device");
            return nullptr;
        }

        NSError* error = nil;
        const std::string fullKernelSource =
            std::string(kKernelSourcePart1) + std::string(kKernelSourcePart2);
        NSString* source = [NSString stringWithUTF8String:fullKernelSource.c_str()];
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:nil
                                                        error:&error];
        if (!library) {
            logging::debug(
                "Metal compute backend unavailable: library compile failed: {}",
                error ? std::string{[[error localizedDescription] UTF8String]}
                      : "unknown error");
            return nullptr;
        }

        auto makePipeline = [&](NSString* functionName) -> id<MTLComputePipelineState> {
            id<MTLFunction> function = [library newFunctionWithName:functionName];
            if (!function) {
                logging::debug("Metal compute backend unavailable: no function named {}",
                               std::string{[functionName UTF8String]});
                return nil;
            }
            NSError* pipelineError = nil;
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function
                                                      error:&pipelineError];
            if (!pipeline) {
                logging::debug(
                    "Metal compute backend unavailable: pipeline for {} failed: {}",
                    std::string{[functionName UTF8String]},
                    pipelineError
                        ? std::string{[[pipelineError localizedDescription] UTF8String]}
                        : "unknown error");
            }
            return pipeline;
        };

        auto impl = std::make_unique<Impl>();
        impl->device = device;
        impl->queue = [device newCommandQueue];
        impl->saxpyPipeline = makePipeline(@"saxpy_kernel");
        impl->sumPipeline = makePipeline(@"sum_kernel");
        impl->linearCombinePipeline = makePipeline(@"linear_combine_kernel");
        impl->gravitationalNBodyPipeline = makePipeline(@"gravitational_nbody_kernel");
        impl->electricFieldNBodyPipeline = makePipeline(@"electric_field_nbody_kernel");
        impl->magneticFieldNBodyPipeline = makePipeline(@"magnetic_field_nbody_kernel");
        impl->sphDensityPressurePipeline = makePipeline(@"sph_density_pressure_kernel");
        impl->sphPressureAccelerationPipeline =
            makePipeline(@"sph_pressure_acceleration_kernel");
        impl->heatEquation3DStepPipeline = makePipeline(@"heat_equation_3d_step_kernel");
        impl->acoustic3DVelocityPipeline = makePipeline(@"acoustic_3d_velocity_kernel");
        impl->acoustic3DPressurePipeline = makePipeline(@"acoustic_3d_pressure_kernel");
        impl->maxwell3DBPipeline = makePipeline(@"maxwell_3d_b_kernel");
        impl->maxwell3DEPipeline = makePipeline(@"maxwell_3d_e_kernel");
        impl->eulerianFluid3DSweepPipeline =
            makePipeline(@"eulerian_fluid_3d_sweep_kernel");
        impl->fftBitReversalPermutePipeline =
            makePipeline(@"fft_bit_reversal_permute_kernel");
        impl->fftButterflyStagePipeline = makePipeline(@"fft_butterfly_stage_kernel");
        impl->fftScalePipeline = makePipeline(@"fft_scale_kernel");
        impl->matVecPipeline = makePipeline(@"mat_vec_kernel");
        impl->matMulPipeline = makePipeline(@"mat_mul_kernel");
        impl->argMaxAbsColumnPipeline = makePipeline(@"arg_max_abs_column_kernel");
        impl->swapRowsPipeline = makePipeline(@"swap_rows_kernel");
        impl->luEliminationStepPipeline = makePipeline(@"lu_elimination_step_kernel");
        impl->choleskyDiagonalPipeline = makePipeline(@"cholesky_diagonal_kernel");
        impl->choleskyColumnPipeline = makePipeline(@"cholesky_column_kernel");
        impl->qrColumnNormSquaredPipeline =
            makePipeline(@"qr_column_norm_squared_kernel");
        impl->qrApplyLeftPipeline = makePipeline(@"qr_apply_left_kernel");
        impl->qrAccumulateQPipeline = makePipeline(@"qr_accumulate_q_kernel");
        impl->jacobiEigenColumnMixPipeline =
            makePipeline(@"jacobi_eigen_column_mix_kernel");
        impl->jacobiEigenRowMixPipeline = makePipeline(@"jacobi_eigen_row_mix_kernel");
        impl->jacobiSvdRoundPipeline = makePipeline(@"jacobi_svd_round_kernel");
        impl->batchErfPipeline = makePipeline(@"batch_erf_kernel");
        impl->batchErfcPipeline = makePipeline(@"batch_erfc_kernel");
        impl->batchGammaPipeline = makePipeline(@"batch_gamma_kernel");
        impl->batchLogGammaPipeline = makePipeline(@"batch_log_gamma_kernel");
        impl->batchLegendrePPipeline = makePipeline(@"batch_legendre_p_kernel");
        impl->batchPolynomialEvalPipeline = makePipeline(@"batch_polynomial_eval_kernel");
        impl->batchCubicSplineEvalPipeline =
            makePipeline(@"batch_cubic_spline_eval_kernel");
        impl->batchUniformRealPipeline = makePipeline(@"batch_uniform_real_kernel");
        impl->batchNormalPipeline = makePipeline(@"batch_normal_kernel");
        impl->bitonicCompareExchangePipeline =
            makePipeline(@"bitonic_compare_exchange_kernel");
        impl->multigridRestrict3DPipeline = makePipeline(@"multigrid_restrict_3d_kernel");
        impl->multigridProlongateAndAdd3DPipeline =
            makePipeline(@"multigrid_prolongate_and_add_3d_kernel");
        impl->minIndexPipeline = makePipeline(@"min_index_kernel");

        if (!impl->queue || !impl->saxpyPipeline || !impl->sumPipeline ||
            !impl->linearCombinePipeline || !impl->gravitationalNBodyPipeline ||
            !impl->electricFieldNBodyPipeline || !impl->magneticFieldNBodyPipeline ||
            !impl->sphDensityPressurePipeline || !impl->sphPressureAccelerationPipeline ||
            !impl->heatEquation3DStepPipeline || !impl->acoustic3DVelocityPipeline ||
            !impl->acoustic3DPressurePipeline || !impl->maxwell3DBPipeline ||
            !impl->maxwell3DEPipeline || !impl->eulerianFluid3DSweepPipeline ||
            !impl->fftBitReversalPermutePipeline || !impl->fftButterflyStagePipeline ||
            !impl->fftScalePipeline || !impl->matVecPipeline || !impl->matMulPipeline ||
            !impl->argMaxAbsColumnPipeline || !impl->swapRowsPipeline ||
            !impl->luEliminationStepPipeline || !impl->choleskyDiagonalPipeline ||
            !impl->choleskyColumnPipeline || !impl->qrColumnNormSquaredPipeline ||
            !impl->qrApplyLeftPipeline || !impl->qrAccumulateQPipeline ||
            !impl->jacobiEigenColumnMixPipeline || !impl->jacobiEigenRowMixPipeline ||
            !impl->jacobiSvdRoundPipeline || !impl->batchErfPipeline ||
            !impl->batchErfcPipeline || !impl->batchGammaPipeline ||
            !impl->batchLogGammaPipeline || !impl->batchLegendrePPipeline ||
            !impl->batchPolynomialEvalPipeline || !impl->batchCubicSplineEvalPipeline ||
            !impl->batchUniformRealPipeline || !impl->batchNormalPipeline ||
            !impl->bitonicCompareExchangePipeline || !impl->multigridRestrict3DPipeline ||
            !impl->multigridProlongateAndAdd3DPipeline || !impl->minIndexPipeline) {
            return nullptr;
        }

        return std::unique_ptr<ComputeBackend>{new MetalBackend(std::move(impl))};
    }
}

MetalBackend::MetalBackend(std::unique_ptr<Impl> impl) noexcept
    : m_impl(std::move(impl)) {}

MetalBackend::~MetalBackend() = default;

void MetalBackend::saxpy(std::span<const float> x, std::span<float> y, float a) const {
    assert(x.size() == y.size() && "saxpy needs matching spans");
    if (y.empty()) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(y.size() * sizeof(float));

        id<MTLBuffer> xBuffer = [device newBufferWithBytes:x.data()
                                                    length:byteCount
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> yBuffer = [device newBufferWithBytes:y.data()
                                                    length:byteCount
                                                   options:MTLResourceStorageModeShared];
        const auto count = static_cast<std::uint32_t>(y.size());

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->saxpyPipeline];
        [encoder setBuffer:xBuffer offset:0 atIndex:0];
        [encoder setBuffer:yBuffer offset:0 atIndex:1];
        [encoder setBytes:&a length:sizeof(a) atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(y.size()), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(y.data(), [yBuffer contents], byteCount);
    }
}

float MetalBackend::sum(std::span<const float> x) const {
    if (x.empty()) {
        return 0.0f;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto groups = static_cast<std::size_t>(groupCountFor(x.size()));
        const auto inBytes = static_cast<NSUInteger>(x.size() * sizeof(float));
        const auto outBytes = static_cast<NSUInteger>(groups * sizeof(float));

        id<MTLBuffer> inBuffer = [device newBufferWithBytes:x.data()
                                                     length:inBytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuffer =
            [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
        const auto count = static_cast<std::uint32_t>(x.size());

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->sumPipeline];
        [encoder setBuffer:inBuffer offset:0 atIndex:0];
        [encoder setBuffer:outBuffer offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setThreadgroupMemoryLength:(kThreadgroupSize * sizeof(float)) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(groups), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::vector<float> partials(groups);
        std::memcpy(partials.data(), [outBuffer contents], outBytes);

        // One partial per threadgroup, so finishing on the CPU does not earn
        // a third kernel; matches OpenGLBackend::sum's own convention.
        float total = 0.0f;
        for (const float partial : partials) {
            total += partial;
        }
        return total;
    }
}

void MetalBackend::linearCombine(std::span<const std::span<const float>> terms,
                                 std::span<const float> coefficients,
                                 std::span<float> y) const {
    assert(terms.size() == coefficients.size() && "one coefficient per term");
    assert(terms.size() >= 1 && terms.size() <= 4 &&
           "linearCombine takes one to four terms");
    if (y.empty()) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(y.size() * sizeof(float));

        id<MTLBuffer> yBuffer = [device newBufferWithLength:byteCount
                                                    options:MTLResourceStorageModeShared];

        std::array<id<MTLBuffer>, 4> termBuffers{nil, nil, nil, nil};
        for (std::size_t k = 0; k < terms.size(); ++k) {
            assert(terms[k].size() == y.size() && "every term must match y's length");
            termBuffers[k] = [device newBufferWithBytes:terms[k].data()
                                                 length:byteCount
                                                options:MTLResourceStorageModeShared];
        }
        // Slots past terms.size() are bound to y's own (not-yet-written)
        // buffer purely to give the binding point something legal; termCount
        // gates the kernel body so they are never actually read.
        for (std::size_t k = terms.size(); k < 4; ++k) {
            termBuffers[k] = yBuffer;
        }

        std::array<float, 4> coefficientSlots{0.0f, 0.0f, 0.0f, 0.0f};
        for (std::size_t k = 0; k < coefficients.size(); ++k) {
            coefficientSlots[k] = coefficients[k];
        }
        const auto count = static_cast<std::uint32_t>(y.size());
        const auto termCount = static_cast<std::int32_t>(terms.size());

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->linearCombinePipeline];
        for (std::size_t k = 0; k < 4; ++k) {
            [encoder setBuffer:termBuffers[k] offset:0 atIndex:k];
        }
        [encoder setBuffer:yBuffer offset:0 atIndex:4];
        [encoder setBytes:&count length:sizeof(count) atIndex:5];
        [encoder setBytes:&termCount length:sizeof(termCount) atIndex:6];
        [encoder setBytes:coefficientSlots.data() length:sizeof(float) * 4 atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(y.size()), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(y.data(), [yBuffer contents], byteCount);
    }
}

void MetalBackend::gravitationalNBody(std::span<const float> positionsX,
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
    if (n == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(n * sizeof(float));

        const auto makeInput = [&](std::span<const float> data) {
            return [device newBufferWithBytes:data.data()
                                       length:byteCount
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> posXBuffer = makeInput(positionsX);
        id<MTLBuffer> posYBuffer = makeInput(positionsY);
        id<MTLBuffer> posZBuffer = makeInput(positionsZ);
        id<MTLBuffer> gmBuffer = makeInput(gm);
        id<MTLBuffer> accXBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        id<MTLBuffer> accYBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        id<MTLBuffer> accZBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        const auto count = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->gravitationalNBodyPipeline];
        [encoder setBuffer:posXBuffer offset:0 atIndex:0];
        [encoder setBuffer:posYBuffer offset:0 atIndex:1];
        [encoder setBuffer:posZBuffer offset:0 atIndex:2];
        [encoder setBuffer:gmBuffer offset:0 atIndex:3];
        [encoder setBuffer:accXBuffer offset:0 atIndex:4];
        [encoder setBuffer:accYBuffer offset:0 atIndex:5];
        [encoder setBuffer:accZBuffer offset:0 atIndex:6];
        [encoder setBytes:&count length:sizeof(count) atIndex:7];
        [encoder setBytes:&softeningSquared length:sizeof(softeningSquared) atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(accelerationsX.data(), [accXBuffer contents], byteCount);
        std::memcpy(accelerationsY.data(), [accYBuffer contents], byteCount);
        std::memcpy(accelerationsZ.data(), [accZBuffer contents], byteCount);
    }
}

void MetalBackend::electricFieldNBody(std::span<const float> positionsX,
                                      std::span<const float> positionsY,
                                      std::span<const float> positionsZ,
                                      std::span<const float> charge,
                                      float coulombConstant, std::span<float> fieldX,
                                      std::span<float> fieldY,
                                      std::span<float> fieldZ) const {
    const std::size_t n = positionsX.size();
    if (n == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(n * sizeof(float));

        const auto makeInput = [&](std::span<const float> data) {
            return [device newBufferWithBytes:data.data()
                                       length:byteCount
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> posXBuffer = makeInput(positionsX);
        id<MTLBuffer> posYBuffer = makeInput(positionsY);
        id<MTLBuffer> posZBuffer = makeInput(positionsZ);
        id<MTLBuffer> chargeBuffer = makeInput(charge);
        const auto makeOutput = [&] {
            return [device newBufferWithLength:byteCount
                                       options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> fieldXBuffer = makeOutput();
        id<MTLBuffer> fieldYBuffer = makeOutput();
        id<MTLBuffer> fieldZBuffer = makeOutput();
        const auto count = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->electricFieldNBodyPipeline];
        [encoder setBuffer:posXBuffer offset:0 atIndex:0];
        [encoder setBuffer:posYBuffer offset:0 atIndex:1];
        [encoder setBuffer:posZBuffer offset:0 atIndex:2];
        [encoder setBuffer:chargeBuffer offset:0 atIndex:3];
        [encoder setBuffer:fieldXBuffer offset:0 atIndex:4];
        [encoder setBuffer:fieldYBuffer offset:0 atIndex:5];
        [encoder setBuffer:fieldZBuffer offset:0 atIndex:6];
        [encoder setBytes:&count length:sizeof(count) atIndex:7];
        [encoder setBytes:&coulombConstant length:sizeof(coulombConstant) atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(fieldX.data(), [fieldXBuffer contents], byteCount);
        std::memcpy(fieldY.data(), [fieldYBuffer contents], byteCount);
        std::memcpy(fieldZ.data(), [fieldZBuffer contents], byteCount);
    }
}

void MetalBackend::magneticFieldNBody(
    std::span<const float> positionsX, std::span<const float> positionsY,
    std::span<const float> positionsZ, std::span<const float> velocitiesX,
    std::span<const float> velocitiesY, std::span<const float> velocitiesZ,
    std::span<const float> charge, float permeabilityOver4Pi, std::span<float> fieldX,
    std::span<float> fieldY, std::span<float> fieldZ) const {
    const std::size_t n = positionsX.size();
    if (n == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(n * sizeof(float));

        const auto makeInput = [&](std::span<const float> data) {
            return [device newBufferWithBytes:data.data()
                                       length:byteCount
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> posXBuffer = makeInput(positionsX);
        id<MTLBuffer> posYBuffer = makeInput(positionsY);
        id<MTLBuffer> posZBuffer = makeInput(positionsZ);
        id<MTLBuffer> velXBuffer = makeInput(velocitiesX);
        id<MTLBuffer> velYBuffer = makeInput(velocitiesY);
        id<MTLBuffer> velZBuffer = makeInput(velocitiesZ);
        id<MTLBuffer> chargeBuffer = makeInput(charge);
        const auto makeOutput = [&] {
            return [device newBufferWithLength:byteCount
                                       options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> fieldXBuffer = makeOutput();
        id<MTLBuffer> fieldYBuffer = makeOutput();
        id<MTLBuffer> fieldZBuffer = makeOutput();
        const auto count = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->magneticFieldNBodyPipeline];
        [encoder setBuffer:posXBuffer offset:0 atIndex:0];
        [encoder setBuffer:posYBuffer offset:0 atIndex:1];
        [encoder setBuffer:posZBuffer offset:0 atIndex:2];
        [encoder setBuffer:velXBuffer offset:0 atIndex:3];
        [encoder setBuffer:velYBuffer offset:0 atIndex:4];
        [encoder setBuffer:velZBuffer offset:0 atIndex:5];
        [encoder setBuffer:chargeBuffer offset:0 atIndex:6];
        [encoder setBuffer:fieldXBuffer offset:0 atIndex:7];
        [encoder setBuffer:fieldYBuffer offset:0 atIndex:8];
        [encoder setBuffer:fieldZBuffer offset:0 atIndex:9];
        [encoder setBytes:&count length:sizeof(count) atIndex:10];
        [encoder setBytes:&permeabilityOver4Pi
                   length:sizeof(permeabilityOver4Pi)
                  atIndex:11];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(fieldX.data(), [fieldXBuffer contents], byteCount);
        std::memcpy(fieldY.data(), [fieldYBuffer contents], byteCount);
        std::memcpy(fieldZ.data(), [fieldZBuffer contents], byteCount);
    }
}

void MetalBackend::sphDensityPressure(std::span<const float> positionsX,
                                      std::span<const float> positionsY,
                                      std::span<const float> positionsZ,
                                      std::span<const float> mass, float smoothingLength,
                                      float equationOfStateK, float polytropicIndex,
                                      std::span<float> density,
                                      std::span<float> pressure) const {
    const std::size_t n = positionsX.size();
    if (n == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(n * sizeof(float));

        const auto makeInput = [&](std::span<const float> data) {
            return [device newBufferWithBytes:data.data()
                                       length:byteCount
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> posXBuffer = makeInput(positionsX);
        id<MTLBuffer> posYBuffer = makeInput(positionsY);
        id<MTLBuffer> posZBuffer = makeInput(positionsZ);
        id<MTLBuffer> massBuffer = makeInput(mass);
        id<MTLBuffer> densityBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        id<MTLBuffer> pressureBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        const auto count = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->sphDensityPressurePipeline];
        [encoder setBuffer:posXBuffer offset:0 atIndex:0];
        [encoder setBuffer:posYBuffer offset:0 atIndex:1];
        [encoder setBuffer:posZBuffer offset:0 atIndex:2];
        [encoder setBuffer:massBuffer offset:0 atIndex:3];
        [encoder setBuffer:densityBuffer offset:0 atIndex:4];
        [encoder setBuffer:pressureBuffer offset:0 atIndex:5];
        [encoder setBytes:&count length:sizeof(count) atIndex:6];
        [encoder setBytes:&smoothingLength length:sizeof(smoothingLength) atIndex:7];
        [encoder setBytes:&equationOfStateK length:sizeof(equationOfStateK) atIndex:8];
        [encoder setBytes:&polytropicIndex length:sizeof(polytropicIndex) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(density.data(), [densityBuffer contents], byteCount);
        std::memcpy(pressure.data(), [pressureBuffer contents], byteCount);
    }
}

void MetalBackend::sphPressureAcceleration(
    std::span<const float> positionsX, std::span<const float> positionsY,
    std::span<const float> positionsZ, std::span<const float> mass,
    std::span<const float> density, std::span<const float> pressure,
    float smoothingLength, std::span<float> accelerationsX,
    std::span<float> accelerationsY, std::span<float> accelerationsZ) const {
    const std::size_t n = positionsX.size();
    if (n == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(n * sizeof(float));

        const auto makeInput = [&](std::span<const float> data) {
            return [device newBufferWithBytes:data.data()
                                       length:byteCount
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> posXBuffer = makeInput(positionsX);
        id<MTLBuffer> posYBuffer = makeInput(positionsY);
        id<MTLBuffer> posZBuffer = makeInput(positionsZ);
        id<MTLBuffer> massBuffer = makeInput(mass);
        id<MTLBuffer> densityBuffer = makeInput(density);
        id<MTLBuffer> pressureBuffer = makeInput(pressure);
        const auto makeOutput = [&] {
            return [device newBufferWithLength:byteCount
                                       options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> accXBuffer = makeOutput();
        id<MTLBuffer> accYBuffer = makeOutput();
        id<MTLBuffer> accZBuffer = makeOutput();
        const auto count = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->sphPressureAccelerationPipeline];
        [encoder setBuffer:posXBuffer offset:0 atIndex:0];
        [encoder setBuffer:posYBuffer offset:0 atIndex:1];
        [encoder setBuffer:posZBuffer offset:0 atIndex:2];
        [encoder setBuffer:massBuffer offset:0 atIndex:3];
        [encoder setBuffer:densityBuffer offset:0 atIndex:4];
        [encoder setBuffer:pressureBuffer offset:0 atIndex:5];
        [encoder setBuffer:accXBuffer offset:0 atIndex:6];
        [encoder setBuffer:accYBuffer offset:0 atIndex:7];
        [encoder setBuffer:accZBuffer offset:0 atIndex:8];
        [encoder setBytes:&count length:sizeof(count) atIndex:9];
        [encoder setBytes:&smoothingLength length:sizeof(smoothingLength) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(accelerationsX.data(), [accXBuffer contents], byteCount);
        std::memcpy(accelerationsY.data(), [accYBuffer contents], byteCount);
        std::memcpy(accelerationsZ.data(), [accZBuffer contents], byteCount);
    }
}

void MetalBackend::heatEquation3DStep(std::span<const float> temperature, std::size_t nx,
                                      std::size_t ny, std::size_t nz, float factor,
                                      std::span<float> next) const {
    const std::size_t total = nx * ny * nz;
    if (total == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(total * sizeof(float));

        id<MTLBuffer> tempBuffer =
            [device newBufferWithBytes:temperature.data()
                                length:byteCount
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> nextBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        const auto nxInt = static_cast<std::int32_t>(nx);
        const auto nyInt = static_cast<std::int32_t>(ny);
        const auto nzInt = static_cast<std::int32_t>(nz);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->heatEquation3DStepPipeline];
        [encoder setBuffer:tempBuffer offset:0 atIndex:0];
        [encoder setBuffer:nextBuffer offset:0 atIndex:1];
        [encoder setBytes:&nxInt length:sizeof(nxInt) atIndex:2];
        [encoder setBytes:&nyInt length:sizeof(nyInt) atIndex:3];
        [encoder setBytes:&nzInt length:sizeof(nzInt) atIndex:4];
        [encoder setBytes:&factor length:sizeof(factor) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(next.data(), [nextBuffer contents], byteCount);
    }
}

void MetalBackend::acoustic3DStep(
    std::span<const float> pressure, std::span<const float> velocityX,
    std::span<const float> velocityY, std::span<const float> velocityZ, std::size_t nx,
    std::size_t ny, std::size_t nz, float velocityFactor, float pressureFactor,
    std::span<float> nextPressure, std::span<float> nextVelocityX,
    std::span<float> nextVelocityY, std::span<float> nextVelocityZ) const {
    const std::size_t total = nx * ny * nz;
    if (total == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(total * sizeof(float));

        const auto makeInput = [&](std::span<const float> data) {
            return [device newBufferWithBytes:data.data()
                                       length:byteCount
                                      options:MTLResourceStorageModeShared];
        };
        const auto makeOutput = [&] {
            return [device newBufferWithLength:byteCount
                                       options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> pressureBuffer = makeInput(pressure);
        id<MTLBuffer> velXInBuffer = makeInput(velocityX);
        id<MTLBuffer> velYInBuffer = makeInput(velocityY);
        id<MTLBuffer> velZInBuffer = makeInput(velocityZ);
        id<MTLBuffer> velXOutBuffer = makeOutput();
        id<MTLBuffer> velYOutBuffer = makeOutput();
        id<MTLBuffer> velZOutBuffer = makeOutput();
        const auto nxInt = static_cast<std::int32_t>(nx);
        const auto nyInt = static_cast<std::int32_t>(ny);
        const auto nzInt = static_cast<std::int32_t>(nz);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];

        id<MTLComputeCommandEncoder> velocityEncoder =
            [commandBuffer computeCommandEncoder];
        [velocityEncoder setComputePipelineState:m_impl->acoustic3DVelocityPipeline];
        [velocityEncoder setBuffer:pressureBuffer offset:0 atIndex:0];
        [velocityEncoder setBuffer:velXInBuffer offset:0 atIndex:1];
        [velocityEncoder setBuffer:velYInBuffer offset:0 atIndex:2];
        [velocityEncoder setBuffer:velZInBuffer offset:0 atIndex:3];
        [velocityEncoder setBuffer:velXOutBuffer offset:0 atIndex:4];
        [velocityEncoder setBuffer:velYOutBuffer offset:0 atIndex:5];
        [velocityEncoder setBuffer:velZOutBuffer offset:0 atIndex:6];
        [velocityEncoder setBytes:&nxInt length:sizeof(nxInt) atIndex:7];
        [velocityEncoder setBytes:&nyInt length:sizeof(nyInt) atIndex:8];
        [velocityEncoder setBytes:&nzInt length:sizeof(nzInt) atIndex:9];
        [velocityEncoder setBytes:&velocityFactor
                           length:sizeof(velocityFactor)
                          atIndex:10];
        [velocityEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [velocityEncoder endEncoding];

        id<MTLBuffer> pressureOutBuffer = makeOutput();
        id<MTLComputeCommandEncoder> pressureEncoder =
            [commandBuffer computeCommandEncoder];
        [pressureEncoder setComputePipelineState:m_impl->acoustic3DPressurePipeline];
        [pressureEncoder setBuffer:pressureBuffer offset:0 atIndex:0];
        [pressureEncoder setBuffer:velXOutBuffer offset:0 atIndex:1];
        [pressureEncoder setBuffer:velYOutBuffer offset:0 atIndex:2];
        [pressureEncoder setBuffer:velZOutBuffer offset:0 atIndex:3];
        [pressureEncoder setBuffer:pressureOutBuffer offset:0 atIndex:4];
        [pressureEncoder setBytes:&nxInt length:sizeof(nxInt) atIndex:5];
        [pressureEncoder setBytes:&nyInt length:sizeof(nyInt) atIndex:6];
        [pressureEncoder setBytes:&nzInt length:sizeof(nzInt) atIndex:7];
        [pressureEncoder setBytes:&pressureFactor
                           length:sizeof(pressureFactor)
                          atIndex:8];
        [pressureEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [pressureEncoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(nextVelocityX.data(), [velXOutBuffer contents], byteCount);
        std::memcpy(nextVelocityY.data(), [velYOutBuffer contents], byteCount);
        std::memcpy(nextVelocityZ.data(), [velZOutBuffer contents], byteCount);
        std::memcpy(nextPressure.data(), [pressureOutBuffer contents], byteCount);
    }
}

void MetalBackend::maxwell3DStep(std::span<const float> ex, std::span<const float> ey,
                                 std::span<const float> ez, std::span<const float> bx,
                                 std::span<const float> by, std::span<const float> bz,
                                 std::size_t nx, std::size_t ny, std::size_t nz,
                                 float bFactor, float eFactor, std::span<float> nextEx,
                                 std::span<float> nextEy, std::span<float> nextEz,
                                 std::span<float> nextBx, std::span<float> nextBy,
                                 std::span<float> nextBz) const {
    const std::size_t total = nx * ny * nz;
    if (total == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(total * sizeof(float));

        const auto makeInput = [&](std::span<const float> data) {
            return [device newBufferWithBytes:data.data()
                                       length:byteCount
                                      options:MTLResourceStorageModeShared];
        };
        const auto makeOutput = [&] {
            return [device newBufferWithLength:byteCount
                                       options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> exBuffer = makeInput(ex);
        id<MTLBuffer> eyBuffer = makeInput(ey);
        id<MTLBuffer> ezBuffer = makeInput(ez);
        id<MTLBuffer> bxInBuffer = makeInput(bx);
        id<MTLBuffer> byInBuffer = makeInput(by);
        id<MTLBuffer> bzInBuffer = makeInput(bz);
        id<MTLBuffer> bxOutBuffer = makeOutput();
        id<MTLBuffer> byOutBuffer = makeOutput();
        id<MTLBuffer> bzOutBuffer = makeOutput();
        const auto nxInt = static_cast<std::int32_t>(nx);
        const auto nyInt = static_cast<std::int32_t>(ny);
        const auto nzInt = static_cast<std::int32_t>(nz);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];

        id<MTLComputeCommandEncoder> bEncoder = [commandBuffer computeCommandEncoder];
        [bEncoder setComputePipelineState:m_impl->maxwell3DBPipeline];
        [bEncoder setBuffer:exBuffer offset:0 atIndex:0];
        [bEncoder setBuffer:eyBuffer offset:0 atIndex:1];
        [bEncoder setBuffer:ezBuffer offset:0 atIndex:2];
        [bEncoder setBuffer:bxInBuffer offset:0 atIndex:3];
        [bEncoder setBuffer:byInBuffer offset:0 atIndex:4];
        [bEncoder setBuffer:bzInBuffer offset:0 atIndex:5];
        [bEncoder setBuffer:bxOutBuffer offset:0 atIndex:6];
        [bEncoder setBuffer:byOutBuffer offset:0 atIndex:7];
        [bEncoder setBuffer:bzOutBuffer offset:0 atIndex:8];
        [bEncoder setBytes:&nxInt length:sizeof(nxInt) atIndex:9];
        [bEncoder setBytes:&nyInt length:sizeof(nyInt) atIndex:10];
        [bEncoder setBytes:&nzInt length:sizeof(nzInt) atIndex:11];
        [bEncoder setBytes:&bFactor length:sizeof(bFactor) atIndex:12];
        [bEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [bEncoder endEncoding];

        id<MTLBuffer> exOutBuffer = makeOutput();
        id<MTLBuffer> eyOutBuffer = makeOutput();
        id<MTLBuffer> ezOutBuffer = makeOutput();
        id<MTLComputeCommandEncoder> eEncoder = [commandBuffer computeCommandEncoder];
        [eEncoder setComputePipelineState:m_impl->maxwell3DEPipeline];
        [eEncoder setBuffer:exBuffer offset:0 atIndex:0];
        [eEncoder setBuffer:eyBuffer offset:0 atIndex:1];
        [eEncoder setBuffer:ezBuffer offset:0 atIndex:2];
        [eEncoder setBuffer:bxOutBuffer offset:0 atIndex:3];
        [eEncoder setBuffer:byOutBuffer offset:0 atIndex:4];
        [eEncoder setBuffer:bzOutBuffer offset:0 atIndex:5];
        [eEncoder setBuffer:exOutBuffer offset:0 atIndex:6];
        [eEncoder setBuffer:eyOutBuffer offset:0 atIndex:7];
        [eEncoder setBuffer:ezOutBuffer offset:0 atIndex:8];
        [eEncoder setBytes:&nxInt length:sizeof(nxInt) atIndex:9];
        [eEncoder setBytes:&nyInt length:sizeof(nyInt) atIndex:10];
        [eEncoder setBytes:&nzInt length:sizeof(nzInt) atIndex:11];
        [eEncoder setBytes:&eFactor length:sizeof(eFactor) atIndex:12];
        [eEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [eEncoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(nextBx.data(), [bxOutBuffer contents], byteCount);
        std::memcpy(nextBy.data(), [byOutBuffer contents], byteCount);
        std::memcpy(nextBz.data(), [bzOutBuffer contents], byteCount);
        std::memcpy(nextEx.data(), [exOutBuffer contents], byteCount);
        std::memcpy(nextEy.data(), [eyOutBuffer contents], byteCount);
        std::memcpy(nextEz.data(), [ezOutBuffer contents], byteCount);
    }
}

void MetalBackend::eulerianFluid3DSweep(
    std::span<const float> density, std::span<const float> momentumNormal,
    std::span<const float> momentumTangent1, std::span<const float> momentumTangent2,
    std::span<const float> energy, std::size_t nx, std::size_t ny, std::size_t nz,
    int axis, float gamma, float dtOverSpacing, std::span<float> nextDensity,
    std::span<float> nextMomentumNormal, std::span<float> nextMomentumTangent1,
    std::span<float> nextMomentumTangent2, std::span<float> nextEnergy) const {
    const std::size_t total = nx * ny * nz;
    if (total == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(total * sizeof(float));

        const auto makeInput = [&](std::span<const float> data) {
            return [device newBufferWithBytes:data.data()
                                       length:byteCount
                                      options:MTLResourceStorageModeShared];
        };
        const auto makeOutput = [&] {
            return [device newBufferWithLength:byteCount
                                       options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> densityBuffer = makeInput(density);
        id<MTLBuffer> momentumNormalBuffer = makeInput(momentumNormal);
        id<MTLBuffer> momentumTangent1Buffer = makeInput(momentumTangent1);
        id<MTLBuffer> momentumTangent2Buffer = makeInput(momentumTangent2);
        id<MTLBuffer> energyBuffer = makeInput(energy);
        id<MTLBuffer> nextDensityBuffer = makeOutput();
        id<MTLBuffer> nextMomentumNormalBuffer = makeOutput();
        id<MTLBuffer> nextMomentumTangent1Buffer = makeOutput();
        id<MTLBuffer> nextMomentumTangent2Buffer = makeOutput();
        id<MTLBuffer> nextEnergyBuffer = makeOutput();
        const auto nxInt = static_cast<std::int32_t>(nx);
        const auto nyInt = static_cast<std::int32_t>(ny);
        const auto nzInt = static_cast<std::int32_t>(nz);
        const auto axisInt = static_cast<std::int32_t>(axis);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->eulerianFluid3DSweepPipeline];
        [encoder setBuffer:densityBuffer offset:0 atIndex:0];
        [encoder setBuffer:momentumNormalBuffer offset:0 atIndex:1];
        [encoder setBuffer:momentumTangent1Buffer offset:0 atIndex:2];
        [encoder setBuffer:momentumTangent2Buffer offset:0 atIndex:3];
        [encoder setBuffer:energyBuffer offset:0 atIndex:4];
        [encoder setBuffer:nextDensityBuffer offset:0 atIndex:5];
        [encoder setBuffer:nextMomentumNormalBuffer offset:0 atIndex:6];
        [encoder setBuffer:nextMomentumTangent1Buffer offset:0 atIndex:7];
        [encoder setBuffer:nextMomentumTangent2Buffer offset:0 atIndex:8];
        [encoder setBuffer:nextEnergyBuffer offset:0 atIndex:9];
        [encoder setBytes:&nxInt length:sizeof(nxInt) atIndex:10];
        [encoder setBytes:&nyInt length:sizeof(nyInt) atIndex:11];
        [encoder setBytes:&nzInt length:sizeof(nzInt) atIndex:12];
        [encoder setBytes:&axisInt length:sizeof(axisInt) atIndex:13];
        [encoder setBytes:&gamma length:sizeof(gamma) atIndex:14];
        [encoder setBytes:&dtOverSpacing length:sizeof(dtOverSpacing) atIndex:15];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(nextDensity.data(), [nextDensityBuffer contents], byteCount);
        std::memcpy(nextMomentumNormal.data(), [nextMomentumNormalBuffer contents],
                    byteCount);
        std::memcpy(nextMomentumTangent1.data(), [nextMomentumTangent1Buffer contents],
                    byteCount);
        std::memcpy(nextMomentumTangent2.data(), [nextMomentumTangent2Buffer contents],
                    byteCount);
        std::memcpy(nextEnergy.data(), [nextEnergyBuffer contents], byteCount);
    }
}

void MetalBackend::fftBatched(std::span<const float> real, std::span<const float> imag,
                              std::size_t length, std::size_t batchCount, bool inverse,
                              std::span<float> nextReal,
                              std::span<float> nextImag) const {
    if (length == 0 || batchCount == 0) {
        return;
    }
    const std::size_t total = length * batchCount;
    std::size_t bitCount = 0;
    while ((std::size_t{1} << bitCount) < length) {
        ++bitCount;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(total * sizeof(float));

        id<MTLBuffer> realA = [device newBufferWithBytes:real.data()
                                                  length:byteCount
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> imagA = [device newBufferWithBytes:imag.data()
                                                  length:byteCount
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> realB = [device newBufferWithLength:byteCount
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> imagB = [device newBufferWithLength:byteCount
                                                  options:MTLResourceStorageModeShared];

        const auto lengthU = static_cast<std::uint32_t>(length);
        const auto bitCountU = static_cast<std::uint32_t>(bitCount);
        const auto totalU = static_cast<std::uint32_t>(total);
        const auto pairCountU = static_cast<std::uint32_t>(total / 2);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];

        id<MTLComputeCommandEncoder> permuteEncoder =
            [commandBuffer computeCommandEncoder];
        [permuteEncoder setComputePipelineState:m_impl->fftBitReversalPermutePipeline];
        [permuteEncoder setBuffer:realA offset:0 atIndex:0];
        [permuteEncoder setBuffer:imagA offset:0 atIndex:1];
        [permuteEncoder setBuffer:realB offset:0 atIndex:2];
        [permuteEncoder setBuffer:imagB offset:0 atIndex:3];
        [permuteEncoder setBytes:&lengthU length:sizeof(lengthU) atIndex:4];
        [permuteEncoder setBytes:&bitCountU length:sizeof(bitCountU) atIndex:5];
        [permuteEncoder setBytes:&totalU length:sizeof(totalU) atIndex:6];
        [permuteEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [permuteEncoder endEncoding];

        // Bit-reversal wrote A -> B; each stage below ping-pongs between
        // whichever buffer pair holds the current data and the other,
        // spare pair, the same double-buffering CpuBackend::fftBatched
        // uses on the CPU side.
        id<MTLBuffer> currentReal = realB;
        id<MTLBuffer> currentImag = imagB;
        id<MTLBuffer> spareReal = realA;
        id<MTLBuffer> spareImag = imagA;

        for (std::size_t len = 2; len <= length; len <<= 1) {
            const float angle =
                (inverse ? 1.0f : -1.0f) * 6.283185307179586f / static_cast<float>(len);
            const auto lenU = static_cast<std::uint32_t>(len);

            id<MTLComputeCommandEncoder> stageEncoder =
                [commandBuffer computeCommandEncoder];
            [stageEncoder setComputePipelineState:m_impl->fftButterflyStagePipeline];
            [stageEncoder setBuffer:currentReal offset:0 atIndex:0];
            [stageEncoder setBuffer:currentImag offset:0 atIndex:1];
            [stageEncoder setBuffer:spareReal offset:0 atIndex:2];
            [stageEncoder setBuffer:spareImag offset:0 atIndex:3];
            [stageEncoder setBytes:&lengthU length:sizeof(lengthU) atIndex:4];
            [stageEncoder setBytes:&lenU length:sizeof(lenU) atIndex:5];
            [stageEncoder setBytes:&angle length:sizeof(angle) atIndex:6];
            [stageEncoder setBytes:&pairCountU length:sizeof(pairCountU) atIndex:7];
            [stageEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total / 2), 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
            [stageEncoder endEncoding];

            std::swap(currentReal, spareReal);
            std::swap(currentImag, spareImag);
        }

        if (inverse) {
            const float scale = 1.0f / static_cast<float>(length);
            id<MTLComputeCommandEncoder> scaleEncoder =
                [commandBuffer computeCommandEncoder];
            [scaleEncoder setComputePipelineState:m_impl->fftScalePipeline];
            [scaleEncoder setBuffer:currentReal offset:0 atIndex:0];
            [scaleEncoder setBuffer:currentImag offset:0 atIndex:1];
            [scaleEncoder setBuffer:spareReal offset:0 atIndex:2];
            [scaleEncoder setBuffer:spareImag offset:0 atIndex:3];
            [scaleEncoder setBytes:&scale length:sizeof(scale) atIndex:4];
            [scaleEncoder setBytes:&totalU length:sizeof(totalU) atIndex:5];
            [scaleEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
            [scaleEncoder endEncoding];
            std::swap(currentReal, spareReal);
            std::swap(currentImag, spareImag);
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(nextReal.data(), [currentReal contents], byteCount);
        std::memcpy(nextImag.data(), [currentImag contents], byteCount);
    }
}

void MetalBackend::matVec(std::span<const float> matrix, std::size_t rows,
                          std::size_t cols, std::span<const float> vector,
                          std::span<float> result) const {
    if (rows == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        id<MTLBuffer> matrixBuffer = [device
            newBufferWithBytes:matrix.data()
                        length:static_cast<NSUInteger>(rows * cols * sizeof(float))
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> vectorBuffer =
            [device newBufferWithBytes:vector.data()
                                length:static_cast<NSUInteger>(cols * sizeof(float))
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer =
            [device newBufferWithLength:static_cast<NSUInteger>(rows * sizeof(float))
                                options:MTLResourceStorageModeShared];
        const auto rowsU = static_cast<std::uint32_t>(rows);
        const auto colsU = static_cast<std::uint32_t>(cols);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->matVecPipeline];
        [encoder setBuffer:matrixBuffer offset:0 atIndex:0];
        [encoder setBuffer:vectorBuffer offset:0 atIndex:1];
        [encoder setBuffer:resultBuffer offset:0 atIndex:2];
        [encoder setBytes:&rowsU length:sizeof(rowsU) atIndex:3];
        [encoder setBytes:&colsU length:sizeof(colsU) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(rows), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(result.data(), [resultBuffer contents],
                    static_cast<NSUInteger>(rows * sizeof(float)));
    }
}

void MetalBackend::matMul(std::span<const float> a, std::size_t aRows, std::size_t aCols,
                          std::span<const float> b, std::size_t bCols,
                          std::span<float> result) const {
    const std::size_t total = aRows * bCols;
    if (total == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        id<MTLBuffer> aBuffer = [device
            newBufferWithBytes:a.data()
                        length:static_cast<NSUInteger>(aRows * aCols * sizeof(float))
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuffer = [device
            newBufferWithBytes:b.data()
                        length:static_cast<NSUInteger>(aCols * bCols * sizeof(float))
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer =
            [device newBufferWithLength:static_cast<NSUInteger>(total * sizeof(float))
                                options:MTLResourceStorageModeShared];
        const auto aRowsU = static_cast<std::uint32_t>(aRows);
        const auto aColsU = static_cast<std::uint32_t>(aCols);
        const auto bColsU = static_cast<std::uint32_t>(bCols);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->matMulPipeline];
        [encoder setBuffer:aBuffer offset:0 atIndex:0];
        [encoder setBuffer:bBuffer offset:0 atIndex:1];
        [encoder setBuffer:resultBuffer offset:0 atIndex:2];
        [encoder setBytes:&aRowsU length:sizeof(aRowsU) atIndex:3];
        [encoder setBytes:&aColsU length:sizeof(aColsU) atIndex:4];
        [encoder setBytes:&bColsU length:sizeof(bColsU) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(total), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(result.data(), [resultBuffer contents],
                    static_cast<NSUInteger>(total * sizeof(float)));
    }
}

bool MetalBackend::luDecomposeGpu(std::span<const float> matrix, std::size_t n,
                                  std::span<float> lu,
                                  std::span<std::uint32_t> pivot) const {
    if (n == 0) {
        return true;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(n * n * sizeof(float));

        id<MTLBuffer> bufferA = [device newBufferWithBytes:matrix.data()
                                                    length:byteCount
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithLength:byteCount
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> current = bufferA;
        id<MTLBuffer> spare = bufferB;

        std::vector<std::uint32_t> pivotHost(n);
        for (std::size_t i = 0; i < n; ++i) {
            pivotHost[i] = static_cast<std::uint32_t>(i);
        }

        const auto nU = static_cast<std::uint32_t>(n);

        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t activeRows = n - k;
            const auto groups = static_cast<std::size_t>(groupCountFor(activeRows));
            const auto partialBytes = static_cast<NSUInteger>(groups * sizeof(float));

            id<MTLBuffer> partialValues =
                [device newBufferWithLength:partialBytes
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> partialIndices =
                [device newBufferWithLength:partialBytes
                                    options:MTLResourceStorageModeShared];
            const auto columnU = static_cast<std::uint32_t>(k);
            const auto startRowU = static_cast<std::uint32_t>(k);
            const auto activeRowsU = static_cast<std::uint32_t>(activeRows);

            id<MTLCommandBuffer> reduceBuffer = [m_impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> reduceEncoder =
                [reduceBuffer computeCommandEncoder];
            [reduceEncoder setComputePipelineState:m_impl->argMaxAbsColumnPipeline];
            [reduceEncoder setBuffer:current offset:0 atIndex:0];
            [reduceEncoder setBuffer:partialValues offset:0 atIndex:1];
            [reduceEncoder setBuffer:partialIndices offset:0 atIndex:2];
            [reduceEncoder setBytes:&nU length:sizeof(nU) atIndex:3];
            [reduceEncoder setBytes:&columnU length:sizeof(columnU) atIndex:4];
            [reduceEncoder setBytes:&startRowU length:sizeof(startRowU) atIndex:5];
            [reduceEncoder setBytes:&activeRowsU length:sizeof(activeRowsU) atIndex:6];
            [reduceEncoder setThreadgroupMemoryLength:(kThreadgroupSize * sizeof(float))
                                              atIndex:0];
            [reduceEncoder
                setThreadgroupMemoryLength:(kThreadgroupSize * sizeof(std::uint32_t))
                                   atIndex:1];
            [reduceEncoder
                 dispatchThreadgroups:MTLSizeMake(groupCountFor(activeRows), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
            [reduceEncoder endEncoding];
            [reduceBuffer commit];
            [reduceBuffer waitUntilCompleted];

            std::vector<float> values(groups);
            std::vector<float> indices(groups);
            std::memcpy(values.data(), [partialValues contents], partialBytes);
            std::memcpy(indices.data(), [partialIndices contents], partialBytes);

            std::size_t best = 0;
            for (std::size_t i = 1; i < groups; ++i) {
                if (values[best] < values[i]) {
                    best = i;
                }
            }
            const float bestValue = values[best];
            const auto pivotRow = static_cast<std::size_t>(indices[best]);

            if (!(bestValue > 0.0f) || !std::isfinite(bestValue)) {
                return false;
            }

            const auto pivotRowU = static_cast<std::uint32_t>(pivotRow);

            id<MTLCommandBuffer> stepBuffer = [m_impl->queue commandBuffer];

            id<MTLComputeCommandEncoder> swapEncoder = [stepBuffer computeCommandEncoder];
            [swapEncoder setComputePipelineState:m_impl->swapRowsPipeline];
            [swapEncoder setBuffer:current offset:0 atIndex:0];
            [swapEncoder setBuffer:spare offset:0 atIndex:1];
            [swapEncoder setBytes:&nU length:sizeof(nU) atIndex:2];
            [swapEncoder setBytes:&startRowU length:sizeof(startRowU) atIndex:3];
            [swapEncoder setBytes:&pivotRowU length:sizeof(pivotRowU) atIndex:4];
            [swapEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n * n), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
            [swapEncoder endEncoding];

            std::swap(current, spare);

            id<MTLComputeCommandEncoder> elimEncoder = [stepBuffer computeCommandEncoder];
            [elimEncoder setComputePipelineState:m_impl->luEliminationStepPipeline];
            [elimEncoder setBuffer:current offset:0 atIndex:0];
            [elimEncoder setBuffer:spare offset:0 atIndex:1];
            [elimEncoder setBytes:&nU length:sizeof(nU) atIndex:2];
            [elimEncoder setBytes:&startRowU length:sizeof(startRowU) atIndex:3];
            [elimEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n * n), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
            [elimEncoder endEncoding];

            [stepBuffer commit];
            [stepBuffer waitUntilCompleted];

            std::swap(current, spare);

            if (pivotRow != k) {
                std::swap(pivotHost[k], pivotHost[pivotRow]);
            }
        }

        std::memcpy(lu.data(), [current contents], byteCount);
        for (std::size_t i = 0; i < n; ++i) {
            pivot[i] = pivotHost[i];
        }
    }
    return true;
}

bool MetalBackend::choleskyDecomposeGpu(std::span<const float> a, std::size_t n,
                                        std::span<float> l) const {
    if (n == 0) {
        return true;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(n * n * sizeof(float));

        id<MTLBuffer> aBuffer = [device newBufferWithBytes:a.data()
                                                    length:byteCount
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> lBuffer = [device newBufferWithLength:byteCount
                                                    options:MTLResourceStorageModeShared];
        std::memset([lBuffer contents], 0, byteCount);

        const auto nU = static_cast<std::uint32_t>(n);

        for (std::size_t j = 0; j < n; ++j) {
            const auto jU = static_cast<std::uint32_t>(j);

            id<MTLCommandBuffer> diagBuffer = [m_impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> diagEncoder = [diagBuffer computeCommandEncoder];
            [diagEncoder setComputePipelineState:m_impl->choleskyDiagonalPipeline];
            [diagEncoder setBuffer:aBuffer offset:0 atIndex:0];
            [diagEncoder setBuffer:lBuffer offset:0 atIndex:1];
            [diagEncoder setBytes:&nU length:sizeof(nU) atIndex:2];
            [diagEncoder setBytes:&jU length:sizeof(jU) atIndex:3];
            [diagEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [diagEncoder endEncoding];
            [diagBuffer commit];
            [diagBuffer waitUntilCompleted];

            float diagonalValue = 0.0f;
            std::memcpy(&diagonalValue,
                        static_cast<const float*>([lBuffer contents]) + (j * n + j),
                        sizeof(float));
            if (diagonalValue < 0.0f) {
                return false;
            }

            const std::size_t remainingRows = n - j - 1;
            if (remainingRows > 0) {
                id<MTLCommandBuffer> colBuffer = [m_impl->queue commandBuffer];
                id<MTLComputeCommandEncoder> colEncoder =
                    [colBuffer computeCommandEncoder];
                [colEncoder setComputePipelineState:m_impl->choleskyColumnPipeline];
                [colEncoder setBuffer:aBuffer offset:0 atIndex:0];
                [colEncoder setBuffer:lBuffer offset:0 atIndex:1];
                [colEncoder setBytes:&nU length:sizeof(nU) atIndex:2];
                [colEncoder setBytes:&jU length:sizeof(jU) atIndex:3];
                [colEncoder
                     dispatchThreadgroups:MTLSizeMake(groupCountFor(remainingRows), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
                [colEncoder endEncoding];
                [colBuffer commit];
                [colBuffer waitUntilCompleted];
            }
        }

        std::memcpy(l.data(), [lBuffer contents], byteCount);
    }
    return true;
}

void MetalBackend::qrDecomposeGpu(std::span<const float> matrix, std::size_t rows,
                                  std::size_t cols, std::span<float> q,
                                  std::span<float> r) const {
    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto rBytes = static_cast<NSUInteger>(rows * cols * sizeof(float));
        const auto qBytes = static_cast<NSUInteger>(rows * rows * sizeof(float));

        id<MTLBuffer> rBufferA = [device newBufferWithBytes:matrix.data()
                                                     length:rBytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> rBufferB =
            [device newBufferWithLength:rBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> qBufferA =
            [device newBufferWithLength:qBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> qBufferB =
            [device newBufferWithLength:qBytes options:MTLResourceStorageModeShared];
        {
            auto* qInit = static_cast<float*>([qBufferA contents]);
            std::memset(qInit, 0, qBytes);
            for (std::size_t i = 0; i < rows; ++i) {
                qInit[i * rows + i] = 1.0f;
            }
        }

        id<MTLBuffer> currentR = rBufferA;
        id<MTLBuffer> spareR = rBufferB;
        id<MTLBuffer> currentQ = qBufferA;
        id<MTLBuffer> spareQ = qBufferB;

        const auto rowsU = static_cast<std::uint32_t>(rows);
        const auto colsU = static_cast<std::uint32_t>(cols);
        const std::size_t steps = std::min(rows, cols);

        for (std::size_t k = 0; k < steps; ++k) {
            id<MTLBuffer> normOutput =
                [device newBufferWithLength:2 * sizeof(float)
                                    options:MTLResourceStorageModeShared];
            const auto kU = static_cast<std::uint32_t>(k);

            id<MTLCommandBuffer> normBuffer = [m_impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> normEncoder = [normBuffer computeCommandEncoder];
            [normEncoder setComputePipelineState:m_impl->qrColumnNormSquaredPipeline];
            [normEncoder setBuffer:currentR offset:0 atIndex:0];
            [normEncoder setBuffer:normOutput offset:0 atIndex:1];
            [normEncoder setBytes:&rowsU length:sizeof(rowsU) atIndex:2];
            [normEncoder setBytes:&colsU length:sizeof(colsU) atIndex:3];
            [normEncoder setBytes:&kU length:sizeof(kU) atIndex:4];
            [normEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [normEncoder endEncoding];
            [normBuffer commit];
            [normBuffer waitUntilCompleted];

            float outputs[2];
            std::memcpy(outputs, [normOutput contents], sizeof(outputs));
            const float sumSquares = outputs[0];
            const float rkk = outputs[1];
            const float normX = std::sqrt(sumSquares);

            if (normX == 0.0f) {
                continue;
            }
            const float alpha = (rkk < 0.0f) ? normX : -normX;
            const float vNormSquared = sumSquares - 2.0f * alpha * rkk + alpha * alpha;
            if (vNormSquared == 0.0f) {
                continue;
            }

            id<MTLCommandBuffer> stepBuffer = [m_impl->queue commandBuffer];

            id<MTLComputeCommandEncoder> applyEncoder =
                [stepBuffer computeCommandEncoder];
            [applyEncoder setComputePipelineState:m_impl->qrApplyLeftPipeline];
            [applyEncoder setBuffer:currentR offset:0 atIndex:0];
            [applyEncoder setBuffer:spareR offset:0 atIndex:1];
            [applyEncoder setBytes:&rowsU length:sizeof(rowsU) atIndex:2];
            [applyEncoder setBytes:&colsU length:sizeof(colsU) atIndex:3];
            [applyEncoder setBytes:&kU length:sizeof(kU) atIndex:4];
            [applyEncoder setBytes:&alpha length:sizeof(alpha) atIndex:5];
            [applyEncoder setBytes:&vNormSquared length:sizeof(vNormSquared) atIndex:6];
            [applyEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(cols), 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
            [applyEncoder endEncoding];

            id<MTLComputeCommandEncoder> accumulateEncoder =
                [stepBuffer computeCommandEncoder];
            [accumulateEncoder setComputePipelineState:m_impl->qrAccumulateQPipeline];
            [accumulateEncoder setBuffer:currentR offset:0 atIndex:0];
            [accumulateEncoder setBuffer:currentQ offset:0 atIndex:1];
            [accumulateEncoder setBuffer:spareQ offset:0 atIndex:2];
            [accumulateEncoder setBytes:&rowsU length:sizeof(rowsU) atIndex:3];
            [accumulateEncoder setBytes:&colsU length:sizeof(colsU) atIndex:4];
            [accumulateEncoder setBytes:&kU length:sizeof(kU) atIndex:5];
            [accumulateEncoder setBytes:&alpha length:sizeof(alpha) atIndex:6];
            [accumulateEncoder setBytes:&vNormSquared
                                 length:sizeof(vNormSquared)
                                atIndex:7];
            [accumulateEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(rows), 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
            [accumulateEncoder endEncoding];

            [stepBuffer commit];
            [stepBuffer waitUntilCompleted];

            std::swap(currentR, spareR);
            std::swap(currentQ, spareQ);
        }

        std::memcpy(r.data(), [currentR contents], rBytes);
        std::memcpy(q.data(), [currentQ contents], qBytes);
    }
}

namespace {

/// The standard round-robin/tournament pairing schedule (Brent & Luk
/// 1985) for `n` indices: `n - 1` rounds (padded to an even count
/// internally if `n` is odd, with whichever real index would face the pad
/// simply sitting out that round), each a length-`n` array giving every
/// index's partner this round, or itself if unpaired. Shared by both
/// jacobiEigenSymmetricGpu and jacobiSvdGpu below.
std::vector<std::vector<std::uint32_t>> roundRobinSchedule(std::size_t n) {
    if (n < 2) {
        return {};
    }
    const std::size_t n2 = n + (n % 2);
    std::vector<std::uint32_t> perm(n2);
    for (std::size_t i = 0; i < n2; ++i) {
        perm[i] = static_cast<std::uint32_t>(i);
    }

    std::vector<std::vector<std::uint32_t>> rounds;
    for (std::size_t round = 0; round < n2 - 1; ++round) {
        std::vector<std::uint32_t> pairOf(n);
        for (std::size_t i = 0; i < n; ++i) {
            pairOf[i] = static_cast<std::uint32_t>(i);
        }
        for (std::size_t i = 0; i < n2 / 2; ++i) {
            const std::uint32_t p = perm[i];
            const std::uint32_t q = perm[n2 - 1 - i];
            if (p < n && q < n) {
                pairOf[p] = q;
                pairOf[q] = p;
            }
        }
        rounds.push_back(std::move(pairOf));

        const std::uint32_t last = perm[n2 - 1];
        for (std::size_t i = n2 - 1; i > 1; --i) {
            perm[i] = perm[i - 1];
        }
        perm[1] = last;
    }
    return rounds;
}

}  // namespace

void MetalBackend::jacobiEigenSymmetricGpu(std::span<const float> matrix, std::size_t n,
                                           int maxSweeps, float tolerance,
                                           std::span<float> resultDiagonal,
                                           std::span<float> resultEigenvectors) const {
    if (n == 0) {
        return;
    }
    if (n == 1) {
        resultDiagonal[0] = matrix[0];
        resultEigenvectors[0] = 1.0f;
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto nBytes = static_cast<NSUInteger>(n * n * sizeof(float));

        id<MTLBuffer> aBufferA =
            [device newBufferWithLength:nBytes options:MTLResourceStorageModeShared];
        {
            auto* symmetrized = static_cast<float*>([aBufferA contents]);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    symmetrized[i * n + j] =
                        (j <= i) ? matrix[i * n + j] : matrix[j * n + i];
                }
            }
        }
        id<MTLBuffer> aBufferB =
            [device newBufferWithLength:nBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bScratch =
            [device newBufferWithLength:nBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vBufferA =
            [device newBufferWithLength:nBytes options:MTLResourceStorageModeShared];
        {
            auto* vInit = static_cast<float*>([vBufferA contents]);
            std::memset(vInit, 0, nBytes);
            for (std::size_t i = 0; i < n; ++i) {
                vInit[i * n + i] = 1.0f;
            }
        }
        id<MTLBuffer> vBufferB =
            [device newBufferWithLength:nBytes options:MTLResourceStorageModeShared];

        id<MTLBuffer> currentA = aBufferA;
        id<MTLBuffer> spareA = aBufferB;
        id<MTLBuffer> currentV = vBufferA;
        id<MTLBuffer> spareV = vBufferB;

        const auto nU = static_cast<std::uint32_t>(n);
        const float effectiveTolerance =
            (tolerance > 0.0f) ? tolerance
                               : std::numeric_limits<float>::epsilon() * 100.0f;
        const std::vector<std::vector<std::uint32_t>> schedule = roundRobinSchedule(n);

        for (int sweep = 0; sweep < maxSweeps; ++sweep) {
            std::vector<float> snapshot(n * n);
            std::memcpy(snapshot.data(), [currentA contents], nBytes);
            float offDiagonalSquared = 0.0f;
            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t qi = p + 1; qi < n; ++qi) {
                    const float value = snapshot[p * n + qi];
                    offDiagonalSquared += value * value;
                }
            }
            if (offDiagonalSquared < effectiveTolerance * effectiveTolerance) {
                break;
            }

            for (const std::vector<std::uint32_t>& pairOf : schedule) {
                id<MTLBuffer> pairBuffer = [device
                    newBufferWithBytes:pairOf.data()
                                length:static_cast<NSUInteger>(n * sizeof(std::uint32_t))
                               options:MTLResourceStorageModeShared];

                id<MTLCommandBuffer> roundBuffer = [m_impl->queue commandBuffer];

                id<MTLComputeCommandEncoder> columnEncoder =
                    [roundBuffer computeCommandEncoder];
                [columnEncoder
                    setComputePipelineState:m_impl->jacobiEigenColumnMixPipeline];
                [columnEncoder setBuffer:currentA offset:0 atIndex:0];
                [columnEncoder setBuffer:currentV offset:0 atIndex:1];
                [columnEncoder setBuffer:pairBuffer offset:0 atIndex:2];
                [columnEncoder setBuffer:bScratch offset:0 atIndex:3];
                [columnEncoder setBuffer:spareV offset:0 atIndex:4];
                [columnEncoder setBytes:&nU length:sizeof(nU) atIndex:5];
                [columnEncoder setBytes:&effectiveTolerance
                                 length:sizeof(effectiveTolerance)
                                atIndex:6];
                [columnEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n), 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
                [columnEncoder endEncoding];

                id<MTLComputeCommandEncoder> rowEncoder =
                    [roundBuffer computeCommandEncoder];
                [rowEncoder setComputePipelineState:m_impl->jacobiEigenRowMixPipeline];
                [rowEncoder setBuffer:currentA offset:0 atIndex:0];
                [rowEncoder setBuffer:bScratch offset:0 atIndex:1];
                [rowEncoder setBuffer:pairBuffer offset:0 atIndex:2];
                [rowEncoder setBuffer:spareA offset:0 atIndex:3];
                [rowEncoder setBytes:&nU length:sizeof(nU) atIndex:4];
                [rowEncoder setBytes:&effectiveTolerance
                              length:sizeof(effectiveTolerance)
                             atIndex:5];
                [rowEncoder dispatchThreadgroups:MTLSizeMake(groupCountFor(n), 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
                [rowEncoder endEncoding];

                [roundBuffer commit];
                [roundBuffer waitUntilCompleted];

                std::swap(currentA, spareA);
                std::swap(currentV, spareV);
            }
        }

        std::memcpy(resultDiagonal.data(), [currentA contents], nBytes);
        std::memcpy(resultEigenvectors.data(), [currentV contents], nBytes);
    }
}

void MetalBackend::jacobiSvdGpu(std::span<const float> matrix, std::size_t rows,
                                std::size_t cols, int maxSweeps, float tolerance,
                                std::span<float> resultA,
                                std::span<float> resultV) const {
    if (cols == 0) {
        return;
    }
    if (cols == 1) {
        for (std::size_t i = 0; i < rows; ++i) {
            resultA[i] = matrix[i];
        }
        resultV[0] = 1.0f;
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto aBytes = static_cast<NSUInteger>(rows * cols * sizeof(float));
        const auto vBytes = static_cast<NSUInteger>(cols * cols * sizeof(float));

        id<MTLBuffer> aBufferA = [device newBufferWithBytes:matrix.data()
                                                     length:aBytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> aBufferB =
            [device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vBufferA =
            [device newBufferWithLength:vBytes options:MTLResourceStorageModeShared];
        {
            auto* vInit = static_cast<float*>([vBufferA contents]);
            std::memset(vInit, 0, vBytes);
            for (std::size_t i = 0; i < cols; ++i) {
                vInit[i * cols + i] = 1.0f;
            }
        }
        id<MTLBuffer> vBufferB =
            [device newBufferWithLength:vBytes options:MTLResourceStorageModeShared];

        id<MTLBuffer> currentA = aBufferA;
        id<MTLBuffer> spareA = aBufferB;
        id<MTLBuffer> currentV = vBufferA;
        id<MTLBuffer> spareV = vBufferB;

        const auto rowsU = static_cast<std::uint32_t>(rows);
        const auto colsU = static_cast<std::uint32_t>(cols);
        const float effectiveTolerance =
            (tolerance > 0.0f) ? tolerance
                               : std::numeric_limits<float>::epsilon() * 100.0f;
        const std::vector<std::vector<std::uint32_t>> schedule = roundRobinSchedule(cols);

        for (int sweep = 0; sweep < maxSweeps; ++sweep) {
            std::vector<float> snapshot(rows * cols);
            std::memcpy(snapshot.data(), [currentA contents], aBytes);
            float offDiagonalSquared = 0.0f;
            for (std::size_t p = 0; p < cols; ++p) {
                for (std::size_t qi = p + 1; qi < cols; ++qi) {
                    float gamma = 0.0f;
                    for (std::size_t i = 0; i < rows; ++i) {
                        gamma += snapshot[i * cols + p] * snapshot[i * cols + qi];
                    }
                    offDiagonalSquared += gamma * gamma;
                }
            }
            if (offDiagonalSquared < effectiveTolerance * effectiveTolerance) {
                break;
            }

            for (const std::vector<std::uint32_t>& pairOf : schedule) {
                id<MTLBuffer> pairBuffer = [device
                    newBufferWithBytes:pairOf.data()
                                length:static_cast<NSUInteger>(cols *
                                                               sizeof(std::uint32_t))
                               options:MTLResourceStorageModeShared];

                id<MTLCommandBuffer> roundBuffer = [m_impl->queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder =
                    [roundBuffer computeCommandEncoder];
                [encoder setComputePipelineState:m_impl->jacobiSvdRoundPipeline];
                [encoder setBuffer:currentA offset:0 atIndex:0];
                [encoder setBuffer:currentV offset:0 atIndex:1];
                [encoder setBuffer:pairBuffer offset:0 atIndex:2];
                [encoder setBuffer:spareA offset:0 atIndex:3];
                [encoder setBuffer:spareV offset:0 atIndex:4];
                [encoder setBytes:&rowsU length:sizeof(rowsU) atIndex:5];
                [encoder setBytes:&colsU length:sizeof(colsU) atIndex:6];
                [encoder setBytes:&effectiveTolerance
                           length:sizeof(effectiveTolerance)
                          atIndex:7];
                [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(cols), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
                [encoder endEncoding];
                [roundBuffer commit];
                [roundBuffer waitUntilCompleted];

                std::swap(currentA, spareA);
                std::swap(currentV, spareV);
            }
        }

        std::memcpy(resultA.data(), [currentA contents], aBytes);
        std::memcpy(resultV.data(), [currentV contents], vBytes);
    }
}

namespace {

/// The shared shape every batched pointwise kernel below dispatches with:
/// one input buffer, one output buffer, both `x.size()` floats, a single
/// `count` uniform, one thread per element.
void dispatchBatchedPointwise(id<MTLDevice> device, id<MTLCommandQueue> queue,
                              id<MTLComputePipelineState> pipeline,
                              std::span<const float> x, std::span<float> result) {
    if (x.empty()) {
        return;
    }
    @autoreleasepool {
        const auto byteCount = static_cast<NSUInteger>(x.size() * sizeof(float));
        id<MTLBuffer> xBuffer = [device newBufferWithBytes:x.data()
                                                    length:byteCount
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        const auto count = static_cast<std::uint32_t>(x.size());

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:xBuffer offset:0 atIndex:0];
        [encoder setBuffer:resultBuffer offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(x.size()), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(result.data(), [resultBuffer contents], byteCount);
    }
}

}  // namespace

void MetalBackend::batchErf(std::span<const float> x, std::span<float> result) const {
    dispatchBatchedPointwise(m_impl->device, m_impl->queue, m_impl->batchErfPipeline, x,
                             result);
}

void MetalBackend::batchErfc(std::span<const float> x, std::span<float> result) const {
    dispatchBatchedPointwise(m_impl->device, m_impl->queue, m_impl->batchErfcPipeline, x,
                             result);
}

void MetalBackend::batchGamma(std::span<const float> x, std::span<float> result) const {
    dispatchBatchedPointwise(m_impl->device, m_impl->queue, m_impl->batchGammaPipeline, x,
                             result);
}

void MetalBackend::batchLogGamma(std::span<const float> x,
                                 std::span<float> result) const {
    dispatchBatchedPointwise(m_impl->device, m_impl->queue, m_impl->batchLogGammaPipeline,
                             x, result);
}

void MetalBackend::batchLegendreP(unsigned n, unsigned m, std::span<const float> x,
                                  std::span<float> result) const {
    if (x.empty()) {
        return;
    }
    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto byteCount = static_cast<NSUInteger>(x.size() * sizeof(float));
        id<MTLBuffer> xBuffer = [device newBufferWithBytes:x.data()
                                                    length:byteCount
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        const auto count = static_cast<std::uint32_t>(x.size());
        const auto nU = static_cast<std::uint32_t>(n);
        const auto mU = static_cast<std::uint32_t>(m);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->batchLegendrePPipeline];
        [encoder setBuffer:xBuffer offset:0 atIndex:0];
        [encoder setBuffer:resultBuffer offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&nU length:sizeof(nU) atIndex:3];
        [encoder setBytes:&mU length:sizeof(mU) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(x.size()), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(result.data(), [resultBuffer contents], byteCount);
    }
}

void MetalBackend::batchPolynomialEval(std::span<const float> coefficients,
                                       std::span<const float> x,
                                       std::span<float> result) const {
    if (x.empty()) {
        return;
    }
    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto coeffBytes =
            static_cast<NSUInteger>(coefficients.size() * sizeof(float));
        const auto xBytes = static_cast<NSUInteger>(x.size() * sizeof(float));
        id<MTLBuffer> coeffBuffer =
            [device newBufferWithBytes:coefficients.data()
                                length:coeffBytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> xBuffer = [device newBufferWithBytes:x.data()
                                                    length:xBytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer =
            [device newBufferWithLength:xBytes options:MTLResourceStorageModeShared];
        const auto coefficientCount = static_cast<std::uint32_t>(coefficients.size());
        const auto count = static_cast<std::uint32_t>(x.size());

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->batchPolynomialEvalPipeline];
        [encoder setBuffer:coeffBuffer offset:0 atIndex:0];
        [encoder setBuffer:xBuffer offset:0 atIndex:1];
        [encoder setBuffer:resultBuffer offset:0 atIndex:2];
        [encoder setBytes:&coefficientCount length:sizeof(coefficientCount) atIndex:3];
        [encoder setBytes:&count length:sizeof(count) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(x.size()), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(result.data(), [resultBuffer contents], xBytes);
    }
}

void MetalBackend::batchCubicSplineEval(std::span<const float> knotsX,
                                        std::span<const float> knotsY,
                                        std::span<const float> secondDerivatives,
                                        std::span<const float> queryX,
                                        std::span<float> result) const {
    if (queryX.empty()) {
        return;
    }
    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto knotBytes = static_cast<NSUInteger>(knotsX.size() * sizeof(float));
        const auto queryBytes = static_cast<NSUInteger>(queryX.size() * sizeof(float));
        id<MTLBuffer> knotsXBuffer =
            [device newBufferWithBytes:knotsX.data()
                                length:knotBytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> knotsYBuffer =
            [device newBufferWithBytes:knotsY.data()
                                length:knotBytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> secondDerivativesBuffer =
            [device newBufferWithBytes:secondDerivatives.data()
                                length:knotBytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> queryXBuffer =
            [device newBufferWithBytes:queryX.data()
                                length:queryBytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer =
            [device newBufferWithLength:queryBytes options:MTLResourceStorageModeShared];
        const auto knotCount = static_cast<std::uint32_t>(knotsX.size());
        const auto count = static_cast<std::uint32_t>(queryX.size());

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->batchCubicSplineEvalPipeline];
        [encoder setBuffer:knotsXBuffer offset:0 atIndex:0];
        [encoder setBuffer:knotsYBuffer offset:0 atIndex:1];
        [encoder setBuffer:secondDerivativesBuffer offset:0 atIndex:2];
        [encoder setBuffer:queryXBuffer offset:0 atIndex:3];
        [encoder setBuffer:resultBuffer offset:0 atIndex:4];
        [encoder setBytes:&knotCount length:sizeof(knotCount) atIndex:5];
        [encoder setBytes:&count length:sizeof(count) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(queryX.size()), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(result.data(), [resultBuffer contents], queryBytes);
    }
}

namespace {

/// The shared shape batchUniformReal/batchNormal both dispatch with: no
/// input buffer at all (Philox is counter-based, needing only `seed` and
/// `offset` as small uniforms, not per-element input data), one output
/// buffer, one thread per element.
void dispatchBatchRandom(id<MTLDevice> device, id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline, std::uint64_t seed,
                         std::uint64_t offset, std::span<float> result) {
    if (result.empty()) {
        return;
    }
    @autoreleasepool {
        const auto byteCount = static_cast<NSUInteger>(result.size() * sizeof(float));
        id<MTLBuffer> resultBuffer =
            [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
        const auto count = static_cast<std::uint32_t>(result.size());
        const auto seedLo = static_cast<std::uint32_t>(seed);
        const auto seedHi = static_cast<std::uint32_t>(seed >> 32);
        const auto offsetLo = static_cast<std::uint32_t>(offset);
        const auto offsetHi = static_cast<std::uint32_t>(offset >> 32);

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:resultBuffer offset:0 atIndex:0];
        [encoder setBytes:&count length:sizeof(count) atIndex:1];
        [encoder setBytes:&seedLo length:sizeof(seedLo) atIndex:2];
        [encoder setBytes:&seedHi length:sizeof(seedHi) atIndex:3];
        [encoder setBytes:&offsetLo length:sizeof(offsetLo) atIndex:4];
        [encoder setBytes:&offsetHi length:sizeof(offsetHi) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(result.size()), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(result.data(), [resultBuffer contents], byteCount);
    }
}

}  // namespace

void MetalBackend::batchUniformReal(std::uint64_t seed, std::uint64_t offset,
                                    std::span<float> result) const {
    dispatchBatchRandom(m_impl->device, m_impl->queue, m_impl->batchUniformRealPipeline,
                        seed, offset, result);
}

void MetalBackend::batchNormal(std::uint64_t seed, std::uint64_t offset,
                               std::span<float> result) const {
    dispatchBatchRandom(m_impl->device, m_impl->queue, m_impl->batchNormalPipeline, seed,
                        offset, result);
}

void MetalBackend::sortAscending(std::span<float> values) const {
    if (values.size() < 2) {
        return;
    }
    std::size_t paddedSize = 1;
    while (paddedSize < values.size()) {
        paddedSize *= 2;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        std::vector<float> padded(paddedSize, std::numeric_limits<float>::infinity());
        std::copy(values.begin(), values.end(), padded.begin());
        const auto byteCount = static_cast<NSUInteger>(paddedSize * sizeof(float));
        id<MTLBuffer> buffer = [device newBufferWithBytes:padded.data()
                                                   length:byteCount
                                                  options:MTLResourceStorageModeShared];
        const auto n = static_cast<std::uint32_t>(paddedSize);

        for (std::uint32_t stageSize = 2; stageSize <= n; stageSize *= 2) {
            for (std::uint32_t stepSize = stageSize / 2; stepSize >= 1; stepSize /= 2) {
                id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder =
                    [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:m_impl->bitonicCompareExchangePipeline];
                [encoder setBuffer:buffer offset:0 atIndex:0];
                [encoder setBytes:&n length:sizeof(n) atIndex:1];
                [encoder setBytes:&stageSize length:sizeof(stageSize) atIndex:2];
                [encoder setBytes:&stepSize length:sizeof(stepSize) atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(paddedSize), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
            }
        }

        std::memcpy(padded.data(), [buffer contents], byteCount);
        std::copy(padded.begin(),
                  padded.begin() + static_cast<std::ptrdiff_t>(values.size()),
                  values.begin());
    }
}

void MetalBackend::multigridRestrict3D(std::span<const float> fine, std::size_t nx,
                                       std::size_t ny, std::size_t nz,
                                       std::span<float> coarse) const {
    const std::size_t fineTotal = nx * ny * nz;
    const std::size_t coarseTotal = (nx / 2) * (ny / 2) * (nz / 2);
    if (coarseTotal == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto fineByteCount = static_cast<NSUInteger>(fineTotal * sizeof(float));
        const auto coarseByteCount = static_cast<NSUInteger>(coarseTotal * sizeof(float));

        id<MTLBuffer> fineBuffer =
            [device newBufferWithBytes:fine.data()
                                length:fineByteCount
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> coarseBuffer =
            [device newBufferWithLength:coarseByteCount
                                options:MTLResourceStorageModeShared];
        const auto nxInt = static_cast<std::int32_t>(nx);
        const auto nyInt = static_cast<std::int32_t>(ny);
        const auto nzInt = static_cast<std::int32_t>(nz);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->multigridRestrict3DPipeline];
        [encoder setBuffer:fineBuffer offset:0 atIndex:0];
        [encoder setBuffer:coarseBuffer offset:0 atIndex:1];
        [encoder setBytes:&nxInt length:sizeof(nxInt) atIndex:2];
        [encoder setBytes:&nyInt length:sizeof(nyInt) atIndex:3];
        [encoder setBytes:&nzInt length:sizeof(nzInt) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(coarseTotal), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(coarse.data(), [coarseBuffer contents], coarseByteCount);
    }
}

void MetalBackend::multigridProlongateAndAdd3D(std::span<const float> fine,
                                               std::span<const float> coarseCorrection,
                                               std::size_t nx, std::size_t ny,
                                               std::size_t nz,
                                               std::span<float> nextFine) const {
    const std::size_t fineTotal = nx * ny * nz;
    if (fineTotal == 0) {
        return;
    }
    const std::size_t coarseTotal = (nx / 2) * (ny / 2) * (nz / 2);

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto fineByteCount = static_cast<NSUInteger>(fineTotal * sizeof(float));
        const auto coarseByteCount = static_cast<NSUInteger>(coarseTotal * sizeof(float));

        id<MTLBuffer> fineBuffer =
            [device newBufferWithBytes:fine.data()
                                length:fineByteCount
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> coarseBuffer =
            [device newBufferWithBytes:coarseCorrection.data()
                                length:coarseByteCount
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> nextFineBuffer =
            [device newBufferWithLength:fineByteCount
                                options:MTLResourceStorageModeShared];
        const auto nxInt = static_cast<std::int32_t>(nx);
        const auto nyInt = static_cast<std::int32_t>(ny);
        const auto nzInt = static_cast<std::int32_t>(nz);

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->multigridProlongateAndAdd3DPipeline];
        [encoder setBuffer:fineBuffer offset:0 atIndex:0];
        [encoder setBuffer:coarseBuffer offset:0 atIndex:1];
        [encoder setBuffer:nextFineBuffer offset:0 atIndex:2];
        [encoder setBytes:&nxInt length:sizeof(nxInt) atIndex:3];
        [encoder setBytes:&nyInt length:sizeof(nyInt) atIndex:4];
        [encoder setBytes:&nzInt length:sizeof(nzInt) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCountFor(fineTotal), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::memcpy(nextFine.data(), [nextFineBuffer contents], fineByteCount);
    }
}

std::size_t MetalBackend::minIndex(std::span<const float> x) const {
    if (x.empty()) {
        return x.size();
    }

    @autoreleasepool {
        id<MTLDevice> device = m_impl->device;
        const auto groups = static_cast<std::size_t>(groupCountFor(x.size()));
        const auto inBytes = static_cast<NSUInteger>(x.size() * sizeof(float));
        const auto outBytes = static_cast<NSUInteger>(groups * sizeof(float));

        id<MTLBuffer> inBuffer = [device newBufferWithBytes:x.data()
                                                     length:inBytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> valueBuffer =
            [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> indexBuffer =
            [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
        const auto count = static_cast<std::uint32_t>(x.size());

        id<MTLCommandBuffer> commandBuffer = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_impl->minIndexPipeline];
        [encoder setBuffer:inBuffer offset:0 atIndex:0];
        [encoder setBuffer:valueBuffer offset:0 atIndex:1];
        [encoder setBuffer:indexBuffer offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        [encoder setThreadgroupMemoryLength:(kThreadgroupSize * sizeof(float)) atIndex:0];
        [encoder setThreadgroupMemoryLength:(kThreadgroupSize * sizeof(std::uint32_t))
                                    atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(groups), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::vector<float> values(groups);
        std::vector<float> indices(groups);
        std::memcpy(values.data(), [valueBuffer contents], outBytes);
        std::memcpy(indices.data(), [indexBuffer contents], outBytes);

        std::size_t best = 0;
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (values[i] < values[best]) {
                best = i;
            }
        }
        return static_cast<std::size_t>(indices[best]);
    }
}

}  // namespace ysq
