#include <Compute/OpenGL/OpenGLBackend.hpp>

#include <Core/Logger.hpp>

#include <glad/gl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ysq {

namespace {

// Compiled at 430 core to match the 4.3 context requested below.
constexpr std::string_view kSaxpySource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer XBuffer { float x[]; };
layout(std430, binding = 1) buffer YBuffer { float y[]; };
uniform float a;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= y.length()) return;
    y[i] = a * x[i] + y[i];
}
)glsl";

// A two-pass tree reduction: this dispatch reduces each work group's slice
// into one partial sum, and OpenGLBackend::sum() finishes the (tiny)
// remaining reduction over those partials on the CPU rather than earning a
// second shader. Written for correctness on any input size, not peak
// throughput; this is a reference kernel, not a tuned one.
constexpr std::string_view kSumSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer InBuffer { float data[]; };
layout(std430, binding = 1) buffer OutBuffer { float partials[]; };
shared float scratch[256];
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint local = gl_LocalInvocationID.x;
    scratch[local] = (i < data.length()) ? data[i] : 0.0;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            scratch[local] += scratch[local + stride];
        }
        barrier();
    }
    if (local == 0u) {
        partials[gl_WorkGroupID.x] = scratch[0];
    }
}
)glsl";

// Four term buffers are always bound, whether or not the caller actually
// asked for that many: termCount gates which ones the shader body actually
// reads, and any slot past termCount is bound to y's own buffer (already
// valid and the right length) purely to give the binding point something
// legal, never read at that index while termCount says not to.
constexpr std::string_view kLinearCombineSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Term0Buffer { float term0[]; };
layout(std430, binding = 1) readonly buffer Term1Buffer { float term1[]; };
layout(std430, binding = 2) readonly buffer Term2Buffer { float term2[]; };
layout(std430, binding = 3) readonly buffer Term3Buffer { float term3[]; };
layout(std430, binding = 4) buffer YBuffer { float y[]; };
uniform int termCount;
uniform vec4 coefficients;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= y.length()) return;
    float total = 0.0;
    if (termCount > 0) total += coefficients.x * term0[i];
    if (termCount > 1) total += coefficients.y * term1[i];
    if (termCount > 2) total += coefficients.z * term2[i];
    if (termCount > 3) total += coefficients.w * term3[i];
    y[i] = total;
}
)glsl";

// Direct-sum (O(n^2)) Newtonian gravity, Plummer-softened. count comes from
// posX.length() rather than a uniform, the same convention kSaxpySource and
// kSumSource already use for their own array lengths.
constexpr std::string_view kGravitationalNBodySource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PosXBuffer { float posX[]; };
layout(std430, binding = 1) readonly buffer PosYBuffer { float posY[]; };
layout(std430, binding = 2) readonly buffer PosZBuffer { float posZ[]; };
layout(std430, binding = 3) readonly buffer GmBuffer { float gm[]; };
layout(std430, binding = 4) buffer AccXBuffer { float accX[]; };
layout(std430, binding = 5) buffer AccYBuffer { float accY[]; };
layout(std430, binding = 6) buffer AccZBuffer { float accZ[]; };
uniform float softeningSquared;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint count = uint(posX.length());
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float ax = 0.0;
    float ay = 0.0;
    float az = 0.0;
    for (uint j = 0u; j < count; ++j) {
        if (j == i) continue;
        float dx = posX[j] - px;
        float dy = posY[j] - py;
        float dz = posZ[j] - pz;
        float r2 = dx * dx + dy * dy + dz * dz + softeningSquared;
        float invR3 = 1.0 / (r2 * sqrt(r2));
        float factor = gm[j] * invR3;
        ax += factor * dx;
        ay += factor * dy;
        az += factor * dz;
    }
    accX[i] = ax;
    accY[i] = ay;
    accZ[i] = az;
}
)glsl";

// Direct-sum Coulomb field, one thread per query charge. Matches
// Physics/Electromagnetism/Field.cpp's electricField exactly: unsoftened,
// skipping a source at zero separation.
constexpr std::string_view kElectricFieldNBodySource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PosXBuffer { float posX[]; };
layout(std430, binding = 1) readonly buffer PosYBuffer { float posY[]; };
layout(std430, binding = 2) readonly buffer PosZBuffer { float posZ[]; };
layout(std430, binding = 3) readonly buffer ChargeBuffer { float charge[]; };
layout(std430, binding = 4) buffer FieldXBuffer { float fieldX[]; };
layout(std430, binding = 5) buffer FieldYBuffer { float fieldY[]; };
layout(std430, binding = 6) buffer FieldZBuffer { float fieldZ[]; };
uniform float coulombConstant;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint count = uint(posX.length());
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float ex = 0.0;
    float ey = 0.0;
    float ez = 0.0;
    for (uint j = 0u; j < count; ++j) {
        if (j == i) continue;
        float dx = px - posX[j];
        float dy = py - posY[j];
        float dz = pz - posZ[j];
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= 0.0) continue;
        float invR3 = 1.0 / (r2 * sqrt(r2));
        float factor = charge[j] * invR3;
        ex += factor * dx;
        ey += factor * dy;
        ez += factor * dz;
    }
    fieldX[i] = coulombConstant * ex;
    fieldY[i] = coulombConstant * ey;
    fieldZ[i] = coulombConstant * ez;
}
)glsl";

// Point-charge Biot-Savart. Matches Field.cpp's magneticField exactly.
// Ten SSBO bindings: above the GL 4.3 spec's guaranteed minimum of 8, so
// this is a real (if unlikely on current hardware) portability limit worth
// flagging; unverified on this machine either way, since macOS never
// reaches OpenGL 4.3 at all (see this file's own create()).
constexpr std::string_view kMagneticFieldNBodySource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PosXBuffer { float posX[]; };
layout(std430, binding = 1) readonly buffer PosYBuffer { float posY[]; };
layout(std430, binding = 2) readonly buffer PosZBuffer { float posZ[]; };
layout(std430, binding = 3) readonly buffer VelXBuffer { float velX[]; };
layout(std430, binding = 4) readonly buffer VelYBuffer { float velY[]; };
layout(std430, binding = 5) readonly buffer VelZBuffer { float velZ[]; };
layout(std430, binding = 6) readonly buffer ChargeBuffer { float charge[]; };
layout(std430, binding = 7) buffer FieldXBuffer { float fieldX[]; };
layout(std430, binding = 8) buffer FieldYBuffer { float fieldY[]; };
layout(std430, binding = 9) buffer FieldZBuffer { float fieldZ[]; };
uniform float permeabilityOver4Pi;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint count = uint(posX.length());
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float bx = 0.0;
    float by = 0.0;
    float bz = 0.0;
    for (uint j = 0u; j < count; ++j) {
        if (j == i) continue;
        float dx = px - posX[j];
        float dy = py - posY[j];
        float dz = pz - posZ[j];
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= 0.0) continue;
        float invR3 = 1.0 / (r2 * sqrt(r2));
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
)glsl";

// GLSL functions matching Physics/Fluids/SPH.hpp's cubicSplineKernel and
// cubicSplineKernelGradient exactly (Monaghan & Lattanzio 1985, normalized
// for 3D). GLSL has no #include, so this is concatenated between each SPH
// shader's own header (buffers, #version) and main() at compile time in
// create(), rather than repeated inline in both.
constexpr std::string_view kCubicSplineFunctions = R"glsl(
float cubicSplineKernel(float r, float h) {
    float q = r / h;
    float sigma = 0.3183098861837907 / (h * h * h);
    if (q < 1.0) return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    if (q < 2.0) { float t = 2.0 - q; return sigma * 0.25 * t * t * t; }
    return 0.0;
}
float cubicSplineKernelGradientOverR(float r, float h) {
    if (r <= 0.0) return 0.0;
    float q = r / h;
    float sigma = 0.3183098861837907 / (h * h * h);
    float dWdq;
    if (q < 1.0) dWdq = sigma * (-3.0 * q + 2.25 * q * q);
    else if (q < 2.0) { float t = 2.0 - q; dWdq = -0.75 * sigma * t * t; }
    else dWdq = 0.0;
    return (dWdq / h) / r;
}
)glsl";

// Direct O(n^2) with a distance cutoff at the kernel's compact support
// (2h), not a spatial acceleration structure; see src/Compute/README.md.
// Matches Physics/Fluids/SPH.hpp's computeDensityAndPressure exactly,
// including the self term (r = 0 contributes cubicSplineKernel(0, h)).
// Split into a header (compiled first) and a main body, concatenated with
// kCubicSplineFunctions between them in create(): #version must be the
// shader's first line, ahead of even the shared kernel functions.
constexpr std::string_view kSphDensityPressureHeader = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PosXBuffer { float posX[]; };
layout(std430, binding = 1) readonly buffer PosYBuffer { float posY[]; };
layout(std430, binding = 2) readonly buffer PosZBuffer { float posZ[]; };
layout(std430, binding = 3) readonly buffer MassBuffer { float mass[]; };
layout(std430, binding = 4) buffer DensityBuffer { float density[]; };
layout(std430, binding = 5) buffer PressureBuffer { float pressure[]; };
uniform float smoothingLength;
uniform float equationOfStateK;
uniform float polytropicIndex;
)glsl";

constexpr std::string_view kSphDensityPressureMain = R"glsl(
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint count = uint(posX.length());
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float supportRadius = 2.0 * smoothingLength;
    float sum = 0.0;
    for (uint j = 0u; j < count; ++j) {
        float dx = px - posX[j];
        float dy = py - posY[j];
        float dz = pz - posZ[j];
        float r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r >= supportRadius) continue;
        sum += mass[j] * cubicSplineKernel(r, smoothingLength);
    }
    density[i] = sum;
    pressure[i] = equationOfStateK * pow(sum, polytropicIndex);
}
)glsl";

// Nine SSBO bindings: above the GL 4.3 spec's guaranteed minimum of 8, the
// same caveat kMagneticFieldNBodySource carries and for the same reason
// (unverified on this machine either way; macOS never reaches OpenGL 4.3).
// Split into header + main around kCubicSplineFunctions, the same reason
// kSphDensityPressureHeader/Main is.
constexpr std::string_view kSphPressureAccelerationHeader = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PosXBuffer { float posX[]; };
layout(std430, binding = 1) readonly buffer PosYBuffer { float posY[]; };
layout(std430, binding = 2) readonly buffer PosZBuffer { float posZ[]; };
layout(std430, binding = 3) readonly buffer MassBuffer { float mass[]; };
layout(std430, binding = 4) readonly buffer DensityBuffer { float density[]; };
layout(std430, binding = 5) readonly buffer PressureBuffer { float pressure[]; };
layout(std430, binding = 6) buffer AccXBuffer { float accX[]; };
layout(std430, binding = 7) buffer AccYBuffer { float accY[]; };
layout(std430, binding = 8) buffer AccZBuffer { float accZ[]; };
uniform float smoothingLength;
)glsl";

constexpr std::string_view kSphPressureAccelerationMain = R"glsl(
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint count = uint(posX.length());
    if (i >= count) return;
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float supportRadius = 2.0 * smoothingLength;
    float targetTerm = pressure[i] / (density[i] * density[i]);
    float ax = 0.0;
    float ay = 0.0;
    float az = 0.0;
    for (uint j = 0u; j < count; ++j) {
        if (j == i) continue;
        float dx = px - posX[j];
        float dy = py - posY[j];
        float dz = pz - posZ[j];
        float r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r >= supportRadius) continue;
        float sourceTerm = pressure[j] / (density[j] * density[j]);
        float gradOverR = cubicSplineKernelGradientOverR(r, smoothingLength);
        float factor = gradOverR * mass[j] * (targetTerm + sourceTerm);
        ax += factor * dx;
        ay += factor * dy;
        az += factor * dz;
    }
    accX[i] = -ax;
    accY[i] = -ay;
    accZ[i] = -az;
}
)glsl";

// One explicit-Euler diffusion step over a periodic grid, flat row-major,
// interior cells only: periodic wraparound is computed directly (modular
// index arithmetic) rather than uploading Grid3D's own ghost cells, so the
// GPU-side layout does not need to match its storage exactly. Matches
// Physics/Thermodynamics/HeatEquation3D.cpp's step() exactly. One thread
// per grid cell, flattened; nx/ny/nz as int uniforms (not uint) since
// ComputeShader has no uint setter and grid sizes here never approach
// INT_MAX.
constexpr std::string_view kHeatEquation3DStepSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer TempBuffer { float temperature[]; };
layout(std430, binding = 1) buffer NextBuffer { float next[]; };
uniform int nx;
uniform int ny;
uniform int nz;
uniform float factor;
void main() {
    int idx = int(gl_GlobalInvocationID.x);
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
                     - 6.0 * here;
    next[idx] = here + factor * laplacian;
}
)glsl";

// Two dispatches, run in sequence by OpenGLBackend::acoustic3DStep(): this
// one updates velocity from the input pressure (matches
// Physics/Acoustics/Acoustic3D.cpp's step() velocity half-step exactly),
// the next one below updates pressure from the velocity this one just
// wrote. Periodic wraparound via modular index arithmetic, no ghost cells.
constexpr std::string_view kAcoustic3DVelocitySource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PressureBuffer { float pressure[]; };
layout(std430, binding = 1) readonly buffer VelXInBuffer { float velocityXIn[]; };
layout(std430, binding = 2) readonly buffer VelYInBuffer { float velocityYIn[]; };
layout(std430, binding = 3) readonly buffer VelZInBuffer { float velocityZIn[]; };
layout(std430, binding = 4) buffer VelXOutBuffer { float velocityXOut[]; };
layout(std430, binding = 5) buffer VelYOutBuffer { float velocityYOut[]; };
layout(std430, binding = 6) buffer VelZOutBuffer { float velocityZOut[]; };
uniform int nx;
uniform int ny;
uniform int nz;
uniform float velocityFactor;
void main() {
    int idx = int(gl_GlobalInvocationID.x);
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
)glsl";

// The second half of acoustic3DStep: updates pressure from the velocity
// kAcoustic3DVelocitySource just wrote (bound as the plain "velocity"
// input here) and the original pressure. Matches Acoustic3D.cpp's
// pressure full-step exactly.
constexpr std::string_view kAcoustic3DPressureSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PressureInBuffer { float pressureIn[]; };
layout(std430, binding = 1) readonly buffer VelXBuffer { float velocityX[]; };
layout(std430, binding = 2) readonly buffer VelYBuffer { float velocityY[]; };
layout(std430, binding = 3) readonly buffer VelZBuffer { float velocityZ[]; };
layout(std430, binding = 4) buffer PressureOutBuffer { float pressureOut[]; };
uniform int nx;
uniform int ny;
uniform int nz;
uniform float pressureFactor;
void main() {
    int idx = int(gl_GlobalInvocationID.x);
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
)glsl";

// Two dispatches, run in sequence by OpenGLBackend::maxwell3DStep(): this
// one updates B from the input E (matches
// Physics/Electromagnetism/Maxwell3D.cpp's step() B half-step exactly),
// the next one below updates E from the B this one just wrote. Nine SSBO
// bindings: above the GL 4.3 spec's guaranteed minimum of 8, the same
// caveat kMagneticFieldNBodySource carries and for the same reason
// (unverified on this machine either way; macOS never reaches OpenGL 4.3).
constexpr std::string_view kMaxwell3DBSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer ExBuffer { float ex[]; };
layout(std430, binding = 1) readonly buffer EyBuffer { float ey[]; };
layout(std430, binding = 2) readonly buffer EzBuffer { float ez[]; };
layout(std430, binding = 3) readonly buffer BxInBuffer { float bxIn[]; };
layout(std430, binding = 4) readonly buffer ByInBuffer { float byIn[]; };
layout(std430, binding = 5) readonly buffer BzInBuffer { float bzIn[]; };
layout(std430, binding = 6) buffer BxOutBuffer { float bxOut[]; };
layout(std430, binding = 7) buffer ByOutBuffer { float byOut[]; };
layout(std430, binding = 8) buffer BzOutBuffer { float bzOut[]; };
uniform int nx;
uniform int ny;
uniform int nz;
uniform float bFactor;
void main() {
    int idx = int(gl_GlobalInvocationID.x);
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
)glsl";

// The second half of maxwell3DStep: updates E from the B
// kMaxwell3DBSource just wrote and the original E. Matches
// Maxwell3D.cpp's E full-step exactly.
constexpr std::string_view kMaxwell3DESource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer ExInBuffer { float exIn[]; };
layout(std430, binding = 1) readonly buffer EyInBuffer { float eyIn[]; };
layout(std430, binding = 2) readonly buffer EzInBuffer { float ezIn[]; };
layout(std430, binding = 3) readonly buffer BxBuffer { float bx[]; };
layout(std430, binding = 4) readonly buffer ByBuffer { float by[]; };
layout(std430, binding = 5) readonly buffer BzBuffer { float bz[]; };
layout(std430, binding = 6) buffer ExOutBuffer { float exOut[]; };
layout(std430, binding = 7) buffer EyOutBuffer { float eyOut[]; };
layout(std430, binding = 8) buffer EzOutBuffer { float ezOut[]; };
uniform int nx;
uniform int ny;
uniform int nz;
uniform float eFactor;
void main() {
    int idx = int(gl_GlobalInvocationID.x);
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
)glsl";

// The compressible Euler equations' gas-dynamics relations, inlined as GLSL
// functions shared by the sweep kernel below. Matches
// Physics/Fluids/Eulerian3D.cpp's own (double-precision) copies exactly.
constexpr std::string_view kEulerGasDynamicsFunctions = R"glsl(
float eulerPressureOf(float density, float momentumNormal, float momentumTangent1,
                      float momentumTangent2, float energy, float gamma) {
    float u = momentumNormal / density;
    float v = momentumTangent1 / density;
    float w = momentumTangent2 / density;
    float kinetic = 0.5 * density * (u * u + v * v + w * w);
    return (gamma - 1.0) * (energy - kinetic);
}

float eulerSoundSpeedOf(float density, float pressure, float gamma) {
    return sqrt(gamma * pressure / density);
}
)glsl";

constexpr std::string_view kEulerianFluid3DSweepHeader = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer DensityBuffer { float density[]; };
layout(std430, binding = 1) readonly buffer MomentumNormalBuffer { float momentumNormal[]; };
layout(std430, binding = 2) readonly buffer MomentumTangent1Buffer { float momentumTangent1[]; };
layout(std430, binding = 3) readonly buffer MomentumTangent2Buffer { float momentumTangent2[]; };
layout(std430, binding = 4) readonly buffer EnergyBuffer { float energy[]; };
layout(std430, binding = 5) buffer NextDensityBuffer { float nextDensity[]; };
layout(std430, binding = 6) buffer NextMomentumNormalBuffer { float nextMomentumNormal[]; };
layout(std430, binding = 7) buffer NextMomentumTangent1Buffer { float nextMomentumTangent1[]; };
layout(std430, binding = 8) buffer NextMomentumTangent2Buffer { float nextMomentumTangent2[]; };
layout(std430, binding = 9) buffer NextEnergyBuffer { float nextEnergy[]; };
uniform int nx;
uniform int ny;
uniform int nz;
uniform int axis;
uniform float gamma;
uniform float dtOverSpacing;
)glsl";

// One dimensional-split sweep along `axis` (0/1/2), first-order Rusanov
// flux, periodic wraparound via modular index arithmetic. Matches
// Physics/Fluids/Eulerian3D.cpp's sweep() exactly for one axis. Each face
// flux is recomputed independently by both cells that share it (no shared
// flux buffer), the same embarrassingly-parallel-per-cell shape every
// other grid kernel here uses.
constexpr std::string_view kEulerianFluid3DSweepMain = R"glsl(
void main() {
    int idx = int(gl_GlobalInvocationID.x);
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

    float pHere = eulerPressureOf(density[idx], momentumNormal[idx], momentumTangent1[idx],
                                  momentumTangent2[idx], energy[idx], gamma);
    float pLeft = eulerPressureOf(density[leftIdx], momentumNormal[leftIdx],
                                  momentumTangent1[leftIdx], momentumTangent2[leftIdx],
                                  energy[leftIdx], gamma);
    float pRight = eulerPressureOf(density[rightIdx], momentumNormal[rightIdx],
                                   momentumTangent1[rightIdx], momentumTangent2[rightIdx],
                                   energy[rightIdx], gamma);

    float uHere = momentumNormal[idx] / density[idx];
    float uLeft = momentumNormal[leftIdx] / density[leftIdx];
    float uRight = momentumNormal[rightIdx] / density[rightIdx];
    float soundHere = eulerSoundSpeedOf(density[idx], pHere, gamma);
    float soundLeft = eulerSoundSpeedOf(density[leftIdx], pLeft, gamma);
    float soundRight = eulerSoundSpeedOf(density[rightIdx], pRight, gamma);

    // fluxOf(state) = { momentumNormal, momentumNormal*u + p, momentumNormal*v,
    //                    momentumNormal*w, u*(energy + p) }, inlined at each
    // of the three cells (here/left/right) rather than as its own function.
    float fDensityHere = momentumNormal[idx];
    float fMomentumNormalHere = momentumNormal[idx] * uHere + pHere;
    float fMomentumTangent1Here = momentumNormal[idx] * (momentumTangent1[idx] / density[idx]);
    float fMomentumTangent2Here = momentumNormal[idx] * (momentumTangent2[idx] / density[idx]);
    float fEnergyHere = uHere * (energy[idx] + pHere);

    float fDensityLeft = momentumNormal[leftIdx];
    float fMomentumNormalLeft = momentumNormal[leftIdx] * uLeft + pLeft;
    float fMomentumTangent1Left =
        momentumNormal[leftIdx] * (momentumTangent1[leftIdx] / density[leftIdx]);
    float fMomentumTangent2Left =
        momentumNormal[leftIdx] * (momentumTangent2[leftIdx] / density[leftIdx]);
    float fEnergyLeft = uLeft * (energy[leftIdx] + pLeft);

    float fDensityRight = momentumNormal[rightIdx];
    float fMomentumNormalRight = momentumNormal[rightIdx] * uRight + pRight;
    float fMomentumTangent1Right =
        momentumNormal[rightIdx] * (momentumTangent1[rightIdx] / density[rightIdx]);
    float fMomentumTangent2Right =
        momentumNormal[rightIdx] * (momentumTangent2[rightIdx] / density[rightIdx]);
    float fEnergyRight = uRight * (energy[rightIdx] + pRight);

    float maxSpeedLeftFace = max(abs(uLeft) + soundLeft, abs(uHere) + soundHere);
    float maxSpeedRightFace = max(abs(uHere) + soundHere, abs(uRight) + soundRight);

    float fluxLeftDensity = 0.5 * (fDensityLeft + fDensityHere) -
                            0.5 * maxSpeedLeftFace * (density[idx] - density[leftIdx]);
    float fluxLeftMomentumNormal =
        0.5 * (fMomentumNormalLeft + fMomentumNormalHere) -
        0.5 * maxSpeedLeftFace * (momentumNormal[idx] - momentumNormal[leftIdx]);
    float fluxLeftMomentumTangent1 =
        0.5 * (fMomentumTangent1Left + fMomentumTangent1Here) -
        0.5 * maxSpeedLeftFace * (momentumTangent1[idx] - momentumTangent1[leftIdx]);
    float fluxLeftMomentumTangent2 =
        0.5 * (fMomentumTangent2Left + fMomentumTangent2Here) -
        0.5 * maxSpeedLeftFace * (momentumTangent2[idx] - momentumTangent2[leftIdx]);
    float fluxLeftEnergy = 0.5 * (fEnergyLeft + fEnergyHere) -
                           0.5 * maxSpeedLeftFace * (energy[idx] - energy[leftIdx]);

    float fluxRightDensity = 0.5 * (fDensityHere + fDensityRight) -
                             0.5 * maxSpeedRightFace * (density[rightIdx] - density[idx]);
    float fluxRightMomentumNormal =
        0.5 * (fMomentumNormalHere + fMomentumNormalRight) -
        0.5 * maxSpeedRightFace * (momentumNormal[rightIdx] - momentumNormal[idx]);
    float fluxRightMomentumTangent1 =
        0.5 * (fMomentumTangent1Here + fMomentumTangent1Right) -
        0.5 * maxSpeedRightFace * (momentumTangent1[rightIdx] - momentumTangent1[idx]);
    float fluxRightMomentumTangent2 =
        0.5 * (fMomentumTangent2Here + fMomentumTangent2Right) -
        0.5 * maxSpeedRightFace * (momentumTangent2[rightIdx] - momentumTangent2[idx]);
    float fluxRightEnergy = 0.5 * (fEnergyHere + fEnergyRight) -
                            0.5 * maxSpeedRightFace * (energy[rightIdx] - energy[idx]);

    nextDensity[idx] = density[idx] - dtOverSpacing * (fluxRightDensity - fluxLeftDensity);
    nextMomentumNormal[idx] =
        momentumNormal[idx] -
        dtOverSpacing * (fluxRightMomentumNormal - fluxLeftMomentumNormal);
    nextMomentumTangent1[idx] =
        momentumTangent1[idx] -
        dtOverSpacing * (fluxRightMomentumTangent1 - fluxLeftMomentumTangent1);
    nextMomentumTangent2[idx] =
        momentumTangent2[idx] -
        dtOverSpacing * (fluxRightMomentumTangent2 - fluxLeftMomentumTangent2);
    nextEnergy[idx] = energy[idx] - dtOverSpacing * (fluxRightEnergy - fluxLeftEnergy);
}
)glsl";

// A batched, power-of-two, radix-2 Cooley-Tukey FFT: two shaders dispatched
// in a host-side loop (kFFTBitReversalPermuteSource once, then
// kFFTButterflyStageSource once per stage), matching Math/FFT.hpp's
// detail::fftImpl exactly per batch. Out-of-place throughout, the same
// ping-pong-between-two-buffers convention MetalBackend::fftBatched uses.
// Every thread bounds-checks against an explicit `total`/`pairCount`
// uniform rather than an in-shader buffer-length query, since the
// dispatch's actual thread count is always rounded up to a workgroup
// multiple.
constexpr std::string_view kFFTBitReversalPermuteSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer RealBuffer { float real_[]; };
layout(std430, binding = 1) readonly buffer ImagBuffer { float imag_[]; };
layout(std430, binding = 2) buffer NextRealBuffer { float nextReal[]; };
layout(std430, binding = 3) buffer NextImagBuffer { float nextImag[]; };
uniform int length;
uniform int bitCount;
uniform int total;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(total)) return;
    uint batch = idx / uint(length);
    uint i = idx % uint(length);

    uint reversed = 0u;
    uint value = i;
    for (int b = 0; b < bitCount; ++b) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }

    uint destination = batch * uint(length) + reversed;
    nextReal[destination] = real_[idx];
    nextImag[destination] = imag_[idx];
}
)glsl";

constexpr std::string_view kFFTButterflyStageSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer RealBuffer { float real_[]; };
layout(std430, binding = 1) readonly buffer ImagBuffer { float imag_[]; };
layout(std430, binding = 2) buffer NextRealBuffer { float nextReal[]; };
layout(std430, binding = 3) buffer NextImagBuffer { float nextImag[]; };
uniform int length;
uniform int len;
uniform float angle;
uniform int pairCount;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(pairCount)) return;
    uint half_ = uint(len) / 2u;
    uint pairsPerBatch = uint(length) / 2u;
    uint batch = tid / pairsPerBatch;
    uint pairInBatch = tid % pairsPerBatch;
    uint blockIndex = pairInBatch / half_;
    uint k = pairInBatch % half_;
    uint blockStart = blockIndex * uint(len);
    uint base = batch * uint(length);
    uint i0 = base + blockStart + k;
    uint i1 = i0 + half_;

    float twiddleReal = cos(float(k) * angle);
    float twiddleImag = sin(float(k) * angle);
    float evenReal = real_[i0];
    float evenImag = imag_[i0];
    float oddReal0 = real_[i1];
    float oddImag0 = imag_[i1];
    float oddReal = oddReal0 * twiddleReal - oddImag0 * twiddleImag;
    float oddImag = oddReal0 * twiddleImag + oddImag0 * twiddleReal;

    nextReal[i0] = evenReal + oddReal;
    nextImag[i0] = evenImag + oddImag;
    nextReal[i1] = evenReal - oddReal;
    nextImag[i1] = evenImag - oddImag;
}
)glsl";

constexpr std::string_view kFFTScaleSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer RealBuffer { float real_[]; };
layout(std430, binding = 1) readonly buffer ImagBuffer { float imag_[]; };
layout(std430, binding = 2) buffer NextRealBuffer { float nextReal[]; };
layout(std430, binding = 3) buffer NextImagBuffer { float nextImag[]; };
uniform float scale;
uniform int total;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(total)) return;
    nextReal[tid] = real_[tid] * scale;
    nextImag[tid] = imag_[tid] * scale;
}
)glsl";

// Dense matrix-vector and matrix-matrix multiply, row-major, one thread
// per output element -- a reference kernel, not a tiled/shared-memory
// GEMM (see MetalBackend.mm's own comment on the same tradeoff).
constexpr std::string_view kMatVecSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer MatBuffer { float mat[]; };
layout(std430, binding = 1) readonly buffer VecBuffer { float vec[]; };
layout(std430, binding = 2) buffer ResultBuffer { float result[]; };
uniform int rows;
uniform int cols;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(rows)) return;
    float total = 0.0;
    for (int c = 0; c < cols; ++c) {
        total += mat[tid * uint(cols) + uint(c)] * vec[c];
    }
    result[tid] = total;
}
)glsl";

constexpr std::string_view kMatMulSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer ABuffer { float a[]; };
layout(std430, binding = 1) readonly buffer BBuffer { float b[]; };
layout(std430, binding = 2) buffer ResultBuffer { float result[]; };
uniform int aRows;
uniform int aCols;
uniform int bCols;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint total = uint(aRows) * uint(bCols);
    if (tid >= total) return;
    uint r = tid / uint(bCols);
    uint c = tid % uint(bCols);
    float sum = 0.0;
    for (int k = 0; k < aCols; ++k) {
        sum += a[r * uint(aCols) + uint(k)] * b[uint(k) * uint(bCols) + c];
    }
    result[tid] = sum;
}
)glsl";

// Dense LU decomposition with partial pivoting: three shaders dispatched
// in a host-side loop, one iteration per pivot column
// (OpenGLBackend::luDecomposeGpu), matching Math/LinearSolve.hpp's
// luDecompose exactly. kArgMaxAbsColumnSource is kMinIndexSource's own
// two-pass reduce-then-finish-on-CPU shape, adapted to a strided column
// read, a max (not min) comparison, and recovering the winning value (not
// just the index) for luDecomposeGpu's singularity check.
constexpr std::string_view kArgMaxAbsColumnSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer MatBuffer { float mat[]; };
layout(std430, binding = 1) buffer OutValueBuffer { float partialValues[]; };
layout(std430, binding = 2) buffer OutIndexBuffer { float partialIndices[]; };
uniform int n;
uniform int column;
uniform int startRow;
uniform int activeRows;
shared float scratchValue[256];
shared uint scratchIndex[256];
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint local = gl_LocalInvocationID.x;
    if (i < uint(activeRows)) {
        uint row = uint(startRow) + i;
        scratchValue[local] = abs(mat[row * uint(n) + uint(column)]);
        scratchIndex[local] = row;
    } else {
        scratchValue[local] = -1.0;
        scratchIndex[local] = 0u;
    }
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride && scratchValue[local + stride] > scratchValue[local]) {
            scratchValue[local] = scratchValue[local + stride];
            scratchIndex[local] = scratchIndex[local + stride];
        }
        barrier();
    }
    if (local == 0u) {
        partialValues[gl_WorkGroupID.x] = scratchValue[0];
        partialIndices[gl_WorkGroupID.x] = float(scratchIndex[0]);
    }
}
)glsl";

constexpr std::string_view kSwapRowsSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer MatBuffer { float mat[]; };
layout(std430, binding = 1) buffer NextMatBuffer { float nextMat[]; };
uniform int n;
uniform int rowA;
uniform int rowB;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint total = uint(n) * uint(n);
    if (tid >= total) return;
    uint row = tid / uint(n);
    uint col = tid % uint(n);
    uint sourceRow = row;
    if (row == uint(rowA)) {
        sourceRow = uint(rowB);
    } else if (row == uint(rowB)) {
        sourceRow = uint(rowA);
    }
    nextMat[tid] = mat[sourceRow * uint(n) + col];
}
)glsl";

// The trailing-submatrix rank-1 update for pivot column k. Matches
// Math/LinearSolve.hpp's luDecompose inner loop exactly; see
// MetalBackend.mm's lu_elimination_step_kernel for the same comment on why
// this dispatches over the whole matrix with a passthrough guard.
constexpr std::string_view kLuEliminationStepSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer MatBuffer { float mat[]; };
layout(std430, binding = 1) buffer NextMatBuffer { float nextMat[]; };
uniform int n;
uniform int k;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint total = uint(n) * uint(n);
    if (tid >= total) return;
    uint row = tid / uint(n);
    uint col = tid % uint(n);
    if (row <= uint(k) || col < uint(k)) {
        nextMat[tid] = mat[tid];
        return;
    }
    float factor = mat[row * uint(n) + uint(k)] / mat[uint(k) * uint(n) + uint(k)];
    if (col == uint(k)) {
        nextMat[tid] = factor;
    } else {
        nextMat[tid] = mat[row * uint(n) + col] - factor * mat[uint(k) * uint(n) + col];
    }
}
)glsl";

// Dense Cholesky factorization: two shaders dispatched in a host-side
// loop, one iteration per column (OpenGLBackend::choleskyDecomposeGpu),
// matching Math/LinearSolve.hpp's choleskyDecompose exactly. Both
// accumulate into one resident `l` buffer in place across the whole
// decomposition; see MetalBackend.mm's cholesky_diagonal_kernel comment
// for why (and for the -1 sentinel / early-readback singularity check).
constexpr std::string_view kCholeskyDiagonalSource = R"glsl(
#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) readonly buffer ABuffer { float a[]; };
layout(std430, binding = 1) buffer LBuffer { float l[]; };
uniform int n;
uniform int j;
void main() {
    float total = a[uint(j) * uint(n) + uint(j)];
    for (int k = 0; k < j; ++k) {
        float ljk = l[uint(j) * uint(n) + uint(k)];
        total -= ljk * ljk;
    }
    if (!(total > 0.0) || isnan(total) || isinf(total)) {
        l[uint(j) * uint(n) + uint(j)] = -1.0;
        return;
    }
    l[uint(j) * uint(n) + uint(j)] = sqrt(total);
}
)glsl";

constexpr std::string_view kCholeskyColumnSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer ABuffer { float a[]; };
layout(std430, binding = 1) buffer LBuffer { float l[]; };
uniform int n;
uniform int j;
void main() {
    uint i = uint(j) + 1u + gl_GlobalInvocationID.x;
    if (i >= uint(n)) return;
    float total = a[i * uint(n) + uint(j)];
    for (int k = 0; k < j; ++k) {
        total -= l[i * uint(n) + uint(k)] * l[uint(j) * uint(n) + uint(k)];
    }
    l[i * uint(n) + uint(j)] = total / l[uint(j) * uint(n) + uint(j)];
}
)glsl";

// Dense QR decomposition via Householder reflections: three shaders
// dispatched in a host-side loop, one iteration per column
// (OpenGLBackend::qrDecomposeGpu). See MetalBackend.mm's own comment on
// this kernel family for the full rationale; ported here identically,
// including qr_apply_left's passthrough copying the *whole* column (not
// just the rows before k) for an already-finalized column -- easy to get
// wrong (an earlier version of the Metal kernel did), since the output
// buffer is separate from the input and every row needs an explicit copy.
constexpr std::string_view kQrColumnNormSquaredSource = R"glsl(
#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) readonly buffer RBuffer { float r[]; };
layout(std430, binding = 1) buffer OutputBuffer { float output_[]; };
uniform int rows;
uniform int cols;
uniform int k;
void main() {
    float sumSquares = 0.0;
    for (int i = k; i < rows; ++i) {
        float value = r[i * cols + k];
        sumSquares += value * value;
    }
    output_[0] = sumSquares;
    output_[1] = r[k * cols + k];
}
)glsl";

constexpr std::string_view kQrApplyLeftSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer RBuffer { float r[]; };
layout(std430, binding = 1) buffer NextRBuffer { float nextR[]; };
uniform int rows;
uniform int cols;
uniform int k;
uniform float alpha;
uniform float vNormSquared;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(cols)) return;
    if (int(tid) < k) {
        for (int i = 0; i < rows; ++i) {
            nextR[i * cols + int(tid)] = r[i * cols + int(tid)];
        }
        return;
    }
    for (int i = 0; i < k; ++i) {
        nextR[i * cols + int(tid)] = r[i * cols + int(tid)];
    }
    float dotProduct = 0.0;
    for (int i = k; i < rows; ++i) {
        float vi = (i == k) ? (r[k * cols + k] - alpha) : r[i * cols + k];
        dotProduct += vi * r[i * cols + int(tid)];
    }
    float factor = 2.0 * dotProduct / vNormSquared;
    for (int i = k; i < rows; ++i) {
        float vi = (i == k) ? (r[k * cols + k] - alpha) : r[i * cols + k];
        nextR[i * cols + int(tid)] = r[i * cols + int(tid)] - factor * vi;
    }
}
)glsl";

constexpr std::string_view kQrAccumulateQSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer RBuffer { float r[]; };
layout(std430, binding = 1) readonly buffer QBuffer { float q[]; };
layout(std430, binding = 2) buffer NextQBuffer { float nextQ[]; };
uniform int rows;
uniform int cols;
uniform int k;
uniform float alpha;
uniform float vNormSquared;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(rows)) return;
    int row = int(tid);
    for (int i = 0; i < k; ++i) {
        nextQ[row * rows + i] = q[row * rows + i];
    }
    float dotProduct = 0.0;
    for (int i = k; i < rows; ++i) {
        float vi = (i == k) ? (r[k * cols + k] - alpha) : r[i * cols + k];
        dotProduct += vi * q[row * rows + i];
    }
    float factor = 2.0 * dotProduct / vNormSquared;
    for (int i = k; i < rows; ++i) {
        float vi = (i == k) ? (r[k * cols + k] - alpha) : r[i * cols + k];
        nextQ[row * rows + i] = q[row * rows + i] - factor * vi;
    }
}
)glsl";

// The cyclic Jacobi eigenvalue algorithm's round-robin/tournament-ordered
// parallel form. See MetalBackend.mm's own comment on this kernel family
// for the full rationale (why this is two passes, not one, and why both
// recompute the same rotation from the pair's original pre-round values).
constexpr std::string_view kJacobiRotationFunction = R"glsl(
void jacobiRotationOf(float app, float aqq, float apq, out float cosOut, out float sinOut) {
    float theta = (aqq - app) / (2.0 * apq);
    float sign = (theta < 0.0) ? -1.0 : 1.0;
    float t = sign / (abs(theta) + sqrt(theta * theta + 1.0));
    cosOut = 1.0 / sqrt(t * t + 1.0);
    sinOut = t * cosOut;
}
)glsl";

constexpr std::string_view kJacobiEigenColumnMixHeader = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer ABuffer { float a[]; };
layout(std430, binding = 1) readonly buffer VBuffer { float v[]; };
layout(std430, binding = 2) readonly buffer PairBuffer { uint pairOf[]; };
layout(std430, binding = 3) buffer NextBBuffer { float nextB[]; };
layout(std430, binding = 4) buffer NextVBuffer { float nextV[]; };
uniform int n;
uniform float epsilon;
)glsl";

constexpr std::string_view kJacobiEigenColumnMixMain = R"glsl(
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(n)) return;
    int c = int(tid);
    int partner = int(pairOf[c]);
    if (partner == c) {
        for (int i = 0; i < n; ++i) {
            nextB[i * n + c] = a[i * n + c];
            nextV[i * n + c] = v[i * n + c];
        }
        return;
    }
    int p = min(c, partner);
    int q = max(c, partner);
    float apq = a[p * n + q];
    if (abs(apq) <= epsilon) {
        for (int i = 0; i < n; ++i) {
            nextB[i * n + c] = a[i * n + c];
            nextV[i * n + c] = v[i * n + c];
        }
        return;
    }
    float cosT;
    float sinT;
    jacobiRotationOf(a[p * n + p], a[q * n + q], apq, cosT, sinT);
    if (c == p) {
        for (int i = 0; i < n; ++i) {
            nextB[i * n + p] = cosT * a[i * n + p] - sinT * a[i * n + q];
            nextV[i * n + p] = cosT * v[i * n + p] - sinT * v[i * n + q];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            nextB[i * n + q] = sinT * a[i * n + p] + cosT * a[i * n + q];
            nextV[i * n + q] = sinT * v[i * n + p] + cosT * v[i * n + q];
        }
    }
}
)glsl";

constexpr std::string_view kJacobiEigenRowMixHeader = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer ABuffer { float a[]; };
layout(std430, binding = 1) readonly buffer BBuffer { float b[]; };
layout(std430, binding = 2) readonly buffer PairBuffer { uint pairOf[]; };
layout(std430, binding = 3) buffer NextABuffer { float nextA[]; };
uniform int n;
uniform float epsilon;
)glsl";

constexpr std::string_view kJacobiEigenRowMixMain = R"glsl(
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(n)) return;
    int r = int(tid);
    int partner = int(pairOf[r]);
    if (partner == r) {
        for (int j = 0; j < n; ++j) {
            nextA[r * n + j] = b[r * n + j];
        }
        return;
    }
    int p = min(r, partner);
    int q = max(r, partner);
    float apq = a[p * n + q];
    if (abs(apq) <= epsilon) {
        for (int j = 0; j < n; ++j) {
            nextA[r * n + j] = b[r * n + j];
        }
        return;
    }
    float cosT;
    float sinT;
    jacobiRotationOf(a[p * n + p], a[q * n + q], apq, cosT, sinT);
    if (r == p) {
        for (int j = 0; j < n; ++j) {
            nextA[p * n + j] = cosT * b[p * n + j] - sinT * b[q * n + j];
        }
    } else {
        for (int j = 0; j < n; ++j) {
            nextA[q * n + j] = sinT * b[p * n + j] + cosT * b[q * n + j];
        }
    }
}
)glsl";

constexpr std::string_view kJacobiSvdRoundSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer ABuffer { float a[]; };
layout(std430, binding = 1) readonly buffer VBuffer { float v[]; };
layout(std430, binding = 2) readonly buffer PairBuffer { uint pairOf[]; };
layout(std430, binding = 3) buffer NextABuffer { float nextA[]; };
layout(std430, binding = 4) buffer NextVBuffer { float nextV[]; };
uniform int rows;
uniform int cols;
uniform float epsilon;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(cols)) return;
    int c = int(tid);
    int partner = int(pairOf[c]);
    if (partner == c) {
        for (int i = 0; i < rows; ++i) {
            nextA[i * cols + c] = a[i * cols + c];
        }
        for (int i = 0; i < cols; ++i) {
            nextV[i * cols + c] = v[i * cols + c];
        }
        return;
    }
    int p = min(c, partner);
    int q = max(c, partner);
    float alpha = 0.0;
    float beta = 0.0;
    float gamma = 0.0;
    for (int i = 0; i < rows; ++i) {
        float ap = a[i * cols + p];
        float aq = a[i * cols + q];
        alpha += ap * ap;
        beta += aq * aq;
        gamma += ap * aq;
    }
    float threshold = epsilon * sqrt(alpha * beta);
    if (abs(gamma) <= threshold) {
        for (int i = 0; i < rows; ++i) {
            nextA[i * cols + c] = a[i * cols + c];
        }
        for (int i = 0; i < cols; ++i) {
            nextV[i * cols + c] = v[i * cols + c];
        }
        return;
    }
    float zeta = (beta - alpha) / (2.0 * gamma);
    float sign = (zeta < 0.0) ? -1.0 : 1.0;
    float t = sign / (abs(zeta) + sqrt(1.0 + zeta * zeta));
    float cosT = 1.0 / sqrt(1.0 + t * t);
    float sinT = cosT * t;
    if (c == p) {
        for (int i = 0; i < rows; ++i) {
            nextA[i * cols + p] = cosT * a[i * cols + p] - sinT * a[i * cols + q];
        }
        for (int i = 0; i < cols; ++i) {
            nextV[i * cols + p] = cosT * v[i * cols + p] - sinT * v[i * cols + q];
        }
    } else {
        for (int i = 0; i < rows; ++i) {
            nextA[i * cols + q] = sinT * a[i * cols + p] + cosT * a[i * cols + q];
        }
        for (int i = 0; i < cols; ++i) {
            nextV[i * cols + q] = sinT * v[i * cols + p] + cosT * v[i * cols + q];
        }
    }
}
)glsl";

// Batched, independent, pointwise evaluation: one invocation per element of
// `x`, no reduction, no multi-pass structure. See ComputeBackend.hpp's own
// comment on this family for why erf/erfc/gamma/logGamma use these
// well-conditioned approximations rather than a C99 math library (which
// GLSL, like MSL, does not have), and why besselJ/besselY have no batched
// kernel here at all. Ported directly from MetalBackend.mm's own copy of
// these functions (duplicated rather than shared, matching this codebase's
// convention of small per-backend-file helpers).
constexpr std::string_view kBatchEvaluationFunctions = R"glsl(
const float kPi = 3.14159265358979323846;

float erfcOf(float x) {
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float poly = t * (0.254829592 +
                      t * (-0.284496736 +
                           t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    float y = poly * exp(-ax * ax);
    return (x >= 0.0) ? y : (2.0 - y);
}

float erfOf(float x) {
    return 1.0 - erfcOf(x);
}

float lgammaPositiveOf(float x) {
    float y = x;
    float tmp = x + 5.5;
    tmp -= (x + 0.5) * log(tmp);
    float ser = 1.000000000190015;
    y += 1.0;
    ser += 76.18009172947146 / y;
    y += 1.0;
    ser += -86.50532032941677 / y;
    y += 1.0;
    ser += 24.01409824083091 / y;
    y += 1.0;
    ser += -1.231739572450155 / y;
    y += 1.0;
    ser += 0.1208650973866179e-2 / y;
    y += 1.0;
    ser += -0.5395239384953e-5 / y;
    return -tmp + log(2.5066282746310005 * ser / x);
}

float lgammaOf(float x) {
    if (x > 0.0) {
        return lgammaPositiveOf(x);
    }
    return log(kPi) - log(abs(sin(kPi * x))) - lgammaPositiveOf(1.0 - x);
}

float gammaOf(float x) {
    if (x > 0.0) {
        return exp(lgammaPositiveOf(x));
    }
    return kPi / (sin(kPi * x) * exp(lgammaPositiveOf(1.0 - x)));
}
)glsl";

// Shared buffer layout for the four erf/erfc/gamma/logGamma shaders;
// concatenated with kBatchEvaluationFunctions and each one's own main() in
// create(), the same split kCubicSplineFunctions already uses.
constexpr std::string_view kBatchScalarHeader = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer XBuffer { float x[]; };
layout(std430, binding = 1) buffer ResultBuffer { float result[]; };
)glsl";

constexpr std::string_view kBatchErfMain = R"glsl(
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(x.length())) return;
    result[tid] = erfOf(x[tid]);
}
)glsl";

constexpr std::string_view kBatchErfcMain = R"glsl(
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(x.length())) return;
    result[tid] = erfcOf(x[tid]);
}
)glsl";

constexpr std::string_view kBatchGammaMain = R"glsl(
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(x.length())) return;
    result[tid] = gammaOf(x[tid]);
}
)glsl";

constexpr std::string_view kBatchLogGammaMain = R"glsl(
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(x.length())) return;
    result[tid] = lgammaOf(x[tid]);
}
)glsl";

// The same upward recurrence Math/SpecialFunctions.hpp's legendreP uses,
// one invocation per point, n/m fixed for the whole batch.
constexpr std::string_view kBatchLegendrePSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer XBuffer { float x[]; };
layout(std430, binding = 1) buffer ResultBuffer { float result[]; };
uniform int n;
uniform int m;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(x.length())) return;
    float value = x[tid];

    float pmm = 1.0;
    if (m > 0) {
        float oneMinusX2 = (1.0 - value) * (1.0 + value);
        float somx2 = sqrt(oneMinusX2);
        float fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
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

    float pll = 0.0;
    for (int ll = m + 2; ll <= n; ++ll) {
        pll = (value * float(2 * ll - 1) * pmmp1 - float(ll + m - 1) * pmm) / float(ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    result[tid] = pll;
}
)glsl";

// Horner's method, one invocation per point, the same fixed coefficients
// evaluated at every point.
constexpr std::string_view kBatchPolynomialEvalSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer CoefficientsBuffer { float coefficients[]; };
layout(std430, binding = 1) readonly buffer XBuffer { float x[]; };
layout(std430, binding = 2) buffer ResultBuffer { float result[]; };
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(x.length())) return;
    int coefficientCount = coefficients.length();
    float value = coefficients[coefficientCount - 1];
    for (int i = coefficientCount - 1; i-- > 0;) {
        value = value * x[tid] + coefficients[i];
    }
    result[tid] = value;
}
)glsl";

// Held flat outside the table and a binary search per point, matching
// CubicSpline::operator() exactly; the fixed knots/second derivatives are
// the same for every point in the batch.
constexpr std::string_view kBatchCubicSplineEvalSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer KnotsXBuffer { float knotsX[]; };
layout(std430, binding = 1) readonly buffer KnotsYBuffer { float knotsY[]; };
layout(std430, binding = 2) readonly buffer SecondDerivativesBuffer { float secondDerivatives[]; };
layout(std430, binding = 3) readonly buffer QueryXBuffer { float queryX[]; };
layout(std430, binding = 4) buffer ResultBuffer { float result[]; };
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(queryX.length())) return;
    int knotCount = knotsX.length();
    float at = queryX[tid];
    if (at <= knotsX[0]) {
        result[tid] = knotsY[0];
        return;
    }
    if (knotsX[knotCount - 1] <= at) {
        result[tid] = knotsY[knotCount - 1];
        return;
    }

    int lo = 0;
    int hi = knotCount;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (knotsX[mid] <= at) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int i = (lo == 0) ? 0 : lo - 1;
    i = min(i, knotCount - 2);

    float width = knotsX[i + 1] - knotsX[i];
    float left = (knotsX[i + 1] - at) / width;
    float right = (at - knotsX[i]) / width;

    result[tid] = left * knotsY[i] + right * knotsY[i + 1] +
                 ((left * left * left - left) * secondDerivatives[i] +
                  (right * right * right - right) * secondDerivatives[i + 1]) *
                     (width * width) / 6.0;
}
)glsl";

// Philox4x32-10 (Salmon, Moraes, Hadjidoukas & Schulten 2011): a
// counter-based RNG, embarrassingly parallel by construction (invocation
// tid's output depends only on (seed, offset + tid), no shared state, no
// sequential dependency between invocations). Ported directly from
// MetalBackend.mm's own copy of these functions (duplicated rather than
// shared, matching this codebase's convention of small per-backend-file
// helpers); mulhi32 uses only 16-bit-limb schoolbook long multiplication
// rather than a 64-bit intermediate, since GLSL 430 (this engine's compute
// shader floor) has no portable 64-bit integer type.
constexpr std::string_view kBatchRandomFunctions = R"glsl(
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

void philox4x32_10(inout uint c0, inout uint c1, inout uint c2, inout uint c3,
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
)glsl";

// seed/offset travel as a 4-element uint SSBO ([seedLo, seedHi, offsetLo,
// offsetHi]) rather than uniforms: ComputeShader has no uint uniform
// setter (only float/int/vec4), and this reuses the uint-buffer upload
// path jacobiEigenSymmetricGpu/jacobiSvdGpu's round-robin schedule already
// established, rather than introducing an int/uint bit-reinterpretation
// uniform trick for the first time here. Split into a header (compiled
// first) and each kernel's own main body, concatenated with
// kBatchRandomFunctions between them in create(), the same reason
// kSphDensityPressureHeader/Main is.
constexpr std::string_view kBatchRandomHeader = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer ResultBuffer { float result[]; };
layout(std430, binding = 1) readonly buffer ParamsBuffer { uint params[]; };
)glsl";

constexpr std::string_view kBatchUniformRealMain = R"glsl(
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(result.length())) return;
    uint seedLo = params[0];
    uint seedHi = params[1];
    uint offsetLo = params[2];
    uint offsetHi = params[3];
    // counter = offset + tid, as a 64-bit value split into two 32-bit
    // words with an explicit carry, matching CpuBackend's std::uint64_t
    // addition exactly (the standard unsigned-overflow idiom: the sum
    // wrapped iff it came out smaller than either original operand).
    uint c0 = offsetLo + tid;
    uint carry = (c0 < offsetLo) ? 1u : 0u;
    uint c1 = offsetHi + carry;
    uint c2 = 0u;
    uint c3 = 0u;
    philox4x32_10(c0, c1, c2, c3, seedLo, seedHi);
    result[tid] = float(c0) * 2.3283064365386963e-10;
}
)glsl";

constexpr std::string_view kBatchNormalMain = R"glsl(
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(result.length())) return;
    uint seedLo = params[0];
    uint seedHi = params[1];
    uint offsetLo = params[2];
    uint offsetHi = params[3];
    uint c0 = offsetLo + tid;
    uint carry = (c0 < offsetLo) ? 1u : 0u;
    uint c1 = offsetHi + carry;
    uint c2 = 0u;
    uint c3 = 0u;
    philox4x32_10(c0, c1, c2, c3, seedLo, seedHi);
    // u1 in (0, 1], never exactly 0, so log(u1) is always finite: offsetting
    // the 32-bit word up by one before scaling avoids a special case for the
    // all-zero output word instead, matching CpuBackend::batchNormal exactly.
    float u1 = (float(c0) + 1.0) * 2.3283064365386963e-10;
    float u2 = float(c1) * 2.3283064365386963e-10;
    result[tid] = sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2);
}
)glsl";

// Bitonic sort (Batcher 1968): the standard data-parallel sorting network,
// ported directly from MetalBackend.mm's own copy of this kernel (see its
// comment for the full rationale). One compare-exchange pass per
// invocation, `log2(n) * (log2(n) + 1) / 2` dispatches total
// (OpenGLBackend::sortAscending's own host loop), `n` a power of two --
// the caller pads with +infinity and truncates back.
constexpr std::string_view kBitonicCompareExchangeSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer ValuesBuffer { float values[]; };
uniform int n;
uniform int stageSize;
uniform int stepSize;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(n)) return;
    uint partner = tid ^ uint(stepSize);
    if (partner <= tid || partner >= uint(n)) return;
    bool ascending = (tid & uint(stageSize)) == 0u;
    float a = values[tid];
    float b = values[partner];
    if ((a > b) == ascending) {
        values[tid] = b;
        values[partner] = a;
    }
}
)glsl";

// Geometric-multigrid transfer operators. Equation-independent: matches
// Math/Multigrid.hpp's detail::restrictGrid exactly, one invocation per
// coarse cell.
constexpr std::string_view kMultigridRestrict3DSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer FineBuffer { float fine[]; };
layout(std430, binding = 1) buffer CoarseBuffer { float coarse[]; };
uniform int nx;
uniform int ny;
uniform int nz;
void main() {
    int cnx = nx / 2;
    int cny = ny / 2;
    int cnz = nz / 2;
    int idx = int(gl_GlobalInvocationID.x);
    int total = cnx * cny * cnz;
    if (idx >= total) return;
    int ck = idx % cnz;
    int cj = (idx / cnz) % cny;
    int ci = idx / (cny * cnz);
    float sum = 0.0;
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
    coarse[idx] = sum / 8.0;
}
)glsl";

// Matches Math/Multigrid.hpp's detail::prolongateAndAdd exactly, one
// invocation per fine cell.
constexpr std::string_view kMultigridProlongateAndAdd3DSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer FineBuffer { float fine[]; };
layout(std430, binding = 1) readonly buffer CoarseCorrectionBuffer { float coarseCorrection[]; };
layout(std430, binding = 2) buffer NextFineBuffer { float nextFine[]; };
uniform int nx;
uniform int ny;
uniform int nz;
void main() {
    int idx = int(gl_GlobalInvocationID.x);
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
)glsl";

// Same two-pass shape as kSumSource, tracking (value, index) pairs instead of
// a running total, so a tie within a workgroup keeps whichever came from the
// lower local index (the strict '<' below never replaces on equality) and
// the CPU finish in minIndex() applies the same rule across workgroups.
// Indices travel through the float output buffer (exact for any index this
// engine will ever see) rather than adding a uint buffer type.
constexpr std::string_view kMinIndexSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer InBuffer { float data[]; };
layout(std430, binding = 1) buffer OutValueBuffer { float partialValues[]; };
layout(std430, binding = 2) buffer OutIndexBuffer { float partialIndices[]; };
shared float scratchValue[256];
shared uint scratchIndex[256];
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint local = gl_LocalInvocationID.x;
    uint count = uint(data.length());
    if (i < count) {
        scratchValue[local] = data[i];
        scratchIndex[local] = i;
    } else {
        scratchValue[local] = 3.402823466e+38;
        scratchIndex[local] = 0u;
    }
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride && scratchValue[local + stride] < scratchValue[local]) {
            scratchValue[local] = scratchValue[local + stride];
            scratchIndex[local] = scratchIndex[local + stride];
        }
        barrier();
    }
    if (local == 0u) {
        partialValues[gl_WorkGroupID.x] = scratchValue[0];
        partialIndices[gl_WorkGroupID.x] = float(scratchIndex[0]);
    }
}
)glsl";

constexpr unsigned kWorkGroupSize = 256;

unsigned groupCountFor(std::size_t elementCount) {
    return static_cast<unsigned>((elementCount + kWorkGroupSize - 1) /
                                 static_cast<std::size_t>(kWorkGroupSize));
}

/// An SSBO holding `count` floats, or initialised from `data`. Bound and torn
/// down within a single saxpy()/sum() call; nothing here is meant to outlive
/// one dispatch.
class ShaderBuffer {
public:
    explicit ShaderBuffer(std::span<const float> data) {
        glGenBuffers(1, &m_handle);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(),
                     GL_DYNAMIC_DRAW);
    }

    /// For a GLSL `uint[]` block (the round-robin pairing table
    /// jacobiEigenSymmetricGpu/jacobiSvdGpu upload once per round); every
    /// other buffer here holds `float[]` data.
    explicit ShaderBuffer(std::span<const std::uint32_t> data) {
        glGenBuffers(1, &m_handle);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(data.size() * sizeof(std::uint32_t)),
                     data.data(), GL_DYNAMIC_DRAW);
    }

    explicit ShaderBuffer(std::size_t count) {
        glGenBuffers(1, &m_handle);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(count * sizeof(float)), nullptr,
                     GL_DYNAMIC_DRAW);
    }

    ShaderBuffer(const ShaderBuffer&) = delete;
    ShaderBuffer& operator=(const ShaderBuffer&) = delete;
    ~ShaderBuffer() { glDeleteBuffers(1, &m_handle); }

    void read(std::span<float> out) const {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(out.size() * sizeof(float)),
                           out.data());
    }

    [[nodiscard]] unsigned handle() const noexcept { return m_handle; }

private:
    unsigned m_handle = 0;
};

/// The standard round-robin/tournament pairing schedule (Brent & Luk
/// 1985) for `n` indices; see MetalBackend.mm's own copy of this function
/// for the full rationale (duplicated rather than shared, matching this
/// codebase's convention of small per-backend-file helpers rather than an
/// internal-only shared header).
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

std::unique_ptr<ComputeBackend> OpenGLBackend::create() {
    ContextSettings context;
    context.versionMajor = 4;
    context.versionMinor = 3;

    WindowError windowError;
    std::optional<Window> window = Window::createOffscreen(1, 1, context, &windowError);
    if (!window) {
        logging::debug("OpenGL compute backend unavailable: {}", windowError.message);
        return nullptr;
    }

    std::string shaderError;
    std::optional<ComputeShader> saxpy =
        ComputeShader::compile(kSaxpySource, &shaderError);
    if (!saxpy) {
        logging::debug("OpenGL compute backend unavailable: saxpy shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> sum = ComputeShader::compile(kSumSource, &shaderError);
    if (!sum) {
        logging::debug("OpenGL compute backend unavailable: sum shader: {}", shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> linearCombine =
        ComputeShader::compile(kLinearCombineSource, &shaderError);
    if (!linearCombine) {
        logging::debug("OpenGL compute backend unavailable: linearCombine shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> gravitationalNBody =
        ComputeShader::compile(kGravitationalNBodySource, &shaderError);
    if (!gravitationalNBody) {
        logging::debug(
            "OpenGL compute backend unavailable: gravitationalNBody shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> electricFieldNBody =
        ComputeShader::compile(kElectricFieldNBodySource, &shaderError);
    if (!electricFieldNBody) {
        logging::debug(
            "OpenGL compute backend unavailable: electricFieldNBody shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> magneticFieldNBody =
        ComputeShader::compile(kMagneticFieldNBodySource, &shaderError);
    if (!magneticFieldNBody) {
        logging::debug(
            "OpenGL compute backend unavailable: magneticFieldNBody shader: {}",
            shaderError);
        return nullptr;
    }
    const std::string sphDensityPressureSource = std::string(kSphDensityPressureHeader) +
                                                 std::string(kCubicSplineFunctions) +
                                                 std::string(kSphDensityPressureMain);
    std::optional<ComputeShader> sphDensityPressure =
        ComputeShader::compile(sphDensityPressureSource, &shaderError);
    if (!sphDensityPressure) {
        logging::debug(
            "OpenGL compute backend unavailable: sphDensityPressure shader: {}",
            shaderError);
        return nullptr;
    }
    const std::string sphPressureAccelerationSource =
        std::string(kSphPressureAccelerationHeader) + std::string(kCubicSplineFunctions) +
        std::string(kSphPressureAccelerationMain);
    std::optional<ComputeShader> sphPressureAcceleration =
        ComputeShader::compile(sphPressureAccelerationSource, &shaderError);
    if (!sphPressureAcceleration) {
        logging::debug(
            "OpenGL compute backend unavailable: sphPressureAcceleration shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> heatEquation3DStep =
        ComputeShader::compile(kHeatEquation3DStepSource, &shaderError);
    if (!heatEquation3DStep) {
        logging::debug(
            "OpenGL compute backend unavailable: heatEquation3DStep shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> acoustic3DVelocity =
        ComputeShader::compile(kAcoustic3DVelocitySource, &shaderError);
    if (!acoustic3DVelocity) {
        logging::debug(
            "OpenGL compute backend unavailable: acoustic3DVelocity shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> acoustic3DPressure =
        ComputeShader::compile(kAcoustic3DPressureSource, &shaderError);
    if (!acoustic3DPressure) {
        logging::debug(
            "OpenGL compute backend unavailable: acoustic3DPressure shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> maxwell3DB =
        ComputeShader::compile(kMaxwell3DBSource, &shaderError);
    if (!maxwell3DB) {
        logging::debug("OpenGL compute backend unavailable: maxwell3DB shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> maxwell3DE =
        ComputeShader::compile(kMaxwell3DESource, &shaderError);
    if (!maxwell3DE) {
        logging::debug("OpenGL compute backend unavailable: maxwell3DE shader: {}",
                       shaderError);
        return nullptr;
    }
    const std::string eulerianFluid3DSweepSource =
        std::string(kEulerianFluid3DSweepHeader) +
        std::string(kEulerGasDynamicsFunctions) + std::string(kEulerianFluid3DSweepMain);
    std::optional<ComputeShader> eulerianFluid3DSweep =
        ComputeShader::compile(eulerianFluid3DSweepSource, &shaderError);
    if (!eulerianFluid3DSweep) {
        logging::debug(
            "OpenGL compute backend unavailable: eulerianFluid3DSweep shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> fftBitReversalPermute =
        ComputeShader::compile(kFFTBitReversalPermuteSource, &shaderError);
    if (!fftBitReversalPermute) {
        logging::debug(
            "OpenGL compute backend unavailable: fftBitReversalPermute shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> fftButterflyStage =
        ComputeShader::compile(kFFTButterflyStageSource, &shaderError);
    if (!fftButterflyStage) {
        logging::debug("OpenGL compute backend unavailable: fftButterflyStage shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> fftScale =
        ComputeShader::compile(kFFTScaleSource, &shaderError);
    if (!fftScale) {
        logging::debug("OpenGL compute backend unavailable: fftScale shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> matVec =
        ComputeShader::compile(kMatVecSource, &shaderError);
    if (!matVec) {
        logging::debug("OpenGL compute backend unavailable: matVec shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> matMul =
        ComputeShader::compile(kMatMulSource, &shaderError);
    if (!matMul) {
        logging::debug("OpenGL compute backend unavailable: matMul shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> argMaxAbsColumn =
        ComputeShader::compile(kArgMaxAbsColumnSource, &shaderError);
    if (!argMaxAbsColumn) {
        logging::debug("OpenGL compute backend unavailable: argMaxAbsColumn shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> swapRows =
        ComputeShader::compile(kSwapRowsSource, &shaderError);
    if (!swapRows) {
        logging::debug("OpenGL compute backend unavailable: swapRows shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> luEliminationStep =
        ComputeShader::compile(kLuEliminationStepSource, &shaderError);
    if (!luEliminationStep) {
        logging::debug("OpenGL compute backend unavailable: luEliminationStep shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> choleskyDiagonal =
        ComputeShader::compile(kCholeskyDiagonalSource, &shaderError);
    if (!choleskyDiagonal) {
        logging::debug("OpenGL compute backend unavailable: choleskyDiagonal shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> choleskyColumn =
        ComputeShader::compile(kCholeskyColumnSource, &shaderError);
    if (!choleskyColumn) {
        logging::debug("OpenGL compute backend unavailable: choleskyColumn shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> qrColumnNormSquared =
        ComputeShader::compile(kQrColumnNormSquaredSource, &shaderError);
    if (!qrColumnNormSquared) {
        logging::debug(
            "OpenGL compute backend unavailable: qrColumnNormSquared shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> qrApplyLeft =
        ComputeShader::compile(kQrApplyLeftSource, &shaderError);
    if (!qrApplyLeft) {
        logging::debug("OpenGL compute backend unavailable: qrApplyLeft shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> qrAccumulateQ =
        ComputeShader::compile(kQrAccumulateQSource, &shaderError);
    if (!qrAccumulateQ) {
        logging::debug("OpenGL compute backend unavailable: qrAccumulateQ shader: {}",
                       shaderError);
        return nullptr;
    }
    const std::string jacobiEigenColumnMixSource =
        std::string(kJacobiEigenColumnMixHeader) + std::string(kJacobiRotationFunction) +
        std::string(kJacobiEigenColumnMixMain);
    std::optional<ComputeShader> jacobiEigenColumnMix =
        ComputeShader::compile(jacobiEigenColumnMixSource, &shaderError);
    if (!jacobiEigenColumnMix) {
        logging::debug(
            "OpenGL compute backend unavailable: jacobiEigenColumnMix shader: {}",
            shaderError);
        return nullptr;
    }
    const std::string jacobiEigenRowMixSource = std::string(kJacobiEigenRowMixHeader) +
                                                std::string(kJacobiRotationFunction) +
                                                std::string(kJacobiEigenRowMixMain);
    std::optional<ComputeShader> jacobiEigenRowMix =
        ComputeShader::compile(jacobiEigenRowMixSource, &shaderError);
    if (!jacobiEigenRowMix) {
        logging::debug("OpenGL compute backend unavailable: jacobiEigenRowMix shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> jacobiSvdRound =
        ComputeShader::compile(kJacobiSvdRoundSource, &shaderError);
    if (!jacobiSvdRound) {
        logging::debug("OpenGL compute backend unavailable: jacobiSvdRound shader: {}",
                       shaderError);
        return nullptr;
    }
    const std::string batchErfSource = std::string(kBatchScalarHeader) +
                                       std::string(kBatchEvaluationFunctions) +
                                       std::string(kBatchErfMain);
    std::optional<ComputeShader> batchErf =
        ComputeShader::compile(batchErfSource, &shaderError);
    if (!batchErf) {
        logging::debug("OpenGL compute backend unavailable: batchErf shader: {}",
                       shaderError);
        return nullptr;
    }
    const std::string batchErfcSource = std::string(kBatchScalarHeader) +
                                        std::string(kBatchEvaluationFunctions) +
                                        std::string(kBatchErfcMain);
    std::optional<ComputeShader> batchErfc =
        ComputeShader::compile(batchErfcSource, &shaderError);
    if (!batchErfc) {
        logging::debug("OpenGL compute backend unavailable: batchErfc shader: {}",
                       shaderError);
        return nullptr;
    }
    const std::string batchGammaSource = std::string(kBatchScalarHeader) +
                                         std::string(kBatchEvaluationFunctions) +
                                         std::string(kBatchGammaMain);
    std::optional<ComputeShader> batchGamma =
        ComputeShader::compile(batchGammaSource, &shaderError);
    if (!batchGamma) {
        logging::debug("OpenGL compute backend unavailable: batchGamma shader: {}",
                       shaderError);
        return nullptr;
    }
    const std::string batchLogGammaSource = std::string(kBatchScalarHeader) +
                                            std::string(kBatchEvaluationFunctions) +
                                            std::string(kBatchLogGammaMain);
    std::optional<ComputeShader> batchLogGamma =
        ComputeShader::compile(batchLogGammaSource, &shaderError);
    if (!batchLogGamma) {
        logging::debug("OpenGL compute backend unavailable: batchLogGamma shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> batchLegendreP =
        ComputeShader::compile(kBatchLegendrePSource, &shaderError);
    if (!batchLegendreP) {
        logging::debug("OpenGL compute backend unavailable: batchLegendreP shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> batchPolynomialEval =
        ComputeShader::compile(kBatchPolynomialEvalSource, &shaderError);
    if (!batchPolynomialEval) {
        logging::debug(
            "OpenGL compute backend unavailable: batchPolynomialEval shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> batchCubicSplineEval =
        ComputeShader::compile(kBatchCubicSplineEvalSource, &shaderError);
    if (!batchCubicSplineEval) {
        logging::debug(
            "OpenGL compute backend unavailable: batchCubicSplineEval shader: {}",
            shaderError);
        return nullptr;
    }
    const std::string batchUniformRealSource = std::string(kBatchRandomHeader) +
                                               std::string(kBatchRandomFunctions) +
                                               std::string(kBatchUniformRealMain);
    std::optional<ComputeShader> batchUniformReal =
        ComputeShader::compile(batchUniformRealSource, &shaderError);
    if (!batchUniformReal) {
        logging::debug("OpenGL compute backend unavailable: batchUniformReal shader: {}",
                       shaderError);
        return nullptr;
    }
    const std::string batchNormalSource = std::string(kBatchRandomHeader) +
                                          std::string(kBatchRandomFunctions) +
                                          std::string(kBatchNormalMain);
    std::optional<ComputeShader> batchNormal =
        ComputeShader::compile(batchNormalSource, &shaderError);
    if (!batchNormal) {
        logging::debug("OpenGL compute backend unavailable: batchNormal shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> bitonicCompareExchange =
        ComputeShader::compile(kBitonicCompareExchangeSource, &shaderError);
    if (!bitonicCompareExchange) {
        logging::debug(
            "OpenGL compute backend unavailable: bitonicCompareExchange shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> multigridRestrict3D =
        ComputeShader::compile(kMultigridRestrict3DSource, &shaderError);
    if (!multigridRestrict3D) {
        logging::debug(
            "OpenGL compute backend unavailable: multigridRestrict3D shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> multigridProlongateAndAdd3D =
        ComputeShader::compile(kMultigridProlongateAndAdd3DSource, &shaderError);
    if (!multigridProlongateAndAdd3D) {
        logging::debug(
            "OpenGL compute backend unavailable: multigridProlongateAndAdd3D shader: {}",
            shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> minIndex =
        ComputeShader::compile(kMinIndexSource, &shaderError);
    if (!minIndex) {
        logging::debug("OpenGL compute backend unavailable: minIndex shader: {}",
                       shaderError);
        return nullptr;
    }

    return std::unique_ptr<ComputeBackend>{new OpenGLBackend(
        std::move(*window), std::move(*saxpy), std::move(*sum), std::move(*linearCombine),
        std::move(*gravitationalNBody), std::move(*electricFieldNBody),
        std::move(*magneticFieldNBody), std::move(*sphDensityPressure),
        std::move(*sphPressureAcceleration), std::move(*heatEquation3DStep),
        std::move(*acoustic3DVelocity), std::move(*acoustic3DPressure),
        std::move(*maxwell3DB), std::move(*maxwell3DE), std::move(*eulerianFluid3DSweep),
        std::move(*fftBitReversalPermute), std::move(*fftButterflyStage),
        std::move(*fftScale), std::move(*matVec), std::move(*matMul),
        std::move(*argMaxAbsColumn), std::move(*swapRows), std::move(*luEliminationStep),
        std::move(*choleskyDiagonal), std::move(*choleskyColumn),
        std::move(*qrColumnNormSquared), std::move(*qrApplyLeft),
        std::move(*qrAccumulateQ), std::move(*jacobiEigenColumnMix),
        std::move(*jacobiEigenRowMix), std::move(*jacobiSvdRound), std::move(*batchErf),
        std::move(*batchErfc), std::move(*batchGamma), std::move(*batchLogGamma),
        std::move(*batchLegendreP), std::move(*batchPolynomialEval),
        std::move(*batchCubicSplineEval), std::move(*batchUniformReal),
        std::move(*batchNormal), std::move(*bitonicCompareExchange),
        std::move(*multigridRestrict3D), std::move(*multigridProlongateAndAdd3D),
        std::move(*minIndex))};
}

OpenGLBackend::OpenGLBackend(
    Window window, ComputeShader saxpy, ComputeShader sum, ComputeShader linearCombine,
    ComputeShader gravitationalNBody, ComputeShader electricFieldNBody,
    ComputeShader magneticFieldNBody, ComputeShader sphDensityPressure,
    ComputeShader sphPressureAcceleration, ComputeShader heatEquation3DStep,
    ComputeShader acoustic3DVelocity, ComputeShader acoustic3DPressure,
    ComputeShader maxwell3DB, ComputeShader maxwell3DE,
    ComputeShader eulerianFluid3DSweep, ComputeShader fftBitReversalPermute,
    ComputeShader fftButterflyStage, ComputeShader fftScale, ComputeShader matVec,
    ComputeShader matMul, ComputeShader argMaxAbsColumn, ComputeShader swapRows,
    ComputeShader luEliminationStep, ComputeShader choleskyDiagonal,
    ComputeShader choleskyColumn, ComputeShader qrColumnNormSquared,
    ComputeShader qrApplyLeft, ComputeShader qrAccumulateQ,
    ComputeShader jacobiEigenColumnMix, ComputeShader jacobiEigenRowMix,
    ComputeShader jacobiSvdRound, ComputeShader batchErf, ComputeShader batchErfc,
    ComputeShader batchGamma, ComputeShader batchLogGamma, ComputeShader batchLegendreP,
    ComputeShader batchPolynomialEval, ComputeShader batchCubicSplineEval,
    ComputeShader batchUniformReal, ComputeShader batchNormal,
    ComputeShader bitonicCompareExchange, ComputeShader multigridRestrict3D,
    ComputeShader multigridProlongateAndAdd3D, ComputeShader minIndex) noexcept
    : m_window(std::move(window)),
      m_saxpy(std::move(saxpy)),
      m_sum(std::move(sum)),
      m_linearCombine(std::move(linearCombine)),
      m_gravitationalNBody(std::move(gravitationalNBody)),
      m_electricFieldNBody(std::move(electricFieldNBody)),
      m_magneticFieldNBody(std::move(magneticFieldNBody)),
      m_sphDensityPressure(std::move(sphDensityPressure)),
      m_sphPressureAcceleration(std::move(sphPressureAcceleration)),
      m_heatEquation3DStep(std::move(heatEquation3DStep)),
      m_acoustic3DVelocity(std::move(acoustic3DVelocity)),
      m_acoustic3DPressure(std::move(acoustic3DPressure)),
      m_maxwell3DB(std::move(maxwell3DB)),
      m_maxwell3DE(std::move(maxwell3DE)),
      m_eulerianFluid3DSweep(std::move(eulerianFluid3DSweep)),
      m_fftBitReversalPermute(std::move(fftBitReversalPermute)),
      m_fftButterflyStage(std::move(fftButterflyStage)),
      m_fftScale(std::move(fftScale)),
      m_matVec(std::move(matVec)),
      m_matMul(std::move(matMul)),
      m_argMaxAbsColumn(std::move(argMaxAbsColumn)),
      m_swapRows(std::move(swapRows)),
      m_luEliminationStep(std::move(luEliminationStep)),
      m_choleskyDiagonal(std::move(choleskyDiagonal)),
      m_choleskyColumn(std::move(choleskyColumn)),
      m_qrColumnNormSquared(std::move(qrColumnNormSquared)),
      m_qrApplyLeft(std::move(qrApplyLeft)),
      m_qrAccumulateQ(std::move(qrAccumulateQ)),
      m_jacobiEigenColumnMix(std::move(jacobiEigenColumnMix)),
      m_jacobiEigenRowMix(std::move(jacobiEigenRowMix)),
      m_jacobiSvdRound(std::move(jacobiSvdRound)),
      m_batchErf(std::move(batchErf)),
      m_batchErfc(std::move(batchErfc)),
      m_batchGamma(std::move(batchGamma)),
      m_batchLogGamma(std::move(batchLogGamma)),
      m_batchLegendreP(std::move(batchLegendreP)),
      m_batchPolynomialEval(std::move(batchPolynomialEval)),
      m_batchCubicSplineEval(std::move(batchCubicSplineEval)),
      m_batchUniformReal(std::move(batchUniformReal)),
      m_batchNormal(std::move(batchNormal)),
      m_bitonicCompareExchange(std::move(bitonicCompareExchange)),
      m_multigridRestrict3D(std::move(multigridRestrict3D)),
      m_multigridProlongateAndAdd3D(std::move(multigridProlongateAndAdd3D)),
      m_minIndex(std::move(minIndex)) {}

void OpenGLBackend::saxpy(std::span<const float> x, std::span<float> y, float a) const {
    // Matches CpuBackend's precondition check: without this, a caller bug
    // here would silently truncate against y's length (the shader guards on
    // y.length(), not x.length()) instead of failing loudly like every other
    // backend does in a debug build.
    assert(x.size() == y.size() && "saxpy needs matching spans");
    m_window.makeContextCurrent();

    const ShaderBuffer xBuffer{x};
    const ShaderBuffer yBuffer{y};

    m_saxpy.use();
    m_saxpy.bindBuffer(0, xBuffer.handle());
    m_saxpy.bindBuffer(1, yBuffer.handle());
    m_saxpy.setUniform("a", a);
    m_saxpy.dispatch(groupCountFor(x.size()));

    yBuffer.read(y);
}

float OpenGLBackend::sum(std::span<const float> x) const {
    m_window.makeContextCurrent();

    const unsigned groups = groupCountFor(x.size());
    const ShaderBuffer inBuffer{x};
    const ShaderBuffer outBuffer{static_cast<std::size_t>(groups)};

    m_sum.use();
    m_sum.bindBuffer(0, inBuffer.handle());
    m_sum.bindBuffer(1, outBuffer.handle());
    m_sum.dispatch(groups);

    std::vector<float> partials(groups);
    outBuffer.read(partials);

    // One partial per work group, so finishing on the CPU does not earn a
    // third shader.
    float total = 0.0f;
    for (const float partial : partials) {
        total += partial;
    }
    return total;
}

void OpenGLBackend::linearCombine(std::span<const std::span<const float>> terms,
                                  std::span<const float> coefficients,
                                  std::span<float> y) const {
    assert(terms.size() == coefficients.size() && "one coefficient per term");
    assert(terms.size() >= 1 && terms.size() <= 4 &&
           "linearCombine takes one to four terms");
    m_window.makeContextCurrent();

    const ShaderBuffer yBuffer{y};
    std::array<std::optional<ShaderBuffer>, 4> termBuffers;
    std::array<float, 4> coefficientSlots{0.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t k = 0; k < terms.size(); ++k) {
        assert(terms[k].size() == y.size() && "every term must match y's length");
        termBuffers[k].emplace(terms[k]);
        coefficientSlots[k] = coefficients[k];
    }

    m_linearCombine.use();
    for (std::size_t k = 0; k < 4; ++k) {
        // Slots past terms.size() bind y's own buffer: never read there
        // (termCount gates it in the shader), just a legal binding.
        const unsigned handle =
            termBuffers[k] ? termBuffers[k]->handle() : yBuffer.handle();
        m_linearCombine.bindBuffer(static_cast<unsigned>(k), handle);
    }
    m_linearCombine.bindBuffer(4, yBuffer.handle());
    m_linearCombine.setUniform("termCount", static_cast<int>(terms.size()));
    m_linearCombine.setUniform4("coefficients", coefficientSlots[0], coefficientSlots[1],
                                coefficientSlots[2], coefficientSlots[3]);
    m_linearCombine.dispatch(groupCountFor(y.size()));

    yBuffer.read(y);
}

void OpenGLBackend::gravitationalNBody(std::span<const float> positionsX,
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
    m_window.makeContextCurrent();

    const ShaderBuffer posXBuffer{positionsX};
    const ShaderBuffer posYBuffer{positionsY};
    const ShaderBuffer posZBuffer{positionsZ};
    const ShaderBuffer gmBuffer{gm};
    const ShaderBuffer accXBuffer{n};
    const ShaderBuffer accYBuffer{n};
    const ShaderBuffer accZBuffer{n};

    m_gravitationalNBody.use();
    m_gravitationalNBody.bindBuffer(0, posXBuffer.handle());
    m_gravitationalNBody.bindBuffer(1, posYBuffer.handle());
    m_gravitationalNBody.bindBuffer(2, posZBuffer.handle());
    m_gravitationalNBody.bindBuffer(3, gmBuffer.handle());
    m_gravitationalNBody.bindBuffer(4, accXBuffer.handle());
    m_gravitationalNBody.bindBuffer(5, accYBuffer.handle());
    m_gravitationalNBody.bindBuffer(6, accZBuffer.handle());
    m_gravitationalNBody.setUniform("softeningSquared", softeningSquared);
    m_gravitationalNBody.dispatch(groupCountFor(n));

    accXBuffer.read(accelerationsX);
    accYBuffer.read(accelerationsY);
    accZBuffer.read(accelerationsZ);
}

void OpenGLBackend::electricFieldNBody(std::span<const float> positionsX,
                                       std::span<const float> positionsY,
                                       std::span<const float> positionsZ,
                                       std::span<const float> charge,
                                       float coulombConstant, std::span<float> fieldX,
                                       std::span<float> fieldY,
                                       std::span<float> fieldZ) const {
    const std::size_t n = positionsX.size();
    m_window.makeContextCurrent();

    const ShaderBuffer posXBuffer{positionsX};
    const ShaderBuffer posYBuffer{positionsY};
    const ShaderBuffer posZBuffer{positionsZ};
    const ShaderBuffer chargeBuffer{charge};
    const ShaderBuffer fieldXBuffer{n};
    const ShaderBuffer fieldYBuffer{n};
    const ShaderBuffer fieldZBuffer{n};

    m_electricFieldNBody.use();
    m_electricFieldNBody.bindBuffer(0, posXBuffer.handle());
    m_electricFieldNBody.bindBuffer(1, posYBuffer.handle());
    m_electricFieldNBody.bindBuffer(2, posZBuffer.handle());
    m_electricFieldNBody.bindBuffer(3, chargeBuffer.handle());
    m_electricFieldNBody.bindBuffer(4, fieldXBuffer.handle());
    m_electricFieldNBody.bindBuffer(5, fieldYBuffer.handle());
    m_electricFieldNBody.bindBuffer(6, fieldZBuffer.handle());
    m_electricFieldNBody.setUniform("coulombConstant", coulombConstant);
    m_electricFieldNBody.dispatch(groupCountFor(n));

    fieldXBuffer.read(fieldX);
    fieldYBuffer.read(fieldY);
    fieldZBuffer.read(fieldZ);
}

void OpenGLBackend::magneticFieldNBody(
    std::span<const float> positionsX, std::span<const float> positionsY,
    std::span<const float> positionsZ, std::span<const float> velocitiesX,
    std::span<const float> velocitiesY, std::span<const float> velocitiesZ,
    std::span<const float> charge, float permeabilityOver4Pi, std::span<float> fieldX,
    std::span<float> fieldY, std::span<float> fieldZ) const {
    const std::size_t n = positionsX.size();
    m_window.makeContextCurrent();

    const ShaderBuffer posXBuffer{positionsX};
    const ShaderBuffer posYBuffer{positionsY};
    const ShaderBuffer posZBuffer{positionsZ};
    const ShaderBuffer velXBuffer{velocitiesX};
    const ShaderBuffer velYBuffer{velocitiesY};
    const ShaderBuffer velZBuffer{velocitiesZ};
    const ShaderBuffer chargeBuffer{charge};
    const ShaderBuffer fieldXBuffer{n};
    const ShaderBuffer fieldYBuffer{n};
    const ShaderBuffer fieldZBuffer{n};

    m_magneticFieldNBody.use();
    m_magneticFieldNBody.bindBuffer(0, posXBuffer.handle());
    m_magneticFieldNBody.bindBuffer(1, posYBuffer.handle());
    m_magneticFieldNBody.bindBuffer(2, posZBuffer.handle());
    m_magneticFieldNBody.bindBuffer(3, velXBuffer.handle());
    m_magneticFieldNBody.bindBuffer(4, velYBuffer.handle());
    m_magneticFieldNBody.bindBuffer(5, velZBuffer.handle());
    m_magneticFieldNBody.bindBuffer(6, chargeBuffer.handle());
    m_magneticFieldNBody.bindBuffer(7, fieldXBuffer.handle());
    m_magneticFieldNBody.bindBuffer(8, fieldYBuffer.handle());
    m_magneticFieldNBody.bindBuffer(9, fieldZBuffer.handle());
    m_magneticFieldNBody.setUniform("permeabilityOver4Pi", permeabilityOver4Pi);
    m_magneticFieldNBody.dispatch(groupCountFor(n));

    fieldXBuffer.read(fieldX);
    fieldYBuffer.read(fieldY);
    fieldZBuffer.read(fieldZ);
}

void OpenGLBackend::sphDensityPressure(std::span<const float> positionsX,
                                       std::span<const float> positionsY,
                                       std::span<const float> positionsZ,
                                       std::span<const float> mass, float smoothingLength,
                                       float equationOfStateK, float polytropicIndex,
                                       std::span<float> density,
                                       std::span<float> pressure) const {
    const std::size_t n = positionsX.size();
    m_window.makeContextCurrent();

    const ShaderBuffer posXBuffer{positionsX};
    const ShaderBuffer posYBuffer{positionsY};
    const ShaderBuffer posZBuffer{positionsZ};
    const ShaderBuffer massBuffer{mass};
    const ShaderBuffer densityBuffer{n};
    const ShaderBuffer pressureBuffer{n};

    m_sphDensityPressure.use();
    m_sphDensityPressure.bindBuffer(0, posXBuffer.handle());
    m_sphDensityPressure.bindBuffer(1, posYBuffer.handle());
    m_sphDensityPressure.bindBuffer(2, posZBuffer.handle());
    m_sphDensityPressure.bindBuffer(3, massBuffer.handle());
    m_sphDensityPressure.bindBuffer(4, densityBuffer.handle());
    m_sphDensityPressure.bindBuffer(5, pressureBuffer.handle());
    m_sphDensityPressure.setUniform("smoothingLength", smoothingLength);
    m_sphDensityPressure.setUniform("equationOfStateK", equationOfStateK);
    m_sphDensityPressure.setUniform("polytropicIndex", polytropicIndex);
    m_sphDensityPressure.dispatch(groupCountFor(n));

    densityBuffer.read(density);
    pressureBuffer.read(pressure);
}

void OpenGLBackend::sphPressureAcceleration(
    std::span<const float> positionsX, std::span<const float> positionsY,
    std::span<const float> positionsZ, std::span<const float> mass,
    std::span<const float> density, std::span<const float> pressure,
    float smoothingLength, std::span<float> accelerationsX,
    std::span<float> accelerationsY, std::span<float> accelerationsZ) const {
    const std::size_t n = positionsX.size();
    m_window.makeContextCurrent();

    const ShaderBuffer posXBuffer{positionsX};
    const ShaderBuffer posYBuffer{positionsY};
    const ShaderBuffer posZBuffer{positionsZ};
    const ShaderBuffer massBuffer{mass};
    const ShaderBuffer densityBuffer{density};
    const ShaderBuffer pressureBuffer{pressure};
    const ShaderBuffer accXBuffer{n};
    const ShaderBuffer accYBuffer{n};
    const ShaderBuffer accZBuffer{n};

    m_sphPressureAcceleration.use();
    m_sphPressureAcceleration.bindBuffer(0, posXBuffer.handle());
    m_sphPressureAcceleration.bindBuffer(1, posYBuffer.handle());
    m_sphPressureAcceleration.bindBuffer(2, posZBuffer.handle());
    m_sphPressureAcceleration.bindBuffer(3, massBuffer.handle());
    m_sphPressureAcceleration.bindBuffer(4, densityBuffer.handle());
    m_sphPressureAcceleration.bindBuffer(5, pressureBuffer.handle());
    m_sphPressureAcceleration.bindBuffer(6, accXBuffer.handle());
    m_sphPressureAcceleration.bindBuffer(7, accYBuffer.handle());
    m_sphPressureAcceleration.bindBuffer(8, accZBuffer.handle());
    m_sphPressureAcceleration.setUniform("smoothingLength", smoothingLength);
    m_sphPressureAcceleration.dispatch(groupCountFor(n));

    accXBuffer.read(accelerationsX);
    accYBuffer.read(accelerationsY);
    accZBuffer.read(accelerationsZ);
}

void OpenGLBackend::heatEquation3DStep(std::span<const float> temperature, std::size_t nx,
                                       std::size_t ny, std::size_t nz, float factor,
                                       std::span<float> next) const {
    const std::size_t total = nx * ny * nz;
    m_window.makeContextCurrent();

    const ShaderBuffer tempBuffer{temperature};
    const ShaderBuffer nextBuffer{total};

    m_heatEquation3DStep.use();
    m_heatEquation3DStep.bindBuffer(0, tempBuffer.handle());
    m_heatEquation3DStep.bindBuffer(1, nextBuffer.handle());
    m_heatEquation3DStep.setUniform("nx", static_cast<int>(nx));
    m_heatEquation3DStep.setUniform("ny", static_cast<int>(ny));
    m_heatEquation3DStep.setUniform("nz", static_cast<int>(nz));
    m_heatEquation3DStep.setUniform("factor", factor);
    m_heatEquation3DStep.dispatch(groupCountFor(total));

    nextBuffer.read(next);
}

void OpenGLBackend::acoustic3DStep(
    std::span<const float> pressure, std::span<const float> velocityX,
    std::span<const float> velocityY, std::span<const float> velocityZ, std::size_t nx,
    std::size_t ny, std::size_t nz, float velocityFactor, float pressureFactor,
    std::span<float> nextPressure, std::span<float> nextVelocityX,
    std::span<float> nextVelocityY, std::span<float> nextVelocityZ) const {
    const std::size_t total = nx * ny * nz;
    m_window.makeContextCurrent();

    const ShaderBuffer pressureBuffer{pressure};
    const ShaderBuffer velXInBuffer{velocityX};
    const ShaderBuffer velYInBuffer{velocityY};
    const ShaderBuffer velZInBuffer{velocityZ};
    const ShaderBuffer velXOutBuffer{total};
    const ShaderBuffer velYOutBuffer{total};
    const ShaderBuffer velZOutBuffer{total};

    m_acoustic3DVelocity.use();
    m_acoustic3DVelocity.bindBuffer(0, pressureBuffer.handle());
    m_acoustic3DVelocity.bindBuffer(1, velXInBuffer.handle());
    m_acoustic3DVelocity.bindBuffer(2, velYInBuffer.handle());
    m_acoustic3DVelocity.bindBuffer(3, velZInBuffer.handle());
    m_acoustic3DVelocity.bindBuffer(4, velXOutBuffer.handle());
    m_acoustic3DVelocity.bindBuffer(5, velYOutBuffer.handle());
    m_acoustic3DVelocity.bindBuffer(6, velZOutBuffer.handle());
    m_acoustic3DVelocity.setUniform("nx", static_cast<int>(nx));
    m_acoustic3DVelocity.setUniform("ny", static_cast<int>(ny));
    m_acoustic3DVelocity.setUniform("nz", static_cast<int>(nz));
    m_acoustic3DVelocity.setUniform("velocityFactor", velocityFactor);
    m_acoustic3DVelocity.dispatch(groupCountFor(total));

    const ShaderBuffer pressureOutBuffer{total};
    m_acoustic3DPressure.use();
    m_acoustic3DPressure.bindBuffer(0, pressureBuffer.handle());
    m_acoustic3DPressure.bindBuffer(1, velXOutBuffer.handle());
    m_acoustic3DPressure.bindBuffer(2, velYOutBuffer.handle());
    m_acoustic3DPressure.bindBuffer(3, velZOutBuffer.handle());
    m_acoustic3DPressure.bindBuffer(4, pressureOutBuffer.handle());
    m_acoustic3DPressure.setUniform("nx", static_cast<int>(nx));
    m_acoustic3DPressure.setUniform("ny", static_cast<int>(ny));
    m_acoustic3DPressure.setUniform("nz", static_cast<int>(nz));
    m_acoustic3DPressure.setUniform("pressureFactor", pressureFactor);
    m_acoustic3DPressure.dispatch(groupCountFor(total));

    velXOutBuffer.read(nextVelocityX);
    velYOutBuffer.read(nextVelocityY);
    velZOutBuffer.read(nextVelocityZ);
    pressureOutBuffer.read(nextPressure);
}

void OpenGLBackend::maxwell3DStep(std::span<const float> ex, std::span<const float> ey,
                                  std::span<const float> ez, std::span<const float> bx,
                                  std::span<const float> by, std::span<const float> bz,
                                  std::size_t nx, std::size_t ny, std::size_t nz,
                                  float bFactor, float eFactor, std::span<float> nextEx,
                                  std::span<float> nextEy, std::span<float> nextEz,
                                  std::span<float> nextBx, std::span<float> nextBy,
                                  std::span<float> nextBz) const {
    const std::size_t total = nx * ny * nz;
    m_window.makeContextCurrent();

    const ShaderBuffer exBuffer{ex};
    const ShaderBuffer eyBuffer{ey};
    const ShaderBuffer ezBuffer{ez};
    const ShaderBuffer bxInBuffer{bx};
    const ShaderBuffer byInBuffer{by};
    const ShaderBuffer bzInBuffer{bz};
    const ShaderBuffer bxOutBuffer{total};
    const ShaderBuffer byOutBuffer{total};
    const ShaderBuffer bzOutBuffer{total};

    m_maxwell3DB.use();
    m_maxwell3DB.bindBuffer(0, exBuffer.handle());
    m_maxwell3DB.bindBuffer(1, eyBuffer.handle());
    m_maxwell3DB.bindBuffer(2, ezBuffer.handle());
    m_maxwell3DB.bindBuffer(3, bxInBuffer.handle());
    m_maxwell3DB.bindBuffer(4, byInBuffer.handle());
    m_maxwell3DB.bindBuffer(5, bzInBuffer.handle());
    m_maxwell3DB.bindBuffer(6, bxOutBuffer.handle());
    m_maxwell3DB.bindBuffer(7, byOutBuffer.handle());
    m_maxwell3DB.bindBuffer(8, bzOutBuffer.handle());
    m_maxwell3DB.setUniform("nx", static_cast<int>(nx));
    m_maxwell3DB.setUniform("ny", static_cast<int>(ny));
    m_maxwell3DB.setUniform("nz", static_cast<int>(nz));
    m_maxwell3DB.setUniform("bFactor", bFactor);
    m_maxwell3DB.dispatch(groupCountFor(total));

    const ShaderBuffer exOutBuffer{total};
    const ShaderBuffer eyOutBuffer{total};
    const ShaderBuffer ezOutBuffer{total};

    m_maxwell3DE.use();
    m_maxwell3DE.bindBuffer(0, exBuffer.handle());
    m_maxwell3DE.bindBuffer(1, eyBuffer.handle());
    m_maxwell3DE.bindBuffer(2, ezBuffer.handle());
    m_maxwell3DE.bindBuffer(3, bxOutBuffer.handle());
    m_maxwell3DE.bindBuffer(4, byOutBuffer.handle());
    m_maxwell3DE.bindBuffer(5, bzOutBuffer.handle());
    m_maxwell3DE.bindBuffer(6, exOutBuffer.handle());
    m_maxwell3DE.bindBuffer(7, eyOutBuffer.handle());
    m_maxwell3DE.bindBuffer(8, ezOutBuffer.handle());
    m_maxwell3DE.setUniform("nx", static_cast<int>(nx));
    m_maxwell3DE.setUniform("ny", static_cast<int>(ny));
    m_maxwell3DE.setUniform("nz", static_cast<int>(nz));
    m_maxwell3DE.setUniform("eFactor", eFactor);
    m_maxwell3DE.dispatch(groupCountFor(total));

    bxOutBuffer.read(nextBx);
    byOutBuffer.read(nextBy);
    bzOutBuffer.read(nextBz);
    exOutBuffer.read(nextEx);
    eyOutBuffer.read(nextEy);
    ezOutBuffer.read(nextEz);
}

void OpenGLBackend::eulerianFluid3DSweep(
    std::span<const float> density, std::span<const float> momentumNormal,
    std::span<const float> momentumTangent1, std::span<const float> momentumTangent2,
    std::span<const float> energy, std::size_t nx, std::size_t ny, std::size_t nz,
    int axis, float gamma, float dtOverSpacing, std::span<float> nextDensity,
    std::span<float> nextMomentumNormal, std::span<float> nextMomentumTangent1,
    std::span<float> nextMomentumTangent2, std::span<float> nextEnergy) const {
    const std::size_t total = nx * ny * nz;
    m_window.makeContextCurrent();

    const ShaderBuffer densityBuffer{density};
    const ShaderBuffer momentumNormalBuffer{momentumNormal};
    const ShaderBuffer momentumTangent1Buffer{momentumTangent1};
    const ShaderBuffer momentumTangent2Buffer{momentumTangent2};
    const ShaderBuffer energyBuffer{energy};
    const ShaderBuffer nextDensityBuffer{total};
    const ShaderBuffer nextMomentumNormalBuffer{total};
    const ShaderBuffer nextMomentumTangent1Buffer{total};
    const ShaderBuffer nextMomentumTangent2Buffer{total};
    const ShaderBuffer nextEnergyBuffer{total};

    m_eulerianFluid3DSweep.use();
    m_eulerianFluid3DSweep.bindBuffer(0, densityBuffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(1, momentumNormalBuffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(2, momentumTangent1Buffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(3, momentumTangent2Buffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(4, energyBuffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(5, nextDensityBuffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(6, nextMomentumNormalBuffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(7, nextMomentumTangent1Buffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(8, nextMomentumTangent2Buffer.handle());
    m_eulerianFluid3DSweep.bindBuffer(9, nextEnergyBuffer.handle());
    m_eulerianFluid3DSweep.setUniform("nx", static_cast<int>(nx));
    m_eulerianFluid3DSweep.setUniform("ny", static_cast<int>(ny));
    m_eulerianFluid3DSweep.setUniform("nz", static_cast<int>(nz));
    m_eulerianFluid3DSweep.setUniform("axis", axis);
    m_eulerianFluid3DSweep.setUniform("gamma", gamma);
    m_eulerianFluid3DSweep.setUniform("dtOverSpacing", dtOverSpacing);
    m_eulerianFluid3DSweep.dispatch(groupCountFor(total));

    nextDensityBuffer.read(nextDensity);
    nextMomentumNormalBuffer.read(nextMomentumNormal);
    nextMomentumTangent1Buffer.read(nextMomentumTangent1);
    nextMomentumTangent2Buffer.read(nextMomentumTangent2);
    nextEnergyBuffer.read(nextEnergy);
}

void OpenGLBackend::fftBatched(std::span<const float> real, std::span<const float> imag,
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

    m_window.makeContextCurrent();

    const ShaderBuffer realA{real};
    const ShaderBuffer imagA{imag};
    const ShaderBuffer realB{total};
    const ShaderBuffer imagB{total};

    m_fftBitReversalPermute.use();
    m_fftBitReversalPermute.bindBuffer(0, realA.handle());
    m_fftBitReversalPermute.bindBuffer(1, imagA.handle());
    m_fftBitReversalPermute.bindBuffer(2, realB.handle());
    m_fftBitReversalPermute.bindBuffer(3, imagB.handle());
    m_fftBitReversalPermute.setUniform("length", static_cast<int>(length));
    m_fftBitReversalPermute.setUniform("bitCount", static_cast<int>(bitCount));
    m_fftBitReversalPermute.setUniform("total", static_cast<int>(total));
    m_fftBitReversalPermute.dispatch(groupCountFor(total));

    // Bit-reversal wrote A -> B; each stage below ping-pongs between
    // whichever buffer pair holds the current data and the other, spare
    // pair, the same double-buffering MetalBackend::fftBatched uses.
    const ShaderBuffer* currentReal = &realB;
    const ShaderBuffer* currentImag = &imagB;
    const ShaderBuffer* spareReal = &realA;
    const ShaderBuffer* spareImag = &imagA;

    for (std::size_t len = 2; len <= length; len <<= 1) {
        const float angle =
            (inverse ? 1.0f : -1.0f) * 6.283185307179586f / static_cast<float>(len);

        m_fftButterflyStage.use();
        m_fftButterflyStage.bindBuffer(0, currentReal->handle());
        m_fftButterflyStage.bindBuffer(1, currentImag->handle());
        m_fftButterflyStage.bindBuffer(2, spareReal->handle());
        m_fftButterflyStage.bindBuffer(3, spareImag->handle());
        m_fftButterflyStage.setUniform("length", static_cast<int>(length));
        m_fftButterflyStage.setUniform("len", static_cast<int>(len));
        m_fftButterflyStage.setUniform("angle", angle);
        m_fftButterflyStage.setUniform("pairCount", static_cast<int>(total / 2));
        m_fftButterflyStage.dispatch(groupCountFor(total / 2));

        std::swap(currentReal, spareReal);
        std::swap(currentImag, spareImag);
    }

    if (inverse) {
        const float scale = 1.0f / static_cast<float>(length);
        m_fftScale.use();
        m_fftScale.bindBuffer(0, currentReal->handle());
        m_fftScale.bindBuffer(1, currentImag->handle());
        m_fftScale.bindBuffer(2, spareReal->handle());
        m_fftScale.bindBuffer(3, spareImag->handle());
        m_fftScale.setUniform("scale", scale);
        m_fftScale.setUniform("total", static_cast<int>(total));
        m_fftScale.dispatch(groupCountFor(total));
        std::swap(currentReal, spareReal);
        std::swap(currentImag, spareImag);
    }

    currentReal->read(nextReal);
    currentImag->read(nextImag);
}

void OpenGLBackend::matVec(std::span<const float> matrix, std::size_t rows,
                           std::size_t cols, std::span<const float> vector,
                           std::span<float> result) const {
    if (rows == 0) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer matrixBuffer{matrix};
    const ShaderBuffer vectorBuffer{vector};
    const ShaderBuffer resultBuffer{rows};

    m_matVec.use();
    m_matVec.bindBuffer(0, matrixBuffer.handle());
    m_matVec.bindBuffer(1, vectorBuffer.handle());
    m_matVec.bindBuffer(2, resultBuffer.handle());
    m_matVec.setUniform("rows", static_cast<int>(rows));
    m_matVec.setUniform("cols", static_cast<int>(cols));
    m_matVec.dispatch(groupCountFor(rows));

    resultBuffer.read(result);
}

void OpenGLBackend::matMul(std::span<const float> a, std::size_t aRows, std::size_t aCols,
                           std::span<const float> b, std::size_t bCols,
                           std::span<float> result) const {
    const std::size_t total = aRows * bCols;
    if (total == 0) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer aBuffer{a};
    const ShaderBuffer bBuffer{b};
    const ShaderBuffer resultBuffer{total};

    m_matMul.use();
    m_matMul.bindBuffer(0, aBuffer.handle());
    m_matMul.bindBuffer(1, bBuffer.handle());
    m_matMul.bindBuffer(2, resultBuffer.handle());
    m_matMul.setUniform("aRows", static_cast<int>(aRows));
    m_matMul.setUniform("aCols", static_cast<int>(aCols));
    m_matMul.setUniform("bCols", static_cast<int>(bCols));
    m_matMul.dispatch(groupCountFor(total));

    resultBuffer.read(result);
}

bool OpenGLBackend::luDecomposeGpu(std::span<const float> matrix, std::size_t n,
                                   std::span<float> lu,
                                   std::span<std::uint32_t> pivot) const {
    if (n == 0) {
        return true;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer bufferA{matrix};
    const ShaderBuffer bufferB{n * n};
    const ShaderBuffer* current = &bufferA;
    const ShaderBuffer* spare = &bufferB;

    std::vector<std::uint32_t> pivotHost(n);
    for (std::size_t i = 0; i < n; ++i) {
        pivotHost[i] = static_cast<std::uint32_t>(i);
    }

    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t activeRows = n - k;
        const unsigned groups = groupCountFor(activeRows);
        const ShaderBuffer partialValues{static_cast<std::size_t>(groups)};
        const ShaderBuffer partialIndices{static_cast<std::size_t>(groups)};

        m_argMaxAbsColumn.use();
        m_argMaxAbsColumn.bindBuffer(0, current->handle());
        m_argMaxAbsColumn.bindBuffer(1, partialValues.handle());
        m_argMaxAbsColumn.bindBuffer(2, partialIndices.handle());
        m_argMaxAbsColumn.setUniform("n", static_cast<int>(n));
        m_argMaxAbsColumn.setUniform("column", static_cast<int>(k));
        m_argMaxAbsColumn.setUniform("startRow", static_cast<int>(k));
        m_argMaxAbsColumn.setUniform("activeRows", static_cast<int>(activeRows));
        m_argMaxAbsColumn.dispatch(groups);

        std::vector<float> values(groups);
        std::vector<float> indices(groups);
        partialValues.read(values);
        partialIndices.read(indices);

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

        m_swapRows.use();
        m_swapRows.bindBuffer(0, current->handle());
        m_swapRows.bindBuffer(1, spare->handle());
        m_swapRows.setUniform("n", static_cast<int>(n));
        m_swapRows.setUniform("rowA", static_cast<int>(k));
        m_swapRows.setUniform("rowB", static_cast<int>(pivotRow));
        m_swapRows.dispatch(groupCountFor(n * n));

        std::swap(current, spare);

        m_luEliminationStep.use();
        m_luEliminationStep.bindBuffer(0, current->handle());
        m_luEliminationStep.bindBuffer(1, spare->handle());
        m_luEliminationStep.setUniform("n", static_cast<int>(n));
        m_luEliminationStep.setUniform("k", static_cast<int>(k));
        m_luEliminationStep.dispatch(groupCountFor(n * n));

        std::swap(current, spare);

        if (pivotRow != k) {
            std::swap(pivotHost[k], pivotHost[pivotRow]);
        }
    }

    current->read(lu);
    for (std::size_t i = 0; i < n; ++i) {
        pivot[i] = pivotHost[i];
    }
    return true;
}

bool OpenGLBackend::choleskyDecomposeGpu(std::span<const float> a, std::size_t n,
                                         std::span<float> l) const {
    if (n == 0) {
        return true;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer aBuffer{a};
    const std::vector<float> zeroInit(n * n, 0.0f);
    const ShaderBuffer lBuffer{std::span<const float>(zeroInit)};

    for (std::size_t j = 0; j < n; ++j) {
        m_choleskyDiagonal.use();
        m_choleskyDiagonal.bindBuffer(0, aBuffer.handle());
        m_choleskyDiagonal.bindBuffer(1, lBuffer.handle());
        m_choleskyDiagonal.setUniform("n", static_cast<int>(n));
        m_choleskyDiagonal.setUniform("j", static_cast<int>(j));
        m_choleskyDiagonal.dispatch(1);

        float diagonalValue = 0.0f;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, lBuffer.handle());
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                           static_cast<GLintptr>((j * n + j) * sizeof(float)),
                           sizeof(float), &diagonalValue);
        if (diagonalValue < 0.0f) {
            return false;
        }

        const std::size_t remainingRows = n - j - 1;
        if (remainingRows > 0) {
            m_choleskyColumn.use();
            m_choleskyColumn.bindBuffer(0, aBuffer.handle());
            m_choleskyColumn.bindBuffer(1, lBuffer.handle());
            m_choleskyColumn.setUniform("n", static_cast<int>(n));
            m_choleskyColumn.setUniform("j", static_cast<int>(j));
            m_choleskyColumn.dispatch(groupCountFor(remainingRows));
        }
    }

    lBuffer.read(l);
    return true;
}

void OpenGLBackend::qrDecomposeGpu(std::span<const float> matrix, std::size_t rows,
                                   std::size_t cols, std::span<float> q,
                                   std::span<float> r) const {
    m_window.makeContextCurrent();

    const ShaderBuffer rBufferA{matrix};
    const ShaderBuffer rBufferB{rows * cols};
    std::vector<float> qInit(rows * rows, 0.0f);
    for (std::size_t i = 0; i < rows; ++i) {
        qInit[i * rows + i] = 1.0f;
    }
    const ShaderBuffer qBufferA{std::span<const float>(qInit)};
    const ShaderBuffer qBufferB{rows * rows};

    const ShaderBuffer* currentR = &rBufferA;
    const ShaderBuffer* spareR = &rBufferB;
    const ShaderBuffer* currentQ = &qBufferA;
    const ShaderBuffer* spareQ = &qBufferB;

    const std::size_t steps = std::min(rows, cols);
    for (std::size_t k = 0; k < steps; ++k) {
        const ShaderBuffer normOutput{std::size_t{2}};

        m_qrColumnNormSquared.use();
        m_qrColumnNormSquared.bindBuffer(0, currentR->handle());
        m_qrColumnNormSquared.bindBuffer(1, normOutput.handle());
        m_qrColumnNormSquared.setUniform("rows", static_cast<int>(rows));
        m_qrColumnNormSquared.setUniform("cols", static_cast<int>(cols));
        m_qrColumnNormSquared.setUniform("k", static_cast<int>(k));
        m_qrColumnNormSquared.dispatch(1);

        std::array<float, 2> outputs{};
        normOutput.read(outputs);
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

        m_qrApplyLeft.use();
        m_qrApplyLeft.bindBuffer(0, currentR->handle());
        m_qrApplyLeft.bindBuffer(1, spareR->handle());
        m_qrApplyLeft.setUniform("rows", static_cast<int>(rows));
        m_qrApplyLeft.setUniform("cols", static_cast<int>(cols));
        m_qrApplyLeft.setUniform("k", static_cast<int>(k));
        m_qrApplyLeft.setUniform("alpha", alpha);
        m_qrApplyLeft.setUniform("vNormSquared", vNormSquared);
        m_qrApplyLeft.dispatch(groupCountFor(cols));

        m_qrAccumulateQ.use();
        m_qrAccumulateQ.bindBuffer(0, currentR->handle());
        m_qrAccumulateQ.bindBuffer(1, currentQ->handle());
        m_qrAccumulateQ.bindBuffer(2, spareQ->handle());
        m_qrAccumulateQ.setUniform("rows", static_cast<int>(rows));
        m_qrAccumulateQ.setUniform("cols", static_cast<int>(cols));
        m_qrAccumulateQ.setUniform("k", static_cast<int>(k));
        m_qrAccumulateQ.setUniform("alpha", alpha);
        m_qrAccumulateQ.setUniform("vNormSquared", vNormSquared);
        m_qrAccumulateQ.dispatch(groupCountFor(rows));

        std::swap(currentR, spareR);
        std::swap(currentQ, spareQ);
    }

    currentR->read(r);
    currentQ->read(q);
}

void OpenGLBackend::jacobiEigenSymmetricGpu(std::span<const float> matrix, std::size_t n,
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

    m_window.makeContextCurrent();

    std::vector<float> symmetrized(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            symmetrized[i * n + j] = (j <= i) ? matrix[i * n + j] : matrix[j * n + i];
        }
    }
    const ShaderBuffer aBufferA{std::span<const float>(symmetrized)};
    const ShaderBuffer aBufferB{n * n};
    const ShaderBuffer bScratch{n * n};
    std::vector<float> vInit(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        vInit[i * n + i] = 1.0f;
    }
    const ShaderBuffer vBufferA{std::span<const float>(vInit)};
    const ShaderBuffer vBufferB{n * n};

    const ShaderBuffer* currentA = &aBufferA;
    const ShaderBuffer* spareA = &aBufferB;
    const ShaderBuffer* currentV = &vBufferA;
    const ShaderBuffer* spareV = &vBufferB;

    const float effectiveTolerance =
        (tolerance > 0.0f) ? tolerance : std::numeric_limits<float>::epsilon() * 100.0f;
    const std::vector<std::vector<std::uint32_t>> schedule = roundRobinSchedule(n);

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        std::vector<float> snapshot(n * n);
        currentA->read(snapshot);
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
            const ShaderBuffer pairBuffer{std::span<const std::uint32_t>(pairOf)};

            m_jacobiEigenColumnMix.use();
            m_jacobiEigenColumnMix.bindBuffer(0, currentA->handle());
            m_jacobiEigenColumnMix.bindBuffer(1, currentV->handle());
            m_jacobiEigenColumnMix.bindBuffer(2, pairBuffer.handle());
            m_jacobiEigenColumnMix.bindBuffer(3, bScratch.handle());
            m_jacobiEigenColumnMix.bindBuffer(4, spareV->handle());
            m_jacobiEigenColumnMix.setUniform("n", static_cast<int>(n));
            m_jacobiEigenColumnMix.setUniform("epsilon", effectiveTolerance);
            m_jacobiEigenColumnMix.dispatch(groupCountFor(n));

            m_jacobiEigenRowMix.use();
            m_jacobiEigenRowMix.bindBuffer(0, currentA->handle());
            m_jacobiEigenRowMix.bindBuffer(1, bScratch.handle());
            m_jacobiEigenRowMix.bindBuffer(2, pairBuffer.handle());
            m_jacobiEigenRowMix.bindBuffer(3, spareA->handle());
            m_jacobiEigenRowMix.setUniform("n", static_cast<int>(n));
            m_jacobiEigenRowMix.setUniform("epsilon", effectiveTolerance);
            m_jacobiEigenRowMix.dispatch(groupCountFor(n));

            std::swap(currentA, spareA);
            std::swap(currentV, spareV);
        }
    }

    currentA->read(resultDiagonal);
    currentV->read(resultEigenvectors);
}

void OpenGLBackend::jacobiSvdGpu(std::span<const float> matrix, std::size_t rows,
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

    m_window.makeContextCurrent();

    const ShaderBuffer aBufferA{matrix};
    const ShaderBuffer aBufferB{rows * cols};
    std::vector<float> vInit(cols * cols, 0.0f);
    for (std::size_t i = 0; i < cols; ++i) {
        vInit[i * cols + i] = 1.0f;
    }
    const ShaderBuffer vBufferA{std::span<const float>(vInit)};
    const ShaderBuffer vBufferB{cols * cols};

    const ShaderBuffer* currentA = &aBufferA;
    const ShaderBuffer* spareA = &aBufferB;
    const ShaderBuffer* currentV = &vBufferA;
    const ShaderBuffer* spareV = &vBufferB;

    const float effectiveTolerance =
        (tolerance > 0.0f) ? tolerance : std::numeric_limits<float>::epsilon() * 100.0f;
    const std::vector<std::vector<std::uint32_t>> schedule = roundRobinSchedule(cols);

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        std::vector<float> snapshot(rows * cols);
        currentA->read(snapshot);
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
            const ShaderBuffer pairBuffer{std::span<const std::uint32_t>(pairOf)};

            m_jacobiSvdRound.use();
            m_jacobiSvdRound.bindBuffer(0, currentA->handle());
            m_jacobiSvdRound.bindBuffer(1, currentV->handle());
            m_jacobiSvdRound.bindBuffer(2, pairBuffer.handle());
            m_jacobiSvdRound.bindBuffer(3, spareA->handle());
            m_jacobiSvdRound.bindBuffer(4, spareV->handle());
            m_jacobiSvdRound.setUniform("rows", static_cast<int>(rows));
            m_jacobiSvdRound.setUniform("cols", static_cast<int>(cols));
            m_jacobiSvdRound.setUniform("epsilon", effectiveTolerance);
            m_jacobiSvdRound.dispatch(groupCountFor(cols));

            std::swap(currentA, spareA);
            std::swap(currentV, spareV);
        }
    }

    currentA->read(resultA);
    currentV->read(resultV);
}

void OpenGLBackend::batchErf(std::span<const float> x, std::span<float> result) const {
    if (x.empty()) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer xBuffer{x};
    const ShaderBuffer resultBuffer{x.size()};

    m_batchErf.use();
    m_batchErf.bindBuffer(0, xBuffer.handle());
    m_batchErf.bindBuffer(1, resultBuffer.handle());
    m_batchErf.dispatch(groupCountFor(x.size()));

    resultBuffer.read(result);
}

void OpenGLBackend::batchErfc(std::span<const float> x, std::span<float> result) const {
    if (x.empty()) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer xBuffer{x};
    const ShaderBuffer resultBuffer{x.size()};

    m_batchErfc.use();
    m_batchErfc.bindBuffer(0, xBuffer.handle());
    m_batchErfc.bindBuffer(1, resultBuffer.handle());
    m_batchErfc.dispatch(groupCountFor(x.size()));

    resultBuffer.read(result);
}

void OpenGLBackend::batchGamma(std::span<const float> x, std::span<float> result) const {
    if (x.empty()) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer xBuffer{x};
    const ShaderBuffer resultBuffer{x.size()};

    m_batchGamma.use();
    m_batchGamma.bindBuffer(0, xBuffer.handle());
    m_batchGamma.bindBuffer(1, resultBuffer.handle());
    m_batchGamma.dispatch(groupCountFor(x.size()));

    resultBuffer.read(result);
}

void OpenGLBackend::batchLogGamma(std::span<const float> x,
                                  std::span<float> result) const {
    if (x.empty()) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer xBuffer{x};
    const ShaderBuffer resultBuffer{x.size()};

    m_batchLogGamma.use();
    m_batchLogGamma.bindBuffer(0, xBuffer.handle());
    m_batchLogGamma.bindBuffer(1, resultBuffer.handle());
    m_batchLogGamma.dispatch(groupCountFor(x.size()));

    resultBuffer.read(result);
}

void OpenGLBackend::batchLegendreP(unsigned n, unsigned m, std::span<const float> x,
                                   std::span<float> result) const {
    if (x.empty()) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer xBuffer{x};
    const ShaderBuffer resultBuffer{x.size()};

    m_batchLegendreP.use();
    m_batchLegendreP.bindBuffer(0, xBuffer.handle());
    m_batchLegendreP.bindBuffer(1, resultBuffer.handle());
    m_batchLegendreP.setUniform("n", static_cast<int>(n));
    m_batchLegendreP.setUniform("m", static_cast<int>(m));
    m_batchLegendreP.dispatch(groupCountFor(x.size()));

    resultBuffer.read(result);
}

void OpenGLBackend::batchPolynomialEval(std::span<const float> coefficients,
                                        std::span<const float> x,
                                        std::span<float> result) const {
    if (x.empty()) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer coefficientsBuffer{coefficients};
    const ShaderBuffer xBuffer{x};
    const ShaderBuffer resultBuffer{x.size()};

    m_batchPolynomialEval.use();
    m_batchPolynomialEval.bindBuffer(0, coefficientsBuffer.handle());
    m_batchPolynomialEval.bindBuffer(1, xBuffer.handle());
    m_batchPolynomialEval.bindBuffer(2, resultBuffer.handle());
    m_batchPolynomialEval.dispatch(groupCountFor(x.size()));

    resultBuffer.read(result);
}

void OpenGLBackend::batchCubicSplineEval(std::span<const float> knotsX,
                                         std::span<const float> knotsY,
                                         std::span<const float> secondDerivatives,
                                         std::span<const float> queryX,
                                         std::span<float> result) const {
    if (queryX.empty()) {
        return;
    }
    m_window.makeContextCurrent();

    const ShaderBuffer knotsXBuffer{knotsX};
    const ShaderBuffer knotsYBuffer{knotsY};
    const ShaderBuffer secondDerivativesBuffer{secondDerivatives};
    const ShaderBuffer queryXBuffer{queryX};
    const ShaderBuffer resultBuffer{queryX.size()};

    m_batchCubicSplineEval.use();
    m_batchCubicSplineEval.bindBuffer(0, knotsXBuffer.handle());
    m_batchCubicSplineEval.bindBuffer(1, knotsYBuffer.handle());
    m_batchCubicSplineEval.bindBuffer(2, secondDerivativesBuffer.handle());
    m_batchCubicSplineEval.bindBuffer(3, queryXBuffer.handle());
    m_batchCubicSplineEval.bindBuffer(4, resultBuffer.handle());
    m_batchCubicSplineEval.dispatch(groupCountFor(queryX.size()));

    resultBuffer.read(result);
}

namespace {

/// The shared shape batchUniformReal/batchNormal both dispatch with: no
/// input buffer at all (Philox is counter-based, needing only `seed` and
/// `offset`, packed into a 4-element uint SSBO rather than uniforms -- see
/// kBatchRandomHeader's own comment), one output buffer, one invocation
/// per element.
void dispatchBatchRandom(const ComputeShader& shader, std::uint64_t seed,
                         std::uint64_t offset, std::span<float> result) {
    if (result.empty()) {
        return;
    }
    const std::array<std::uint32_t, 4> params{
        static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
        static_cast<std::uint32_t>(offset), static_cast<std::uint32_t>(offset >> 32)};
    const ShaderBuffer paramsBuffer{std::span<const std::uint32_t>(params)};
    const ShaderBuffer resultBuffer{result.size()};

    shader.use();
    shader.bindBuffer(0, resultBuffer.handle());
    shader.bindBuffer(1, paramsBuffer.handle());
    shader.dispatch(groupCountFor(result.size()));

    resultBuffer.read(result);
}

}  // namespace

void OpenGLBackend::batchUniformReal(std::uint64_t seed, std::uint64_t offset,
                                     std::span<float> result) const {
    m_window.makeContextCurrent();
    dispatchBatchRandom(m_batchUniformReal, seed, offset, result);
}

void OpenGLBackend::batchNormal(std::uint64_t seed, std::uint64_t offset,
                                std::span<float> result) const {
    m_window.makeContextCurrent();
    dispatchBatchRandom(m_batchNormal, seed, offset, result);
}

void OpenGLBackend::sortAscending(std::span<float> values) const {
    if (values.size() < 2) {
        return;
    }
    m_window.makeContextCurrent();

    std::size_t paddedSize = 1;
    while (paddedSize < values.size()) {
        paddedSize *= 2;
    }
    std::vector<float> padded(paddedSize, std::numeric_limits<float>::infinity());
    std::copy(values.begin(), values.end(), padded.begin());

    const ShaderBuffer buffer{std::span<const float>(padded)};
    const auto n = static_cast<int>(paddedSize);

    m_bitonicCompareExchange.use();
    m_bitonicCompareExchange.bindBuffer(0, buffer.handle());
    for (int stageSize = 2; stageSize <= n; stageSize *= 2) {
        for (int stepSize = stageSize / 2; stepSize >= 1; stepSize /= 2) {
            m_bitonicCompareExchange.setUniform("n", n);
            m_bitonicCompareExchange.setUniform("stageSize", stageSize);
            m_bitonicCompareExchange.setUniform("stepSize", stepSize);
            m_bitonicCompareExchange.dispatch(groupCountFor(paddedSize));
        }
    }

    buffer.read(padded);
    std::copy(padded.begin(), padded.begin() + static_cast<std::ptrdiff_t>(values.size()),
              values.begin());
}

void OpenGLBackend::multigridRestrict3D(std::span<const float> fine, std::size_t nx,
                                        std::size_t ny, std::size_t nz,
                                        std::span<float> coarse) const {
    const std::size_t coarseTotal = (nx / 2) * (ny / 2) * (nz / 2);
    m_window.makeContextCurrent();

    const ShaderBuffer fineBuffer{fine};
    const ShaderBuffer coarseBuffer{coarseTotal};

    m_multigridRestrict3D.use();
    m_multigridRestrict3D.bindBuffer(0, fineBuffer.handle());
    m_multigridRestrict3D.bindBuffer(1, coarseBuffer.handle());
    m_multigridRestrict3D.setUniform("nx", static_cast<int>(nx));
    m_multigridRestrict3D.setUniform("ny", static_cast<int>(ny));
    m_multigridRestrict3D.setUniform("nz", static_cast<int>(nz));
    m_multigridRestrict3D.dispatch(groupCountFor(coarseTotal));

    coarseBuffer.read(coarse);
}

void OpenGLBackend::multigridProlongateAndAdd3D(std::span<const float> fine,
                                                std::span<const float> coarseCorrection,
                                                std::size_t nx, std::size_t ny,
                                                std::size_t nz,
                                                std::span<float> nextFine) const {
    const std::size_t total = nx * ny * nz;
    m_window.makeContextCurrent();

    const ShaderBuffer fineBuffer{fine};
    const ShaderBuffer coarseBuffer{coarseCorrection};
    const ShaderBuffer nextFineBuffer{total};

    m_multigridProlongateAndAdd3D.use();
    m_multigridProlongateAndAdd3D.bindBuffer(0, fineBuffer.handle());
    m_multigridProlongateAndAdd3D.bindBuffer(1, coarseBuffer.handle());
    m_multigridProlongateAndAdd3D.bindBuffer(2, nextFineBuffer.handle());
    m_multigridProlongateAndAdd3D.setUniform("nx", static_cast<int>(nx));
    m_multigridProlongateAndAdd3D.setUniform("ny", static_cast<int>(ny));
    m_multigridProlongateAndAdd3D.setUniform("nz", static_cast<int>(nz));
    m_multigridProlongateAndAdd3D.dispatch(groupCountFor(total));

    nextFineBuffer.read(nextFine);
}

std::size_t OpenGLBackend::minIndex(std::span<const float> x) const {
    if (x.empty()) {
        return x.size();
    }
    m_window.makeContextCurrent();

    const unsigned groups = groupCountFor(x.size());
    const ShaderBuffer inBuffer{x};
    const ShaderBuffer valueBuffer{static_cast<std::size_t>(groups)};
    const ShaderBuffer indexBuffer{static_cast<std::size_t>(groups)};

    m_minIndex.use();
    m_minIndex.bindBuffer(0, inBuffer.handle());
    m_minIndex.bindBuffer(1, valueBuffer.handle());
    m_minIndex.bindBuffer(2, indexBuffer.handle());
    m_minIndex.dispatch(groups);

    std::vector<float> values(groups);
    std::vector<float> indices(groups);
    valueBuffer.read(values);
    indexBuffer.read(indices);

    std::size_t best = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] < values[best]) {
            best = i;
        }
    }
    return static_cast<std::size_t>(indices[best]);
}

}  // namespace ysq
