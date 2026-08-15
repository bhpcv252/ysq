// A standalone timing tool, not a correctness test: for every GPU-dispatchable
// operation in Math and Physics, times the CPU reference against the fastest
// available GPU backend across a range of sizes and reports the crossover
// point -- the size at which the GPU path first beats the CPU path -- so each
// header's own `kGpuDispatchThreshold` constant can be set from a real
// measurement rather than a guess. See src/Compute/README.md's own section on
// how the results here were folded back into those constants.
//
// Deliberately outside tests/: this is not asserting anything is correct
// (every kernel measured here already has its own tests/unit and
// tests/integration coverage), and its output is meant to be read by a human
// and used to edit source, not to pass or fail in CI. Gated behind
// YSQ_BUILD_BENCHMARKS (default OFF), the same "off unless you ask for it"
// default YSQ_BUILD_TESTS already uses.

#include <Compute/CPU/CpuBackend.hpp>
#include <Compute/ComputeBackend.hpp>
#include <Math/SpatialPartition/KdTree.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Fluids/SPH.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <class F>
double timeSeconds(F&& f, int reps) {
    // One untimed warm-up call first: the first dispatch of a given kernel
    // pays for pipeline/shader compilation and buffer allocation that every
    // later call does not, and that one-time cost has nothing to do with the
    // steady-state crossover this tool is trying to find.
    f();
    const auto start = Clock::now();
    for (int i = 0; i < reps; ++i) {
        f();
    }
    const auto end = Clock::now();
    return std::chrono::duration<double>(end - start).count() / static_cast<double>(reps);
}

/// Runs `cpuFn`/`gpuFn` at each size in `sizes`, printing a row per size and
/// returning the smallest size at which the GPU path was faster -- or the
/// largest size tried, with a note, if the GPU path never won.
std::size_t findCrossover(const std::string& name, const std::string& sizeLabel,
                          const std::vector<std::size_t>& sizes,
                          const std::function<double(std::size_t)>& sizeMetric,
                          const std::function<void(std::size_t)>& cpuFn,
                          const std::function<void(std::size_t)>& gpuFn, int reps) {
    std::printf("\n=== %s ===\n", name.c_str());
    std::printf("%12s %14s %14s %10s %8s\n", sizeLabel.c_str(), "cpu (ms)", "gpu (ms)",
                "ratio", "winner");

    std::size_t crossover = 0;
    bool found = false;
    for (const std::size_t n : sizes) {
        const double cpuTime = timeSeconds([&] { cpuFn(n); }, reps);
        const double gpuTime = timeSeconds([&] { gpuFn(n); }, reps);
        const double ratio = cpuTime / gpuTime;
        const char* winner = (gpuTime < cpuTime) ? "gpu" : "cpu";
        std::printf("%12zu %14.4f %14.4f %10.2f %8s\n", n, cpuTime * 1000.0,
                    gpuTime * 1000.0, ratio, winner);
        if (!found && gpuTime < cpuTime) {
            crossover = static_cast<std::size_t>(sizeMetric(n));
            found = true;
        }
    }
    if (!found) {
        std::printf("(GPU never won in the sizes tried; using the largest as a floor)\n");
        crossover = static_cast<std::size_t>(sizeMetric(sizes.back())) * 2;
    }
    std::printf(
        "-> recommended threshold (on the metric %s's own detail:: constant gates): "
        "%zu\n",
        name.c_str(), crossover);
    return crossover;
}

std::vector<float> randomFloats(std::size_t n, unsigned seed) {
    std::vector<float> values(n);
    std::uint32_t state = seed * 2654435761u + 1u;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        values[i] = static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
    }
    return values;
}

}  // namespace

int main() {
    const ysq::CpuBackend cpu;
    const std::unique_ptr<ysq::ComputeBackend> gpu =
        ysq::selectComputeBackend(ysq::ComputeBackendKind::Metal);
    if (!gpu) {
        std::printf(
            "No GPU backend (Metal) available on this machine; nothing to calibrate "
            "against. Run this on a machine with a real GPU backend.\n");
        return 1;
    }
    const std::string_view gpuKindName = ysq::toString(gpu->kind());
    std::printf("GPU backend: %.*s\n", static_cast<int>(gpuKindName.size()),
                gpuKindName.data());

    // --- LinearSolve: matVec, matMul, LU, Cholesky (one shared threshold,
    // gated on rows*cols / aRows*aCols*bCols / n*n respectively) ----------

    {
        const std::vector<std::size_t> sizes{64, 128, 256, 512, 1024, 2048, 4096};
        const auto matVecCpu = [&](std::size_t n) {
            const auto m = randomFloats(n * n, 1);
            const auto v = randomFloats(n, 2);
            std::vector<float> result(n);
            cpu.matVec(m, n, n, v, result);
        };
        const auto matVecGpu = [&](std::size_t n) {
            const auto m = randomFloats(n * n, 1);
            const auto v = randomFloats(n, 2);
            std::vector<float> result(n);
            gpu->matVec(m, n, n, v, result);
        };
        findCrossover(
            "matVec (LinearSolve)", "rows=cols", sizes,
            [](std::size_t n) { return n * n; }, matVecCpu, matVecGpu, 20);
    }

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128, 256, 512};
        const auto matMulCpu = [&](std::size_t n) {
            const auto a = randomFloats(n * n, 3);
            const auto b = randomFloats(n * n, 4);
            std::vector<float> result(n * n);
            cpu.matMul(a, n, n, b, n, result);
        };
        const auto matMulGpu = [&](std::size_t n) {
            const auto a = randomFloats(n * n, 3);
            const auto b = randomFloats(n * n, 4);
            std::vector<float> result(n * n);
            gpu->matMul(a, n, n, b, n, result);
        };
        findCrossover(
            "matMul (LinearSolve)", "n (n^3 work)", sizes,
            [](std::size_t n) { return n * n * n; }, matMulCpu, matMulGpu, 10);
    }

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128, 256, 512};
        const auto luCpu = [&](std::size_t n) {
            std::vector<float> m(n * n);
            for (std::size_t i = 0; i < n; ++i) {
                m[i * n + i] = static_cast<float>(n) * 2.0f;
            }
            const auto noise = randomFloats(n * n, 5);
            for (std::size_t i = 0; i < n * n; ++i) {
                m[i] += noise[i] * 0.01f;
            }
            std::vector<float> lu(n * n);
            std::vector<std::uint32_t> pivot(n);
            (void)cpu.luDecomposeGpu(m, n, lu, pivot);
        };
        const auto luGpu = [&](std::size_t n) {
            std::vector<float> m(n * n);
            for (std::size_t i = 0; i < n; ++i) {
                m[i * n + i] = static_cast<float>(n) * 2.0f;
            }
            const auto noise = randomFloats(n * n, 5);
            for (std::size_t i = 0; i < n * n; ++i) {
                m[i] += noise[i] * 0.01f;
            }
            std::vector<float> lu(n * n);
            std::vector<std::uint32_t> pivot(n);
            (void)gpu->luDecomposeGpu(m, n, lu, pivot);
        };
        findCrossover(
            "luDecompose (LinearSolve)", "n", sizes, [](std::size_t n) { return n * n; },
            luCpu, luGpu, 5);
    }

    // --- Eigen: QR, symmetric Jacobi eigendecomposition, SVD (one shared
    // threshold, gated on n*n or m*n) --------------------------------------

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128, 256};
        const auto qrCpu = [&](std::size_t n) {
            const auto m = randomFloats(n * n, 6);
            std::vector<float> q(n * n);
            std::vector<float> r(n * n);
            cpu.qrDecomposeGpu(m, n, n, q, r);
        };
        const auto qrGpu = [&](std::size_t n) {
            const auto m = randomFloats(n * n, 6);
            std::vector<float> q(n * n);
            std::vector<float> r(n * n);
            gpu->qrDecomposeGpu(m, n, n, q, r);
        };
        findCrossover(
            "qrDecompose (Eigen)", "rows=cols", sizes,
            [](std::size_t n) { return n * n; }, qrCpu, qrGpu, 5);
    }

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128, 256};
        const auto jacobiCpu = [&](std::size_t n) {
            std::vector<float> m(n * n);
            const auto noise = randomFloats(n * n, 7);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    const float v = noise[i * n + j];
                    m[i * n + j] = v;
                    m[j * n + i] = v;
                }
                m[i * n + i] += static_cast<float>(n);
            }
            std::vector<float> diagonal(n * n);
            std::vector<float> vectors(n * n);
            cpu.jacobiEigenSymmetricGpu(m, n, 30, 1e-6f, diagonal, vectors);
        };
        const auto jacobiGpu = [&](std::size_t n) {
            std::vector<float> m(n * n);
            const auto noise = randomFloats(n * n, 7);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    const float v = noise[i * n + j];
                    m[i * n + j] = v;
                    m[j * n + i] = v;
                }
                m[i * n + i] += static_cast<float>(n);
            }
            std::vector<float> diagonal(n * n);
            std::vector<float> vectors(n * n);
            gpu->jacobiEigenSymmetricGpu(m, n, 30, 1e-6f, diagonal, vectors);
        };
        findCrossover(
            "jacobiEigenSymmetric (Eigen)", "n", sizes,
            [](std::size_t n) { return n * n; }, jacobiCpu, jacobiGpu, 3);
    }

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128, 256};
        const auto svdCpu = [&](std::size_t n) {
            const auto m = randomFloats(n * n, 13);
            std::vector<float> resultA(n * n);
            std::vector<float> resultV(n * n);
            cpu.jacobiSvdGpu(m, n, n, 30, 1e-6f, resultA, resultV);
        };
        const auto svdGpu = [&](std::size_t n) {
            const auto m = randomFloats(n * n, 13);
            std::vector<float> resultA(n * n);
            std::vector<float> resultV(n * n);
            gpu->jacobiSvdGpu(m, n, n, 30, 1e-6f, resultA, resultV);
        };
        findCrossover(
            "jacobiSvdGpu (Eigen)", "rows=cols", sizes,
            [](std::size_t n) { return n * n; }, svdCpu, svdGpu, 3);
    }

    // --- FFT: gated on data.size() (length, batchCount = 1) ---------------

    {
        const std::vector<std::size_t> sizes{256,  512,  1024,  2048,
                                             4096, 8192, 16384, 32768};
        const auto fftCpu = [&](std::size_t n) {
            const auto real = randomFloats(n, 8);
            const auto imag = randomFloats(n, 9);
            std::vector<float> nextReal(n);
            std::vector<float> nextImag(n);
            cpu.fftBatched(real, imag, n, 1, false, nextReal, nextImag);
        };
        const auto fftGpu = [&](std::size_t n) {
            const auto real = randomFloats(n, 8);
            const auto imag = randomFloats(n, 9);
            std::vector<float> nextReal(n);
            std::vector<float> nextImag(n);
            gpu->fftBatched(real, imag, n, 1, false, nextReal, nextImag);
        };
        findCrossover(
            "fft (FFT)", "length", sizes, [](std::size_t n) { return n; }, fftCpu, fftGpu,
            20);
    }

    // --- Multigrid: gated on fineNx*fineNy*fineNz --------------------------

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128};
        const auto restrictCpu = [&](std::size_t n) {
            const auto fine = randomFloats(n * n * n, 10);
            std::vector<float> coarse((n / 2) * (n / 2) * (n / 2));
            cpu.multigridRestrict3D(fine, n, n, n, coarse);
        };
        const auto restrictGpu = [&](std::size_t n) {
            const auto fine = randomFloats(n * n * n, 10);
            std::vector<float> coarse((n / 2) * (n / 2) * (n / 2));
            gpu->multigridRestrict3D(fine, n, n, n, coarse);
        };
        findCrossover(
            "multigridRestrict3D (Multigrid)", "n (n^3 cells)", sizes,
            [](std::size_t n) { return n * n * n; }, restrictCpu, restrictGpu, 10);
    }

    // --- Batch-evaluation family: gated on x.size() (erf as representative,
    // shared with erfc/gamma/logGamma/legendreP) ---------------------------

    {
        const std::vector<std::size_t> sizes{512,  1024,  2048,  4096,
                                             8192, 16384, 32768, 65536};
        const auto erfCpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 11);
            std::vector<float> result(n);
            cpu.batchErf(x, result);
        };
        const auto erfGpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 11);
            std::vector<float> result(n);
            gpu->batchErf(x, result);
        };
        findCrossover(
            "batchErf (SpecialFunctions/Polynomial/Interpolation)", "x.size()", sizes,
            [](std::size_t n) { return n; }, erfCpu, erfGpu, 20);
    }

    // --- Parallel RNG: gated on result.size() ------------------------------

    {
        const std::vector<std::size_t> sizes{512,  1024,  2048,  4096,
                                             8192, 16384, 32768, 65536};
        const auto uniformCpu = [&](std::size_t n) {
            std::vector<float> result(n);
            cpu.batchUniformReal(42, 0, result);
        };
        const auto uniformGpu = [&](std::size_t n) {
            std::vector<float> result(n);
            gpu->batchUniformReal(42, 0, result);
        };
        findCrossover(
            "batchUniformReal (Random)", "result.size()", sizes,
            [](std::size_t n) { return n; }, uniformCpu, uniformGpu, 20);
    }

    // --- Sort: gated on values.size() --------------------------------------

    {
        const std::vector<std::size_t> sizes{512,  1024,  2048,  4096,
                                             8192, 16384, 32768, 65536};
        const auto sortCpu = [&](std::size_t n) {
            std::vector<float> values = randomFloats(n, 12);
            cpu.sortAscending(values);
        };
        const auto sortGpu = [&](std::size_t n) {
            std::vector<float> values = randomFloats(n, 12);
            gpu->sortAscending(values);
        };
        findCrossover(
            "sortAscending (Sort)", "values.size()", sizes,
            [](std::size_t n) { return n; }, sortCpu, sortGpu, 10);
    }

    // --- Physics: pairwise N-body (gravity, electric field; magnetic field
    // shares electric field's threshold in Field.cpp, not measured
    // separately -- its extra per-pair velocity/cross-product work only
    // makes it more GPU-favorable, never less, so electric field's
    // crossover is the conservative choice for both) -----------------------

    {
        const std::vector<std::size_t> sizes{256, 512, 1024, 2048, 4096, 8192};
        const auto gravCpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 20);
            const auto y = randomFloats(n, 21);
            const auto z = randomFloats(n, 22);
            const auto gm = randomFloats(n, 23);
            std::vector<float> ax(n), ay(n), az(n);
            cpu.gravitationalNBody(x, y, z, gm, 1e-4f, ax, ay, az);
        };
        const auto gravGpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 20);
            const auto y = randomFloats(n, 21);
            const auto z = randomFloats(n, 22);
            const auto gm = randomFloats(n, 23);
            std::vector<float> ax(n), ay(n), az(n);
            gpu->gravitationalNBody(x, y, z, gm, 1e-4f, ax, ay, az);
        };
        findCrossover(
            "gravitationalNBody (Physics/Gravity/Newtonian)", "body count", sizes,
            [](std::size_t n) { return n; }, gravCpu, gravGpu, 10);
    }

    {
        const std::vector<std::size_t> sizes{256, 512, 1024, 2048, 4096, 8192};
        const auto efieldCpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 24);
            const auto y = randomFloats(n, 25);
            const auto z = randomFloats(n, 26);
            const auto q = randomFloats(n, 27);
            std::vector<float> fx(n), fy(n), fz(n);
            cpu.electricFieldNBody(x, y, z, q, 1.0f, fx, fy, fz);
        };
        const auto efieldGpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 24);
            const auto y = randomFloats(n, 25);
            const auto z = randomFloats(n, 26);
            const auto q = randomFloats(n, 27);
            std::vector<float> fx(n), fy(n), fz(n);
            gpu->electricFieldNBody(x, y, z, q, 1.0f, fx, fy, fz);
        };
        findCrossover(
            "electricFieldNBody (Physics/Electromagnetism/Field)", "charge count", sizes,
            [](std::size_t n) { return n; }, efieldCpu, efieldGpu, 10);
    }

    // --- Physics/Fluids/SPH: neighbor-bounded pairwise (density/pressure and
    // pressure-acceleration share one threshold in SPH.cpp; density/pressure
    // measured as representative). SPH.cpp's own below-threshold CPU path
    // is NOT the naive O(n^2) direct sum the GPU kernel is (unlike gravity's
    // and the electric field's CPU fallbacks, which are): it queries a
    // Math::KdTree3 for each particle's neighbors within the kernel's
    // compact support, an asymptotically better shape entirely. Comparing
    // the GPU kernel against a naive O(n^2) CPU loop here would badly
    // understate the real CPU path's performance and recommend dispatching
    // to the GPU far too early, so this replicates SPH.cpp's actual
    // tree-accelerated loop (using only its public API: Math::KdTree3 and
    // Physics::cubicSplineKernel) rather than reusing CpuBackend's simpler
    // reference kernel the way every other case in this file does.
    {
        const std::vector<std::size_t> sizes{256, 512, 1024, 2048, 4096, 8192, 16384};
        constexpr double smoothingLength = 0.5;
        constexpr double supportRadius = 2.0 * smoothingLength;

        const auto sphTreeCpu = [&](std::size_t n) {
            std::vector<ysq::Vec3> positions(n);
            std::vector<double> mass(n);
            const auto raw = randomFloats(n * 3, 28);
            for (std::size_t i = 0; i < n; ++i) {
                // Packed into a unit cube densely enough that the compact
                // support (radius 2h = 1.0) reaches a realistic, size-
                // independent neighbor count per particle, matching how an
                // actual SPH scenario is set up (fixed physical density,
                // not fixed domain size) -- growing the domain with n
                // instead would make the tree's own advantage disappear
                // (neighbor counts would shrink) without that reflecting
                // anything about a real simulation.
                const double side = std::cbrt(static_cast<double>(n));
                positions[i] = ysq::Vec3{static_cast<double>(raw[i * 3]) * side,
                                         static_cast<double>(raw[i * 3 + 1]) * side,
                                         static_cast<double>(raw[i * 3 + 2]) * side};
                mass[i] = 1.0;
            }
            const ysq::KdTree3<double> tree(positions);
            std::vector<double> density(n);
            for (std::size_t i = 0; i < n; ++i) {
                double total = 0.0;
                for (std::size_t j : tree.radiusQuery(positions[i], supportRadius)) {
                    const double r = length(positions[i] - positions[j]);
                    total += mass[j] * ysq::cubicSplineKernel(r, smoothingLength);
                }
                density[i] = total;
            }
        };
        const auto sphGpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 28);
            const auto y = randomFloats(n, 29);
            const auto z = randomFloats(n, 30);
            const auto mass = randomFloats(n, 31);
            std::vector<float> density(n), pressure(n);
            gpu->sphDensityPressure(x, y, z, mass, static_cast<float>(smoothingLength),
                                    1.0f, 2.0f, density, pressure);
        };
        findCrossover(
            "sphDensityPressure (Physics/Fluids/SPH, tree-accelerated CPU)",
            "particle count", sizes, [](std::size_t n) { return n; }, sphTreeCpu, sphGpu,
            5);
    }

    // --- Physics grid-stencil family: gated on nx*ny*nz -------------------

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128};
        const auto heatCpu = [&](std::size_t n) {
            const auto t = randomFloats(n * n * n, 32);
            std::vector<float> next(n * n * n);
            cpu.heatEquation3DStep(t, n, n, n, 0.1f, next);
        };
        const auto heatGpu = [&](std::size_t n) {
            const auto t = randomFloats(n * n * n, 32);
            std::vector<float> next(n * n * n);
            gpu->heatEquation3DStep(t, n, n, n, 0.1f, next);
        };
        findCrossover(
            "heatEquation3DStep (Physics/Thermodynamics/HeatEquation3D)", "n (n^3 cells)",
            sizes, [](std::size_t n) { return n * n * n; }, heatCpu, heatGpu, 10);
    }

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128};
        const auto acousticCpu = [&](std::size_t n) {
            const auto p = randomFloats(n * n * n, 33);
            const auto vx = randomFloats(n * n * n, 34);
            const auto vy = randomFloats(n * n * n, 35);
            const auto vz = randomFloats(n * n * n, 36);
            std::vector<float> nextP(n * n * n), nextVx(n * n * n), nextVy(n * n * n),
                nextVz(n * n * n);
            cpu.acoustic3DStep(p, vx, vy, vz, n, n, n, 0.1f, 0.1f, nextP, nextVx, nextVy,
                               nextVz);
        };
        const auto acousticGpu = [&](std::size_t n) {
            const auto p = randomFloats(n * n * n, 33);
            const auto vx = randomFloats(n * n * n, 34);
            const auto vy = randomFloats(n * n * n, 35);
            const auto vz = randomFloats(n * n * n, 36);
            std::vector<float> nextP(n * n * n), nextVx(n * n * n), nextVy(n * n * n),
                nextVz(n * n * n);
            gpu->acoustic3DStep(p, vx, vy, vz, n, n, n, 0.1f, 0.1f, nextP, nextVx, nextVy,
                                nextVz);
        };
        findCrossover(
            "acoustic3DStep (Physics/Acoustics/Acoustic3D)", "n (n^3 cells)", sizes,
            [](std::size_t n) { return n * n * n; }, acousticCpu, acousticGpu, 10);
    }

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128};
        const auto maxwellCpu = [&](std::size_t n) {
            const auto ex = randomFloats(n * n * n, 37);
            const auto ey = randomFloats(n * n * n, 38);
            const auto ez = randomFloats(n * n * n, 39);
            const auto bx = randomFloats(n * n * n, 40);
            const auto by = randomFloats(n * n * n, 41);
            const auto bz = randomFloats(n * n * n, 42);
            std::vector<float> nextEx(n * n * n), nextEy(n * n * n), nextEz(n * n * n),
                nextBx(n * n * n), nextBy(n * n * n), nextBz(n * n * n);
            cpu.maxwell3DStep(ex, ey, ez, bx, by, bz, n, n, n, 0.1f, 0.1f, nextEx, nextEy,
                              nextEz, nextBx, nextBy, nextBz);
        };
        const auto maxwellGpu = [&](std::size_t n) {
            const auto ex = randomFloats(n * n * n, 37);
            const auto ey = randomFloats(n * n * n, 38);
            const auto ez = randomFloats(n * n * n, 39);
            const auto bx = randomFloats(n * n * n, 40);
            const auto by = randomFloats(n * n * n, 41);
            const auto bz = randomFloats(n * n * n, 42);
            std::vector<float> nextEx(n * n * n), nextEy(n * n * n), nextEz(n * n * n),
                nextBx(n * n * n), nextBy(n * n * n), nextBz(n * n * n);
            gpu->maxwell3DStep(ex, ey, ez, bx, by, bz, n, n, n, 0.1f, 0.1f, nextEx,
                               nextEy, nextEz, nextBx, nextBy, nextBz);
        };
        findCrossover(
            "maxwell3DStep (Physics/Electromagnetism/Maxwell3D)", "n (n^3 cells)", sizes,
            [](std::size_t n) { return n * n * n; }, maxwellCpu, maxwellGpu, 10);
    }

    {
        const std::vector<std::size_t> sizes{8, 16, 32, 64, 128};
        const auto eulerCpu = [&](std::size_t n) {
            const auto density = randomFloats(n * n * n, 43);
            const auto momX = randomFloats(n * n * n, 44);
            const auto momY = randomFloats(n * n * n, 45);
            const auto momZ = randomFloats(n * n * n, 46);
            const auto energy = randomFloats(n * n * n, 47);
            std::vector<float> nextDensity(n * n * n), nextMomX(n * n * n),
                nextMomY(n * n * n), nextMomZ(n * n * n), nextEnergy(n * n * n);
            cpu.eulerianFluid3DSweep(density, momX, momY, momZ, energy, n, n, n, 0, 1.4f,
                                     0.1f, nextDensity, nextMomX, nextMomY, nextMomZ,
                                     nextEnergy);
        };
        const auto eulerGpu = [&](std::size_t n) {
            const auto density = randomFloats(n * n * n, 43);
            const auto momX = randomFloats(n * n * n, 44);
            const auto momY = randomFloats(n * n * n, 45);
            const auto momZ = randomFloats(n * n * n, 46);
            const auto energy = randomFloats(n * n * n, 47);
            std::vector<float> nextDensity(n * n * n), nextMomX(n * n * n),
                nextMomY(n * n * n), nextMomZ(n * n * n), nextEnergy(n * n * n);
            gpu->eulerianFluid3DSweep(density, momX, momY, momZ, energy, n, n, n, 0, 1.4f,
                                      0.1f, nextDensity, nextMomX, nextMomY, nextMomZ,
                                      nextEnergy);
        };
        findCrossover(
            "eulerianFluid3DSweep (Physics/Fluids/Eulerian3D)", "n (n^3 cells)", sizes,
            [](std::size_t n) { return n * n * n; }, eulerCpu, eulerGpu, 10);
    }

    // --- Physics/Mechanics/Hermite: dispatches through minIndex (a single
    // reduction, not a shape of its own) -------------------------------------

    {
        const std::vector<std::size_t> sizes{512, 1024, 2048, 4096, 8192, 16384};
        const auto minIndexCpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 48);
            (void)cpu.minIndex(x);
        };
        const auto minIndexGpu = [&](std::size_t n) {
            const auto x = randomFloats(n, 48);
            (void)gpu->minIndex(x);
        };
        findCrossover(
            "minIndex (Physics/Mechanics/Hermite)", "body count", sizes,
            [](std::size_t n) { return n; }, minIndexCpu, minIndexGpu, 20);
    }

    std::printf("\nDone. Fold the recommended thresholds above into each header's own\n"
                "kXxxGpuDispatchThreshold constant.\n");
    return 0;
}
