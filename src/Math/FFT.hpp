#pragma once

#include <Compute/ComputeBackend.hpp>
#include <Math/Complex.hpp>
#include <Math/Scalar.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace ysq {

/// The discrete Fourier transform, by the iterative radix-2 Cooley-Tukey
/// algorithm: `O(n log n)` rather than the `O(n^2)` a direct sum over the
/// DFT's own definition would cost, at the one precondition that makes the
/// recursive halving exact, `data.size()` a power of two.
///
/// Forward and inverse share one routine internally, differing only in the
/// twiddle factors' sign and a final `1/n` scale on the inverse — the same
/// symmetry the DFT and its inverse have in their own definitions.

namespace detail {

/// Reorders `data` so index `i` holds what belongs at the bit-reversal of
/// `i` (`i`'s binary digits read back to front): the standard first step of
/// an *iterative* Cooley-Tukey FFT, which processes the transform bottom-up
/// (pairs, then groups of four, then eight, ...) rather than top-down by
/// recursive call, and needs its input in this order for the bottom level's
/// pairs to be the correct ones.
template <std::floating_point T>
void bitReversalPermute(std::vector<Complex<T>>& data) {
    const std::size_t n = data.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

/// One forward or inverse transform, in place. `inverse` flips the sign of
/// the twiddle factors' angle (the DFT and its inverse differ only in the
/// sign of the exponent) and scales the result by `1/n` at the end (the
/// normalization the forward transform leaves out, by the usual convention
/// that only one direction carries it).
template <std::floating_point T>
void fftImpl(std::vector<Complex<T>>& data, bool inverse) {
    const std::size_t n = data.size();
    assert(n > 0 && (n & (n - 1)) == 0);

    bitReversalPermute(data);

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const T angle = (inverse ? T{1} : T{-1}) * kTau<T> / static_cast<T>(len);
        const Complex<T> lengthTwiddle = Complex<T>::polar(T{1}, angle);

        for (std::size_t blockStart = 0; blockStart < n; blockStart += len) {
            Complex<T> twiddle{T{1}, T{0}};
            const std::size_t half = len / 2;
            for (std::size_t k = 0; k < half; ++k) {
                const Complex<T> even = data[blockStart + k];
                const Complex<T> odd = data[blockStart + k + half] * twiddle;
                data[blockStart + k] = even + odd;
                data[blockStart + k + half] = even - odd;
                twiddle *= lengthTwiddle;
            }
        }
    }

    if (inverse) {
        const T scale = T{1} / static_cast<T>(n);
        for (Complex<T>& value : data) {
            value *= scale;
        }
    }
}

/// A 3D DFT factors exactly into three passes of 1D DFTs, one per axis —
/// the row-column algorithm: transforming every line along `x` first,
/// then every line along `y`, then every line along `z` (the order among
/// the three does not matter) reaches the same result a direct 3D
/// definition would, since each 1D pass only mixes points that already
/// share every other coordinate. Each pass extracts a line into a
/// contiguous scratch buffer (`fftImpl` needs contiguous storage; only the
/// `z` pass already has that for free, `x` and `y` lines are strided) and
/// writes it back at the same positions.
template <std::floating_point T>
void fft3DImpl(std::vector<Complex<T>>& data, std::size_t nx, std::size_t ny,
               std::size_t nz, bool inverse) {
    assert(data.size() == nx * ny * nz);

    std::vector<Complex<T>> lineX(nx);
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t i = 0; i < nx; ++i) {
                lineX[i] = data[(i * ny + j) * nz + k];
            }
            fftImpl(lineX, inverse);
            for (std::size_t i = 0; i < nx; ++i) {
                data[(i * ny + j) * nz + k] = lineX[i];
            }
        }
    }

    std::vector<Complex<T>> lineY(ny);
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                lineY[j] = data[(i * ny + j) * nz + k];
            }
            fftImpl(lineY, inverse);
            for (std::size_t j = 0; j < ny; ++j) {
                data[(i * ny + j) * nz + k] = lineY[j];
            }
        }
    }

    std::vector<Complex<T>> lineZ(nz);
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            const std::size_t base = (i * ny + j) * nz;
            for (std::size_t k = 0; k < nz; ++k) {
                lineZ[k] = data[base + k];
            }
            fftImpl(lineZ, inverse);
            for (std::size_t k = 0; k < nz; ++k) {
                data[base + k] = lineZ[k];
            }
        }
    }
}

/// Measured on the development machine (Apple Silicon, Metal backend) by
/// `benchmarks/compute_thresholds.cpp`: the smallest transform length at
/// which the GPU path actually beat the CPU reference. Every existing test
/// in tests/unit/math_fft.cpp uses `Complex<double>`, which never
/// dispatches (see below), so this threshold only bounds the
/// `Complex<float>` path, with no existing exact-precision test to stay
/// under. Re-run the benchmark and update this if the reference machine or
/// backend ever changes.
inline constexpr std::size_t kGpuDispatchThreshold = 16384;

/// Marshals a single 1D transform through Compute::defaultBackend(), only
/// ever called for `Complex<float>`: `fftBatched` is a `float`-only
/// interface (see src/Compute/README.md), so a `Complex<double>` caller
/// would lose precision silently if this ran for it, the same reason
/// CpuBackend keeps saxpyD/sumD outside the shared ComputeBackend interface
/// entirely rather than narrowing through it. `fft`/`ifft` below gate this
/// with `if constexpr` so a `double` instantiation never even mentions it.
inline void fftGpu(std::vector<Complex<float>>& data, bool inverse) {
    const std::size_t n = data.size();
    std::vector<float> real(n);
    std::vector<float> imag(n);
    for (std::size_t i = 0; i < n; ++i) {
        real[i] = data[i].re;
        imag[i] = data[i].im;
    }

    std::vector<float> nextReal(n);
    std::vector<float> nextImag(n);
    defaultBackend().fftBatched(real, imag, n, 1, inverse, nextReal, nextImag);

    for (std::size_t i = 0; i < n; ++i) {
        data[i] = Complex<float>{nextReal[i], nextImag[i]};
    }
}

/// The 3D row-column algorithm's GPU counterpart, only ever called for
/// `Complex<float>` (see fftGpu). Each axis pass is one batched
/// `fftBatched` dispatch: the x and y passes extract their (strided) lines
/// into a contiguous batched buffer first and scatter the result back
/// (the same reason fft3DImpl uses lineX/lineY scratch buffers on the CPU
/// side); the z pass needs no extraction, since `data`'s own flat layout
/// (`(i * ny + j) * nz + k`) already *is* `nx * ny` contiguous batches of
/// length `nz`.
inline void fft3DGpu(std::vector<Complex<float>>& data, std::size_t nx, std::size_t ny,
                     std::size_t nz, bool inverse) {
    ComputeBackend& backend = defaultBackend();

    {
        const std::size_t batchCount = ny * nz;
        std::vector<float> real(nx * batchCount);
        std::vector<float> imag(nx * batchCount);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t batch = j * nz + k;
                for (std::size_t i = 0; i < nx; ++i) {
                    const Complex<float>& c = data[(i * ny + j) * nz + k];
                    real[batch * nx + i] = c.re;
                    imag[batch * nx + i] = c.im;
                }
            }
        }
        std::vector<float> nextReal(nx * batchCount);
        std::vector<float> nextImag(nx * batchCount);
        backend.fftBatched(real, imag, nx, batchCount, inverse, nextReal, nextImag);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t batch = j * nz + k;
                for (std::size_t i = 0; i < nx; ++i) {
                    data[(i * ny + j) * nz + k] = Complex<float>{
                        nextReal[batch * nx + i], nextImag[batch * nx + i]};
                }
            }
        }
    }

    {
        const std::size_t batchCount = nx * nz;
        std::vector<float> real(ny * batchCount);
        std::vector<float> imag(ny * batchCount);
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t batch = i * nz + k;
                for (std::size_t j = 0; j < ny; ++j) {
                    const Complex<float>& c = data[(i * ny + j) * nz + k];
                    real[batch * ny + j] = c.re;
                    imag[batch * ny + j] = c.im;
                }
            }
        }
        std::vector<float> nextReal(ny * batchCount);
        std::vector<float> nextImag(ny * batchCount);
        backend.fftBatched(real, imag, ny, batchCount, inverse, nextReal, nextImag);
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t batch = i * nz + k;
                for (std::size_t j = 0; j < ny; ++j) {
                    data[(i * ny + j) * nz + k] = Complex<float>{
                        nextReal[batch * ny + j], nextImag[batch * ny + j]};
                }
            }
        }
    }

    {
        const std::size_t total = nx * ny * nz;
        std::vector<float> real(total);
        std::vector<float> imag(total);
        for (std::size_t idx = 0; idx < total; ++idx) {
            real[idx] = data[idx].re;
            imag[idx] = data[idx].im;
        }
        std::vector<float> nextReal(total);
        std::vector<float> nextImag(total);
        backend.fftBatched(real, imag, nz, nx * ny, inverse, nextReal, nextImag);
        for (std::size_t idx = 0; idx < total; ++idx) {
            data[idx] = Complex<float>{nextReal[idx], nextImag[idx]};
        }
    }
}

}  // namespace detail

/// The forward DFT, in place. `data.size()` must be a power of two.
template <std::floating_point T>
void fft(std::vector<Complex<T>>& data) {
    if constexpr (std::same_as<T, float>) {
        if (data.size() >= detail::kGpuDispatchThreshold) {
            detail::fftGpu(data, false);
            return;
        }
    }
    detail::fftImpl(data, false);
}

/// The inverse DFT, in place, normalized by `1/n` so `ifft(fft(x)) == x`
/// (to floating-point precision). `data.size()` must be a power of two.
template <std::floating_point T>
void ifft(std::vector<Complex<T>>& data) {
    if constexpr (std::same_as<T, float>) {
        if (data.size() >= detail::kGpuDispatchThreshold) {
            detail::fftGpu(data, true);
            return;
        }
    }
    detail::fftImpl(data, true);
}

/// The forward DFT of a real-valued signal: the convenience a caller with
/// plain samples (an audio buffer, a field sampled on a grid) actually has,
/// rather than a `Complex` sequence with every imaginary part already zero.
/// `samples.size()` must be a power of two.
template <std::floating_point T>
[[nodiscard]] std::vector<Complex<T>> fftReal(std::span<const T> samples) {
    std::vector<Complex<T>> data;
    data.reserve(samples.size());
    for (T sample : samples) {
        data.push_back(Complex<T>{sample, T{0}});
    }
    fft(data);
    return data;
}

/// The forward 3D DFT, in place, over a flat row-major buffer (`x` slowest,
/// `z` fastest, `data[(i * ny + j) * nz + k]`) of size `nx * ny * nz`. Each
/// of `nx`, `ny`, `nz` must independently be a power of two.
template <std::floating_point T>
void fft3D(std::vector<Complex<T>>& data, std::size_t nx, std::size_t ny,
           std::size_t nz) {
    if constexpr (std::same_as<T, float>) {
        if (nx * ny * nz >= detail::kGpuDispatchThreshold) {
            detail::fft3DGpu(data, nx, ny, nz, false);
            return;
        }
    }
    detail::fft3DImpl(data, nx, ny, nz, false);
}

/// The inverse 3D DFT, in place, normalized so `ifft3D(fft3D(x)) == x` (to
/// floating-point precision): each of the three 1D passes normalizes by
/// `1/n` for its own axis length, so the product of all three is exactly
/// the `1/(nx ny nz)` a direct 3D inverse DFT definition would apply once.
template <std::floating_point T>
void ifft3D(std::vector<Complex<T>>& data, std::size_t nx, std::size_t ny,
            std::size_t nz) {
    if constexpr (std::same_as<T, float>) {
        if (nx * ny * nz >= detail::kGpuDispatchThreshold) {
            detail::fft3DGpu(data, nx, ny, nz, true);
            return;
        }
    }
    detail::fft3DImpl(data, nx, ny, nz, true);
}

}  // namespace ysq
