#pragma once

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

}  // namespace detail

/// The forward DFT, in place. `data.size()` must be a power of two.
template <std::floating_point T>
void fft(std::vector<Complex<T>>& data) {
    detail::fftImpl(data, false);
}

/// The inverse DFT, in place, normalized by `1/n` so `ifft(fft(x)) == x`
/// (to floating-point precision). `data.size()` must be a power of two.
template <std::floating_point T>
void ifft(std::vector<Complex<T>>& data) {
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
    detail::fft3DImpl(data, nx, ny, nz, false);
}

/// The inverse 3D DFT, in place, normalized so `ifft3D(fft3D(x)) == x` (to
/// floating-point precision): each of the three 1D passes normalizes by
/// `1/n` for its own axis length, so the product of all three is exactly
/// the `1/(nx ny nz)` a direct 3D inverse DFT definition would apply once.
template <std::floating_point T>
void ifft3D(std::vector<Complex<T>>& data, std::size_t nx, std::size_t ny,
            std::size_t nz) {
    detail::fft3DImpl(data, nx, ny, nz, true);
}

}  // namespace ysq
