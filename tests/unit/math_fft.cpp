#include <Math/FFT.hpp>

#include <Math/Complex.hpp>
#include <Math/Random.hpp>
#include <Math/Scalar.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using ysq::Complex;

}  // namespace

TEST(MathFFT, ForwardTransformOfAConstantSignalIsAllEnergyAtZeroFrequency) {
    // A DC (constant) signal transforms to N at bin 0 and 0 everywhere else.
    std::vector<Complex<double>> data(8, Complex<double>{3.0, 0.0});
    ysq::fft(data);

    EXPECT_NEAR(data[0].re, 24.0, 1e-9);  // 8 * 3
    EXPECT_NEAR(data[0].im, 0.0, 1e-9);
    for (std::size_t i = 1; i < data.size(); ++i) {
        EXPECT_NEAR(data[i].re, 0.0, 1e-9);
        EXPECT_NEAR(data[i].im, 0.0, 1e-9);
    }
}

TEST(MathFFT, ForwardTransformOfAPureSinusoidPeaksAtItsOwnFrequencyBin) {
    // A pure cosine at frequency k=2 over N=16 samples should show up as two
    // symmetric spikes at bins 2 and 14 (N - k), and nothing else.
    constexpr std::size_t kN = 16;
    constexpr std::size_t kFrequencyBin = 2;
    std::vector<Complex<double>> data(kN);
    for (std::size_t n = 0; n < kN; ++n) {
        const double angle = ysq::kTau<double> * static_cast<double>(kFrequencyBin * n) /
                             static_cast<double>(kN);
        data[n] = Complex<double>{std::cos(angle), 0.0};
    }

    ysq::fft(data);

    for (std::size_t k = 0; k < kN; ++k) {
        const double magnitude =
            std::sqrt(data[k].re * data[k].re + data[k].im * data[k].im);
        if (k == kFrequencyBin || k == kN - kFrequencyBin) {
            EXPECT_NEAR(magnitude, static_cast<double>(kN) / 2.0, 1e-9);
        } else {
            EXPECT_NEAR(magnitude, 0.0, 1e-9);
        }
    }
}

TEST(MathFFT, InverseUndoesForwardForAnArbitrarySignal) {
    std::vector<Complex<double>> original{
        Complex<double>{1.0, 0.5},  Complex<double>{2.0, -1.0},
        Complex<double>{-3.0, 2.0}, Complex<double>{0.5, 0.0},
        Complex<double>{4.0, 4.0},  Complex<double>{-2.0, -2.0},
        Complex<double>{1.5, 3.0},  Complex<double>{0.0, -1.0}};
    std::vector<Complex<double>> roundTripped = original;

    ysq::fft(roundTripped);
    ysq::ifft(roundTripped);

    for (std::size_t i = 0; i < original.size(); ++i) {
        EXPECT_NEAR(roundTripped[i].re, original[i].re, 1e-9);
        EXPECT_NEAR(roundTripped[i].im, original[i].im, 1e-9);
    }
}

TEST(MathFFT, ForwardTransformMatchesTheDirectDFTDefinitionOnASmallCase) {
    // Cross-check against the O(n^2) definition directly, independent of the
    // FFT's own internal structure: X_k = sum_n x_n exp(-2 pi i k n / N).
    const std::vector<Complex<double>> input{
        Complex<double>{1.0, 0.0}, Complex<double>{2.0, 0.0}, Complex<double>{0.0, 1.0},
        Complex<double>{-1.0, 3.0}};
    const std::size_t n = input.size();

    std::vector<Complex<double>> viaFft = input;
    ysq::fft(viaFft);

    for (std::size_t k = 0; k < n; ++k) {
        Complex<double> expected{0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            const double angle =
                -ysq::kTau<double> * static_cast<double>(k * j) / static_cast<double>(n);
            expected += input[j] * Complex<double>::polar(1.0, angle);
        }
        EXPECT_NEAR(viaFft[k].re, expected.re, 1e-9);
        EXPECT_NEAR(viaFft[k].im, expected.im, 1e-9);
    }
}

TEST(MathFFT, FftRealMatchesFftOfTheSameSamplesWithZeroImaginaryParts) {
    const std::array<double, 4> samples{1.0, 2.0, 3.0, 4.0};
    const std::vector<Complex<double>> viaFftReal = ysq::fftReal<double>(samples);

    std::vector<Complex<double>> viaFft{
        Complex<double>{1.0, 0.0}, Complex<double>{2.0, 0.0}, Complex<double>{3.0, 0.0},
        Complex<double>{4.0, 0.0}};
    ysq::fft(viaFft);

    for (std::size_t i = 0; i < viaFft.size(); ++i) {
        EXPECT_NEAR(viaFftReal[i].re, viaFft[i].re, 1e-9);
        EXPECT_NEAR(viaFftReal[i].im, viaFft[i].im, 1e-9);
    }
}

TEST(MathFFT, ForwardTransform3DOfAConstantSignalIsAllEnergyAtZeroFrequency) {
    constexpr std::size_t kNx = 4;
    constexpr std::size_t kNy = 2;
    constexpr std::size_t kNz = 2;
    std::vector<Complex<double>> data(kNx * kNy * kNz, Complex<double>{2.0, 0.0});

    ysq::fft3D(data, kNx, kNy, kNz);

    EXPECT_NEAR(data[0].re, static_cast<double>(data.size()) * 2.0, 1e-9);
    EXPECT_NEAR(data[0].im, 0.0, 1e-9);
    for (std::size_t i = 1; i < data.size(); ++i) {
        EXPECT_NEAR(data[i].re, 0.0, 1e-9);
        EXPECT_NEAR(data[i].im, 0.0, 1e-9);
    }
}

TEST(MathFFT, Inverse3DUndoesForward3DForRandomData) {
    constexpr std::size_t kNx = 4;
    constexpr std::size_t kNy = 8;
    constexpr std::size_t kNz = 2;

    ysq::RandomEngine engine = ysq::makeRandomEngine(11);
    std::vector<Complex<double>> original(kNx * kNy * kNz);
    for (Complex<double>& value : original) {
        value = Complex<double>{ysq::uniformReal(engine, -5.0, 5.0),
                                ysq::uniformReal(engine, -5.0, 5.0)};
    }

    std::vector<Complex<double>> roundTripped = original;
    ysq::fft3D(roundTripped, kNx, kNy, kNz);
    ysq::ifft3D(roundTripped, kNx, kNy, kNz);

    for (std::size_t i = 0; i < original.size(); ++i) {
        EXPECT_NEAR(roundTripped[i].re, original[i].re, 1e-9);
        EXPECT_NEAR(roundTripped[i].im, original[i].im, 1e-9);
    }
}

TEST(MathFFT, Forward3DWithTwoFlatAxesAgreesWithThePlain1DTransform) {
    // With ny = nz = 1 the 3D transform has nowhere else to mix into, so it
    // must reduce exactly to the 1D transform along the one real axis.
    constexpr std::size_t kNx = 8;
    std::vector<Complex<double>> original{
        Complex<double>{1.0, 0.5},  Complex<double>{2.0, -1.0},
        Complex<double>{-3.0, 2.0}, Complex<double>{0.5, 0.0},
        Complex<double>{4.0, 4.0},  Complex<double>{-2.0, -2.0},
        Complex<double>{1.5, 3.0},  Complex<double>{0.0, -1.0}};

    std::vector<Complex<double>> via3D = original;
    ysq::fft3D(via3D, kNx, 1, 1);

    std::vector<Complex<double>> via1D = original;
    ysq::fft(via1D);

    for (std::size_t i = 0; i < kNx; ++i) {
        EXPECT_NEAR(via3D[i].re, via1D[i].re, 1e-9);
        EXPECT_NEAR(via3D[i].im, via1D[i].im, 1e-9);
    }
}

TEST(MathFFT, ForwardTransformAtLargeNAgreesWithTheDcBinPropertyOnTheGpuPath) {
    // Above Math/FFT.hpp's own GPU dispatch threshold and Complex<float>
    // (the only type that ever dispatches; see detail::fftGpu), so fft()
    // exercises whichever compute backend is actually available. Checks
    // the same hand-derivable property
    // ForwardTransformOfAConstantSignalIsAllEnergyAtZeroFrequency does at
    // small N (a constant signal's DFT is n at bin 0, 0 elsewhere), which
    // fftGpu's marshaling has to get exactly right end to end to pass.
    // Above kGpuDispatchThreshold (16384; measured by
    // benchmarks/compute_thresholds.cpp).
    constexpr std::size_t kN = 16384;
    std::vector<Complex<float>> data(kN, Complex<float>{3.0f, 0.0f});

    ysq::fft(data);

    EXPECT_NEAR(data[0].re, 3.0f * static_cast<float>(kN), 1.0f);
    EXPECT_NEAR(data[0].im, 0.0f, 1.0f);
    for (const std::size_t i : {kN / 4, kN / 2, kN - 1}) {
        EXPECT_NEAR(data[i].re, 0.0f, 1.0f) << "bin " << i;
        EXPECT_NEAR(data[i].im, 0.0f, 1.0f) << "bin " << i;
    }
}

TEST(MathFFT, InverseUndoesForwardAtLargeNOnTheGpuPath) {
    constexpr std::size_t kN = 16384;  // above kGpuDispatchThreshold
    std::vector<Complex<float>> original(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        original[i] = Complex<float>{std::sin(0.01f * static_cast<float>(i)),
                                     std::cos(0.02f * static_cast<float>(i))};
    }

    std::vector<Complex<float>> roundTripped = original;
    ysq::fft(roundTripped);
    ysq::ifft(roundTripped);

    for (const std::size_t i : {std::size_t{0}, kN / 3, kN - 1}) {
        EXPECT_NEAR(roundTripped[i].re, original[i].re, 1e-2f) << "index " << i;
        EXPECT_NEAR(roundTripped[i].im, original[i].im, 1e-2f) << "index " << i;
    }
}

TEST(MathFFT, Forward3DAtLargeNAgreesWithTheDcBinPropertyOnTheGpuPath) {
    // Above the GPU dispatch threshold and Complex<float>, so all three
    // axis passes inside detail::fft3DGpu dispatch: a bug in the x or y
    // pass's line-extraction/scatter indexing would leak energy into a
    // non-DC bin here, which the small-N CPU-path equivalent test above
    // cannot exercise at all.
    // 32*32*16 = 16384, at kGpuDispatchThreshold (measured by
    // benchmarks/compute_thresholds.cpp).
    constexpr std::size_t kNx = 32, kNy = 32, kNz = 16;
    std::vector<Complex<float>> data(kNx * kNy * kNz, Complex<float>{2.0f, 0.0f});

    ysq::fft3D(data, kNx, kNy, kNz);

    EXPECT_NEAR(data[0].re, 2.0f * static_cast<float>(data.size()), 1.0f);
    EXPECT_NEAR(data[0].im, 0.0f, 1.0f);
    for (const std::size_t i : {data.size() / 4, data.size() / 2, data.size() - 1}) {
        EXPECT_NEAR(data[i].re, 0.0f, 1.0f) << "index " << i;
        EXPECT_NEAR(data[i].im, 0.0f, 1.0f) << "index " << i;
    }
}

TEST(MathFFT, Inverse3DUndoesForward3DAtLargeNOnTheGpuPath) {
    constexpr std::size_t kNx = 32, kNy = 32, kNz = 16;  // 32*32*16 = 16384
    ysq::RandomEngine rng = ysq::makeRandomEngine(7);
    std::vector<Complex<float>> original(kNx * kNy * kNz);
    for (Complex<float>& c : original) {
        c = Complex<float>{ysq::uniformReal<float>(rng, -1.0f, 1.0f),
                           ysq::uniformReal<float>(rng, -1.0f, 1.0f)};
    }

    std::vector<Complex<float>> roundTripped = original;
    ysq::fft3D(roundTripped, kNx, kNy, kNz);
    ysq::ifft3D(roundTripped, kNx, kNy, kNz);

    for (const std::size_t i :
         {std::size_t{0}, original.size() / 2, original.size() - 1}) {
        EXPECT_NEAR(roundTripped[i].re, original[i].re, 1e-2f) << "index " << i;
        EXPECT_NEAR(roundTripped[i].im, original[i].im, 1e-2f) << "index " << i;
    }
}
