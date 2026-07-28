#include <Compute/CPU/CpuBackend.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>

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
