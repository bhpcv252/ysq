#include <Compute/ComputeBackend.hpp>

#include <gtest/gtest.h>

#include <memory>

namespace {

using ysq::ComputeBackend;
using ysq::ComputeBackendKind;

TEST(ComputeBackendSelection, CpuIsAlwaysAvailable) {
    EXPECT_TRUE(ysq::computeBackendAvailable(ComputeBackendKind::Cpu));
}

TEST(ComputeBackendSelection, SelectionAlwaysReturnsSomething) {
    const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend();
    ASSERT_NE(backend, nullptr) << "the CPU backend must be the floor with no hardware";
}

TEST(ComputeBackendSelection, TheSelectedBackendReportsItselfAvailable) {
    const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend();
    ASSERT_NE(backend, nullptr);
    EXPECT_TRUE(ysq::computeBackendAvailable(backend->kind()))
        << "whatever was selected must also say it is available";
}

TEST(ComputeBackendSelection, ForcingCpuAlwaysSucceeds) {
    const std::unique_ptr<ComputeBackend> backend =
        ysq::selectComputeBackend(ComputeBackendKind::Cpu);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->kind(), ComputeBackendKind::Cpu);
}

// This is the one negative path testable on every machine regardless of
// hardware: probing for a backend that is not there must degrade to nullptr
// cleanly rather than crash, whether that is because no SDK was compiled in
// or because the SDK found no device.
TEST(ComputeBackendSelection, ForcingAnUnavailableBackendReturnsNullNotACrash) {
    for (const ComputeBackendKind kind :
         {ComputeBackendKind::OpenGL, ComputeBackendKind::Cuda,
          ComputeBackendKind::Vulkan}) {
        if (ysq::computeBackendAvailable(kind)) {
            continue;  // actually available here; nothing to prove for this kind
        }
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        EXPECT_EQ(backend, nullptr) << ysq::toString(kind);
    }
}

TEST(ComputeBackendSelection, ToStringNamesEveryKind) {
    EXPECT_EQ(ysq::toString(ComputeBackendKind::Cpu), "CPU");
    EXPECT_EQ(ysq::toString(ComputeBackendKind::OpenGL), "OpenGL");
    EXPECT_EQ(ysq::toString(ComputeBackendKind::Cuda), "CUDA");
    EXPECT_EQ(ysq::toString(ComputeBackendKind::Vulkan), "Vulkan");
}

}  // namespace
