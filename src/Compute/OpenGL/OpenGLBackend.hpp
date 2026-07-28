#pragma once

#include <Compute/ComputeBackend.hpp>
#include <Compute/OpenGL/ComputeShader.hpp>

#include <Platform/Window.hpp>

namespace ysq {

/// Dispatches through a 4.3+ offscreen context, which is where compute shaders
/// start; see the comment on ContextSettings in Window.hpp. Compiled in only
/// under YSQ_BUILD_GRAPHICS.
///
/// macOS caps OpenGL at 4.1, so create() always fails there: there is no
/// route to a 4.3 context through OpenGL on that platform at all. See
/// docs/architecture.md for the Vulkan/Metal path that exists instead.
class OpenGLBackend final : public ComputeBackend {
public:
    /// Nullptr if no 4.3 context is available, or either reference kernel
    /// fails to compile, which should not happen for shaders shipped with the
    /// engine but is reported rather than assumed.
    [[nodiscard]] static std::unique_ptr<ComputeBackend> create();

    [[nodiscard]] ComputeBackendKind kind() const noexcept override {
        return ComputeBackendKind::OpenGL;
    }

    void saxpy(std::span<const float> x, std::span<float> y, float a) const override;
    [[nodiscard]] float sum(std::span<const float> x) const override;

private:
    OpenGLBackend(Window window, ComputeShader saxpy, ComputeShader sum) noexcept;

    /// Held for its lifetime: destroying it takes the context that every
    /// buffer and program here belongs to. Mutable because making it current
    /// is implementation state, not logical state a caller observes: saxpy()
    /// and sum() are const on the ComputeBackend interface, since dispatching
    /// a kernel does not change what this backend logically is.
    mutable Window m_window;
    ComputeShader m_saxpy;
    ComputeShader m_sum;
};

}  // namespace ysq
