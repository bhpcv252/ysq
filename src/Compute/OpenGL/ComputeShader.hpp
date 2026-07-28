#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace ysq {

/// A compiled, linked GLSL compute shader program.
///
/// The context this was compiled under must already be current for every
/// method here; ComputeShader never makes one current itself. OpenGLBackend
/// owns that decision.
class ComputeShader {
public:
    [[nodiscard]] static std::optional<ComputeShader>
    compile(std::string_view source, std::string* error = nullptr);

    ComputeShader(const ComputeShader&) = delete;
    ComputeShader& operator=(const ComputeShader&) = delete;
    ComputeShader(ComputeShader&& other) noexcept;
    ComputeShader& operator=(ComputeShader&& other) noexcept;
    ~ComputeShader();

    /// Makes this the active program. bindBuffer and setUniform both need
    /// this called first, same as any other OpenGL state change.
    void use() const;

    /// Binds `bufferHandle` to the shader storage block declared at `binding`
    /// in the source (`layout(std430, binding = N)`).
    void bindBuffer(unsigned binding, unsigned bufferHandle) const;

    void setUniform(std::string_view name, float value) const;

    /// Dispatches groupsX * groupsY * groupsZ work groups and waits for every
    /// shader storage write to become visible before returning.
    void dispatch(unsigned groupsX, unsigned groupsY = 1, unsigned groupsZ = 1) const;

private:
    explicit ComputeShader(unsigned program) noexcept : m_program(program) {}
    void destroy() noexcept;

    unsigned m_program = 0;
};

}  // namespace ysq
