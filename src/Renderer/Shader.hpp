#pragma once

#include <Math/Matrix4.hpp>
#include <Math/Vector3.hpp>

#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ysq {

/// A compiled, linked GLSL vertex+fragment program.
///
/// The context this was compiled under must already be current for every
/// method here; Shader never makes one current itself, the same rule
/// ComputeShader follows. See Compute/OpenGL/ComputeShader.hpp.
class Shader {
public:
    [[nodiscard]] static std::optional<Shader> compile(std::string_view vertexSource,
                                                       std::string_view fragmentSource,
                                                       std::string* error = nullptr);

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;
    Shader(Shader&& other) noexcept;
    Shader& operator=(Shader&& other) noexcept;
    ~Shader();

    void use() const;

    void setUniform(std::string_view name, int value) const;
    void setUniform(std::string_view name, float value) const;
    void setUniform(std::string_view name, const Vec3f& value) const;
    void setUniform(std::string_view name, const Matrix4<float>& value) const;

    /// Sets a prefix of a GLSL uniform array, e.g. `float uName[N];`. RayTracer
    /// uses these instead of a UBO to upload its scene: plain uniform arrays
    /// need no manual std140 padding to get right, and they stay GL 4.1
    /// portable exactly as well as a UBO would. Leaves elements beyond
    /// `values.size()` untouched; the shader is expected to know the true
    /// count from a separate int uniform and ignore the rest.
    void setUniformArray(std::string_view name, std::span<const float> values) const;
    void setUniformArray(std::string_view name, std::span<const Vec3f> values) const;

private:
    explicit Shader(unsigned program) noexcept : m_program(program) {}
    void destroy() noexcept;

    unsigned m_program = 0;
};

}  // namespace ysq
