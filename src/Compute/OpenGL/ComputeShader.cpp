#include <Compute/OpenGL/ComputeShader.hpp>

#include <glad/gl.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace ysq {

namespace {

std::string shaderInfoLog(unsigned shader) {
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    std::vector<char> infoLog(static_cast<std::size_t>(length) + 1, '\0');
    glGetShaderInfoLog(shader, length, nullptr, infoLog.data());
    return infoLog.data();
}

std::string programInfoLog(unsigned program) {
    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    std::vector<char> infoLog(static_cast<std::size_t>(length) + 1, '\0');
    glGetProgramInfoLog(program, length, nullptr, infoLog.data());
    return infoLog.data();
}

}  // namespace

std::optional<ComputeShader> ComputeShader::compile(std::string_view source,
                                                    std::string* error) {
    const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* sourcePtr = source.data();
    const GLint sourceLength = static_cast<GLint>(source.size());
    glShaderSource(shader, 1, &sourcePtr, &sourceLength);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE) {
        if (error) {
            *error = shaderInfoLog(shader);
        }
        glDeleteShader(shader);
        return std::nullopt;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);  // the program keeps its own copy once linked

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        if (error) {
            *error = programInfoLog(program);
        }
        glDeleteProgram(program);
        return std::nullopt;
    }

    return std::optional<ComputeShader>{ComputeShader{program}};
}

ComputeShader::ComputeShader(ComputeShader&& other) noexcept
    // 0u, not 0: std::exchange<unsigned, int> assigns int into the unsigned
    // member inside <utility>'s own template body, which MSVC's strict
    // warning set flags as a signed/unsigned mismatch (C4365) even though
    // Clang and GCC do not, since the mismatch is between the deduced
    // template types rather than a literal a compiler can see is safe.
    : m_program(std::exchange(other.m_program, 0u)) {}

ComputeShader& ComputeShader::operator=(ComputeShader&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();
    m_program = std::exchange(other.m_program, 0u);
    return *this;
}

ComputeShader::~ComputeShader() {
    destroy();
}

void ComputeShader::destroy() noexcept {
    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
}

void ComputeShader::use() const {
    glUseProgram(m_program);
}

void ComputeShader::bindBuffer(unsigned binding, unsigned bufferHandle) const {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, bufferHandle);
}

void ComputeShader::setUniform(std::string_view name, float value) const {
    glUniform1f(glGetUniformLocation(m_program, std::string{name}.c_str()), value);
}

void ComputeShader::dispatch(unsigned groupsX, unsigned groupsY, unsigned groupsZ) const {
    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

}  // namespace ysq
