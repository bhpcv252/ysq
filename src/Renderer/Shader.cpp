#include <Renderer/Shader.hpp>

#include <glad/gl.h>

#include <array>
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

/// nullopt and *error set on failure; the shader object is always deleted
/// either way, since a linked program keeps its own copy once attached.
std::optional<GLuint> compileStage(GLenum stage, std::string_view source,
                                   std::string* error) {
    const GLuint shader = glCreateShader(stage);
    const char* sourcePtr = source.data();
    const auto sourceLength = static_cast<GLint>(source.size());
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
    return shader;
}

}  // namespace

std::optional<Shader> Shader::compile(std::string_view vertexSource,
                                      std::string_view fragmentSource,
                                      std::string* error) {
    const std::optional<GLuint> vertex =
        compileStage(GL_VERTEX_SHADER, vertexSource, error);
    if (!vertex) {
        return std::nullopt;
    }
    const std::optional<GLuint> fragment =
        compileStage(GL_FRAGMENT_SHADER, fragmentSource, error);
    if (!fragment) {
        glDeleteShader(*vertex);
        return std::nullopt;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, *vertex);
    glAttachShader(program, *fragment);
    glLinkProgram(program);
    glDeleteShader(*vertex);
    glDeleteShader(*fragment);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        if (error) {
            *error = programInfoLog(program);
        }
        glDeleteProgram(program);
        return std::nullopt;
    }

    return std::optional<Shader>{Shader{program}};
}

Shader::Shader(Shader&& other) noexcept : m_program(std::exchange(other.m_program, 0u)) {}

Shader& Shader::operator=(Shader&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();
    m_program = std::exchange(other.m_program, 0u);
    return *this;
}

Shader::~Shader() {
    destroy();
}

void Shader::destroy() noexcept {
    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
}

void Shader::use() const {
    glUseProgram(m_program);
}

void Shader::setUniform(std::string_view name, int value) const {
    glUniform1i(glGetUniformLocation(m_program, std::string{name}.c_str()), value);
}

void Shader::setUniform(std::string_view name, float value) const {
    glUniform1f(glGetUniformLocation(m_program, std::string{name}.c_str()), value);
}

void Shader::setUniform(std::string_view name, const Vec3f& value) const {
    glUniform3f(glGetUniformLocation(m_program, std::string{name}.c_str()), value.x,
                value.y, value.z);
}

void Shader::setUniformArray(std::string_view name, std::span<const float> values) const {
    if (values.empty()) {
        return;
    }
    glUniform1fv(glGetUniformLocation(m_program, std::string{name}.c_str()),
                 static_cast<GLsizei>(values.size()), values.data());
}

void Shader::setUniformArray(std::string_view name, std::span<const Vec3f> values) const {
    if (values.empty()) {
        return;
    }
    // Vector3<float> is three tightly-packed floats with no padding, so the
    // span reinterprets directly into the flat array glUniform3fv wants.
    glUniform3fv(glGetUniformLocation(m_program, std::string{name}.c_str()),
                 static_cast<GLsizei>(values.size()),
                 reinterpret_cast<const float*>(values.data()));
}

void Shader::setUniform(std::string_view name, const Matrix4<float>& value) const {
    // Matrix4's only member is a column-major std::array<Vector4<float>, 4>,
    // each a standard-layout {x, y, z, w}: 16 contiguous floats in exactly
    // the layout glUniformMatrix4fv wants, so no staging buffer is needed.
    glUniformMatrix4fv(glGetUniformLocation(m_program, std::string{name}.c_str()), 1,
                       GL_FALSE, reinterpret_cast<const float*>(&value));
}

}  // namespace ysq
