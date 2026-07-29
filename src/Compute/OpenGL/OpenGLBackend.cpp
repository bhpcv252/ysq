#include <Compute/OpenGL/OpenGLBackend.hpp>

#include <Core/Logger.hpp>

#include <glad/gl.h>

#include <cassert>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ysq {

namespace {

// Compiled at 430 core to match the 4.3 context requested below.
constexpr std::string_view kSaxpySource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer XBuffer { float x[]; };
layout(std430, binding = 1) buffer YBuffer { float y[]; };
uniform float a;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= y.length()) return;
    y[i] = a * x[i] + y[i];
}
)glsl";

// A two-pass tree reduction: this dispatch reduces each work group's slice
// into one partial sum, and OpenGLBackend::sum() finishes the (tiny)
// remaining reduction over those partials on the CPU rather than earning a
// second shader. Written for correctness on any input size, not peak
// throughput; this is a reference kernel, not a tuned one.
constexpr std::string_view kSumSource = R"glsl(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer InBuffer { float data[]; };
layout(std430, binding = 1) buffer OutBuffer { float partials[]; };
shared float scratch[256];
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint local = gl_LocalInvocationID.x;
    scratch[local] = (i < data.length()) ? data[i] : 0.0;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            scratch[local] += scratch[local + stride];
        }
        barrier();
    }
    if (local == 0u) {
        partials[gl_WorkGroupID.x] = scratch[0];
    }
}
)glsl";

constexpr unsigned kWorkGroupSize = 256;

unsigned groupCountFor(std::size_t elementCount) {
    return static_cast<unsigned>((elementCount + kWorkGroupSize - 1) /
                                 static_cast<std::size_t>(kWorkGroupSize));
}

/// An SSBO holding `count` floats, or initialised from `data`. Bound and torn
/// down within a single saxpy()/sum() call; nothing here is meant to outlive
/// one dispatch.
class ShaderBuffer {
public:
    explicit ShaderBuffer(std::span<const float> data) {
        glGenBuffers(1, &m_handle);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(),
                     GL_DYNAMIC_DRAW);
    }

    explicit ShaderBuffer(std::size_t count) {
        glGenBuffers(1, &m_handle);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(count * sizeof(float)), nullptr,
                     GL_DYNAMIC_DRAW);
    }

    ShaderBuffer(const ShaderBuffer&) = delete;
    ShaderBuffer& operator=(const ShaderBuffer&) = delete;
    ~ShaderBuffer() { glDeleteBuffers(1, &m_handle); }

    void read(std::span<float> out) const {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(out.size() * sizeof(float)),
                           out.data());
    }

    [[nodiscard]] unsigned handle() const noexcept { return m_handle; }

private:
    unsigned m_handle = 0;
};

}  // namespace

std::unique_ptr<ComputeBackend> OpenGLBackend::create() {
    ContextSettings context;
    context.versionMajor = 4;
    context.versionMinor = 3;

    WindowError windowError;
    std::optional<Window> window = Window::createOffscreen(1, 1, context, &windowError);
    if (!window) {
        logging::debug("OpenGL compute backend unavailable: {}", windowError.message);
        return nullptr;
    }

    std::string shaderError;
    std::optional<ComputeShader> saxpy =
        ComputeShader::compile(kSaxpySource, &shaderError);
    if (!saxpy) {
        logging::debug("OpenGL compute backend unavailable: saxpy shader: {}",
                       shaderError);
        return nullptr;
    }
    std::optional<ComputeShader> sum = ComputeShader::compile(kSumSource, &shaderError);
    if (!sum) {
        logging::debug("OpenGL compute backend unavailable: sum shader: {}", shaderError);
        return nullptr;
    }

    return std::unique_ptr<ComputeBackend>{
        new OpenGLBackend(std::move(*window), std::move(*saxpy), std::move(*sum))};
}

OpenGLBackend::OpenGLBackend(Window window, ComputeShader saxpy,
                             ComputeShader sum) noexcept
    : m_window(std::move(window)), m_saxpy(std::move(saxpy)), m_sum(std::move(sum)) {}

void OpenGLBackend::saxpy(std::span<const float> x, std::span<float> y, float a) const {
    // Matches CpuBackend's precondition check: without this, a caller bug
    // here would silently truncate against y's length (the shader guards on
    // y.length(), not x.length()) instead of failing loudly like every other
    // backend does in a debug build.
    assert(x.size() == y.size() && "saxpy needs matching spans");
    m_window.makeContextCurrent();

    const ShaderBuffer xBuffer{x};
    const ShaderBuffer yBuffer{y};

    m_saxpy.use();
    m_saxpy.bindBuffer(0, xBuffer.handle());
    m_saxpy.bindBuffer(1, yBuffer.handle());
    m_saxpy.setUniform("a", a);
    m_saxpy.dispatch(groupCountFor(x.size()));

    yBuffer.read(y);
}

float OpenGLBackend::sum(std::span<const float> x) const {
    m_window.makeContextCurrent();

    const unsigned groups = groupCountFor(x.size());
    const ShaderBuffer inBuffer{x};
    const ShaderBuffer outBuffer{static_cast<std::size_t>(groups)};

    m_sum.use();
    m_sum.bindBuffer(0, inBuffer.handle());
    m_sum.bindBuffer(1, outBuffer.handle());
    m_sum.dispatch(groups);

    std::vector<float> partials(groups);
    outBuffer.read(partials);

    // One partial per work group, so finishing on the CPU does not earn a
    // third shader.
    float total = 0.0f;
    for (const float partial : partials) {
        total += partial;
    }
    return total;
}

}  // namespace ysq
