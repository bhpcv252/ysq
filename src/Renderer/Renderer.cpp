#include <Renderer/Renderer.hpp>

#include <Renderer/shaders/Basic.frag.hpp>
#include <Renderer/shaders/Basic.vert.hpp>
#include <Renderer/shaders/Instanced.vert.hpp>
#include <Renderer/shaders/Skybox.frag.hpp>
#include <Renderer/shaders/Skybox.vert.hpp>

#include <glad/gl.h>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace ysq {

// --- RenderTarget --------------------------------------------------------------

std::optional<RenderTarget> RenderTarget::create(int width, int height, int samples) {
    if (width <= 0 || height <= 0) {
        return std::nullopt;
    }
    const bool multisampled = samples > 0;

    unsigned fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    unsigned colorTexture = 0;
    unsigned depthRenderbuffer = 0;
    glGenTextures(1, &colorTexture);
    glGenRenderbuffers(1, &depthRenderbuffer);

    if (multisampled) {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, colorTexture);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGBA8, width,
                                height, GL_TRUE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D_MULTISAMPLE, colorTexture, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH24_STENCIL8,
                                         width, height);
    } else {
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               colorTexture, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    }
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, depthRenderbuffer);

    const bool complete =
        glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (!complete) {
        glDeleteTextures(1, &colorTexture);
        glDeleteRenderbuffers(1, &depthRenderbuffer);
        glDeleteFramebuffers(1, &fbo);
        return std::nullopt;
    }

    return std::optional<RenderTarget>{
        RenderTarget{fbo, colorTexture, depthRenderbuffer, width, height, multisampled}};
}

RenderTarget::RenderTarget(RenderTarget&& other) noexcept
    : m_fbo(std::exchange(other.m_fbo, 0u)),
      m_colorTexture(std::exchange(other.m_colorTexture, 0u)),
      m_depthRenderbuffer(std::exchange(other.m_depthRenderbuffer, 0u)),
      m_width(std::exchange(other.m_width, 0)),
      m_height(std::exchange(other.m_height, 0)),
      m_multisampled(std::exchange(other.m_multisampled, false)) {}

RenderTarget& RenderTarget::operator=(RenderTarget&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();
    m_fbo = std::exchange(other.m_fbo, 0u);
    m_colorTexture = std::exchange(other.m_colorTexture, 0u);
    m_depthRenderbuffer = std::exchange(other.m_depthRenderbuffer, 0u);
    m_width = std::exchange(other.m_width, 0);
    m_height = std::exchange(other.m_height, 0);
    m_multisampled = std::exchange(other.m_multisampled, false);
    return *this;
}

RenderTarget::~RenderTarget() {
    destroy();
}

void RenderTarget::destroy() noexcept {
    if (m_colorTexture != 0) {
        glDeleteTextures(1, &m_colorTexture);
    }
    if (m_depthRenderbuffer != 0) {
        glDeleteRenderbuffers(1, &m_depthRenderbuffer);
    }
    if (m_fbo != 0) {
        glDeleteFramebuffers(1, &m_fbo);
    }
    m_fbo = m_colorTexture = m_depthRenderbuffer = 0;
}

void RenderTarget::bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
}

void RenderTarget::bindDefault() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

std::vector<std::uint8_t> RenderTarget::readPixels() const {
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(m_width) *
                                     static_cast<std::size_t>(m_height) * 4);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return pixels;
}

void RenderTarget::resolveTo(const RenderTarget& destination) const {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, destination.m_fbo);
    glBlitFramebuffer(0, 0, m_width, m_height, 0, 0, destination.m_width,
                      destination.m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// --- Renderer --------------------------------------------------------------------

namespace {

constexpr std::size_t kMaxPointLights = 8;
constexpr std::size_t kMaxDirectionalLights = 4;

}  // namespace

std::optional<Renderer> Renderer::create(std::string* error) {
    std::optional<Shader> basicShader =
        Shader::compile(shaders::kBasicVertSource, shaders::kBasicFragSource, error);
    if (!basicShader) {
        return std::nullopt;
    }
    std::optional<Shader> instancedShader =
        Shader::compile(shaders::kInstancedVertSource, shaders::kBasicFragSource, error);
    if (!instancedShader) {
        return std::nullopt;
    }
    std::optional<Shader> skyboxShader =
        Shader::compile(shaders::kSkyboxVertSource, shaders::kSkyboxFragSource, error);
    if (!skyboxShader) {
        return std::nullopt;
    }
    std::optional<DebugDraw> debugDraw = DebugDraw::create(error);
    if (!debugDraw) {
        return std::nullopt;
    }
    std::optional<Mesh> skyboxCube = Mesh::cube(2.0f);
    if (!skyboxCube) {
        return std::nullopt;
    }

    return std::optional<Renderer>{Renderer{
        std::move(*basicShader), std::move(*instancedShader), std::move(*skyboxShader),
        std::move(*debugDraw), std::move(*skyboxCube)}};
}

void Renderer::beginFrame(const Camera& camera, float aspect, int viewportWidth,
                          int viewportHeight, const Vec3f& clearColor) {
    glViewport(0, 0, viewportWidth, viewportHeight);
    glEnable(GL_DEPTH_TEST);
    glClearColor(clearColor.x, clearColor.y, clearColor.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_view = camera.viewMatrix();
    m_projection = camera.projectionMatrix(aspect);
    m_cameraPosition = camera.position;
    m_pointLights.clear();
    m_directionalLights.clear();
    m_drawCallCount = 0;
}

void Renderer::setLights(std::span<const PointLight> pointLights,
                         std::span<const DirectionalLight> directionalLights) {
    m_pointLights.assign(pointLights.begin(), pointLights.end());
    m_directionalLights.assign(directionalLights.begin(), directionalLights.end());
}

void Renderer::applyLights(const Shader& shader) const {
    shader.setUniform("uCameraPosition", m_cameraPosition);

    const std::size_t pointCount = std::min(m_pointLights.size(), kMaxPointLights);
    std::vector<Vec3f> position;
    std::vector<Vec3f> color;
    std::vector<float> intensity;
    std::vector<float> radius;
    for (std::size_t i = 0; i < pointCount; ++i) {
        position.push_back(m_pointLights[i].position);
        color.push_back(m_pointLights[i].color);
        intensity.push_back(m_pointLights[i].intensity);
        radius.push_back(m_pointLights[i].radius);
    }
    shader.setUniform("uPointLightCount", static_cast<int>(pointCount));
    shader.setUniformArray("uPointLightPosition", position);
    shader.setUniformArray("uPointLightColor", color);
    shader.setUniformArray("uPointLightIntensity", intensity);
    shader.setUniformArray("uPointLightRadius", radius);

    const std::size_t directionalCount =
        std::min(m_directionalLights.size(), kMaxDirectionalLights);
    std::vector<Vec3f> direction;
    std::vector<Vec3f> directionalColor;
    std::vector<float> directionalIntensity;
    for (std::size_t i = 0; i < directionalCount; ++i) {
        direction.push_back(m_directionalLights[i].direction);
        directionalColor.push_back(m_directionalLights[i].color);
        directionalIntensity.push_back(m_directionalLights[i].intensity);
    }
    shader.setUniform("uDirectionalLightCount", static_cast<int>(directionalCount));
    shader.setUniformArray("uDirectionalLightDirection", direction);
    shader.setUniformArray("uDirectionalLightColor", directionalColor);
    shader.setUniformArray("uDirectionalLightIntensity", directionalIntensity);
}

namespace {

void applyMaterial(const Shader& shader, const Material& material) {
    shader.setUniform("uAlbedo", material.albedo);
    shader.setUniform("uEmissive", material.emissive);
    shader.setUniform("uAmbient", material.ambient);
    shader.setUniform("uDiffuse", material.diffuse);
    shader.setUniform("uSpecular", material.specular);
    shader.setUniform("uShininess", material.shininess);
}

}  // namespace

void Renderer::draw(const Mesh& mesh, const Material& material,
                    const Matrix4<float>& model) {
    m_basicShader.use();
    m_basicShader.setUniform("uModel", model);
    m_basicShader.setUniform("uViewProjection", m_projection * m_view);
    applyMaterial(m_basicShader, material);
    applyLights(m_basicShader);
    mesh.draw();
    ++m_drawCallCount;
}

void Renderer::drawInstanced(const Mesh& mesh, const Material& material) {
    m_instancedShader.use();
    m_instancedShader.setUniform("uViewProjection", m_projection * m_view);
    applyMaterial(m_instancedShader, material);
    applyLights(m_instancedShader);
    mesh.drawInstanced();
    ++m_drawCallCount;
}

void Renderer::drawSkybox(const Cubemap& sky) {
    glDepthFunc(GL_LEQUAL);
    m_skyboxShader.use();

    // Rotation only: zeroing the translation column keeps the skybox from
    // appearing to move as the camera does.
    Matrix4<float> viewNoTranslation = m_view;
    viewNoTranslation.columns[3] = Vector4<float>{0.0f, 0.0f, 0.0f, 1.0f};
    m_skyboxShader.setUniform("uView", viewNoTranslation);
    m_skyboxShader.setUniform("uProjection", m_projection);

    sky.bind(0);
    m_skyboxShader.setUniform("uSkybox", 0);

    m_skyboxCube.draw();
    glDepthFunc(GL_LESS);
    ++m_drawCallCount;
}

void Renderer::endFrame() {
    m_debugDraw.flush(m_projection * m_view);
}

}  // namespace ysq
