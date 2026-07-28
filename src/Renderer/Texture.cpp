#include <Renderer/Texture.hpp>

#include <glad/gl.h>
#include <stb_image.h>

#include <utility>

namespace ysq {

namespace {

int channelCount(TextureFormat format) {
    return format == TextureFormat::RGBA8 ? 4 : 3;
}

unsigned glFormat(TextureFormat format) {
    return format == TextureFormat::RGBA8 ? GL_RGBA : GL_RGB;
}

int glFilter(TextureFilter filter, bool mipmapped) {
    if (filter == TextureFilter::Nearest) {
        return mipmapped ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST;
    }
    return mipmapped ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR;
}

int glWrap(TextureWrap wrap) {
    return wrap == TextureWrap::ClampToEdge ? GL_CLAMP_TO_EDGE : GL_REPEAT;
}

void applySettings(unsigned target, const TextureSettings& settings) {
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER,
                    glFilter(settings.filter, settings.generateMipmaps));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, glFilter(settings.filter, false));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, glWrap(settings.wrap));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, glWrap(settings.wrap));
}

}  // namespace

std::optional<Texture> Texture::fromPixels(std::span<const std::uint8_t> pixels,
                                           int width, int height, TextureFormat format,
                                           const TextureSettings& settings) {
    if (width <= 0 || height <= 0) {
        return std::nullopt;
    }
    const auto expectedBytes = static_cast<std::size_t>(width) *
                               static_cast<std::size_t>(height) *
                               static_cast<std::size_t>(channelCount(format));
    if (pixels.size() < expectedBytes) {
        return std::nullopt;
    }

    unsigned handle = 0;
    glGenTextures(1, &handle);
    glBindTexture(GL_TEXTURE_2D, handle);
    const unsigned format2D = glFormat(format);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(format2D), width, height, 0,
                 format2D, GL_UNSIGNED_BYTE, pixels.data());
    if (settings.generateMipmaps) {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    applySettings(GL_TEXTURE_2D, settings);
    glBindTexture(GL_TEXTURE_2D, 0);

    return std::optional<Texture>{Texture{handle, width, height}};
}

std::optional<Texture> Texture::fromFile(std::string_view path,
                                         const TextureSettings& settings,
                                         std::string* error) {
    int width = 0;
    int height = 0;
    int channels = 0;
    // stb_image decodes row 0 as the image's top row; OpenGL's texture
    // convention treats row 0 of the uploaded buffer as texel row V = 0, the
    // bottom. Without this flip every loaded texture comes out vertically
    // mirrored against standard OpenGL UVs.
    stbi_set_flip_vertically_on_load(1);
    // Forced to 4 channels: TextureFormat::RGBA8 always matches what
    // stbi_load actually wrote, regardless of the source file's own channel
    // count, so there is no format mismatch to reconcile below.
    unsigned char* decoded =
        stbi_load(std::string{path}.c_str(), &width, &height, &channels, 4);
    if (decoded == nullptr) {
        if (error) {
            *error = stbi_failure_reason();
        }
        return std::nullopt;
    }

    const std::span<const std::uint8_t> pixels{
        reinterpret_cast<const std::uint8_t*>(decoded),
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4};
    std::optional<Texture> texture =
        fromPixels(pixels, width, height, TextureFormat::RGBA8, settings);
    stbi_image_free(decoded);
    return texture;
}

Texture::Texture(Texture&& other) noexcept
    : m_handle(std::exchange(other.m_handle, 0u)),
      m_width(std::exchange(other.m_width, 0)),
      m_height(std::exchange(other.m_height, 0)) {}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();
    m_handle = std::exchange(other.m_handle, 0u);
    m_width = std::exchange(other.m_width, 0);
    m_height = std::exchange(other.m_height, 0);
    return *this;
}

Texture::~Texture() {
    destroy();
}

void Texture::destroy() noexcept {
    if (m_handle != 0) {
        glDeleteTextures(1, &m_handle);
        m_handle = 0;
    }
}

void Texture::bind(unsigned unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, m_handle);
}

std::optional<Cubemap> Cubemap::fromFiles(const std::array<std::string, 6>& facePaths,
                                          std::string* error) {
    unsigned handle = 0;
    glGenTextures(1, &handle);
    glBindTexture(GL_TEXTURE_CUBE_MAP, handle);

    // Same row-order mismatch as Texture::fromFile, and the same fix: each
    // cube face is still an ordinary 2D image at the texel level, decoded
    // top row first by stb_image but expected bottom row first by OpenGL.
    stbi_set_flip_vertically_on_load(1);
    for (std::size_t face = 0; face < facePaths.size(); ++face) {
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* decoded =
            stbi_load(facePaths[face].c_str(), &width, &height, &channels, 4);
        if (decoded == nullptr) {
            if (error) {
                *error = facePaths[face] + ": " + stbi_failure_reason();
            }
            glDeleteTextures(1, &handle);
            return std::nullopt;
        }
        glTexImage2D(static_cast<unsigned>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face), 0,
                     GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, decoded);
        stbi_image_free(decoded);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    return std::optional<Cubemap>{Cubemap{handle}};
}

Cubemap::Cubemap(Cubemap&& other) noexcept
    : m_handle(std::exchange(other.m_handle, 0u)) {}

Cubemap& Cubemap::operator=(Cubemap&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();
    m_handle = std::exchange(other.m_handle, 0u);
    return *this;
}

Cubemap::~Cubemap() {
    destroy();
}

void Cubemap::destroy() noexcept {
    if (m_handle != 0) {
        glDeleteTextures(1, &m_handle);
        m_handle = 0;
    }
}

void Cubemap::bind(unsigned unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_handle);
}

}  // namespace ysq
