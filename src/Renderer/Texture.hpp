#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ysq {

enum class TextureFormat { RGB8, RGBA8 };
enum class TextureFilter { Nearest, Linear };
enum class TextureWrap { Repeat, ClampToEdge };

struct TextureSettings {
    TextureFilter filter = TextureFilter::Linear;
    TextureWrap wrap = TextureWrap::Repeat;
    bool generateMipmaps = true;
};

/// A 2D GL texture, RAII, move-only. The context this was created under must
/// already be current for every method here, the same rule as Shader.
class Texture {
public:
    /// `pixels` is tightly packed, rows top-to-bottom, `width * height *
    /// channels(format)` bytes.
    [[nodiscard]] static std::optional<Texture>
    fromPixels(std::span<const std::uint8_t> pixels, int width, int height,
               TextureFormat format, const TextureSettings& settings = {});

    /// Decodes an image file (PNG, JPEG, and whatever else stb_image reads)
    /// from disk. nullopt if the path cannot be read or decoded; `error`, if
    /// given, is set to stb_image's own failure reason.
    [[nodiscard]] static std::optional<Texture>
    fromFile(std::string_view path, const TextureSettings& settings = {},
             std::string* error = nullptr);

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;
    ~Texture();

    void bind(unsigned unit = 0) const;

    [[nodiscard]] int width() const noexcept { return m_width; }
    [[nodiscard]] int height() const noexcept { return m_height; }
    [[nodiscard]] unsigned handle() const noexcept { return m_handle; }

private:
    Texture(unsigned handle, int width, int height) noexcept
        : m_handle(handle), m_width(width), m_height(height) {}
    void destroy() noexcept;

    unsigned m_handle = 0;
    int m_width = 0;
    int m_height = 0;
};

/// A six-face cubemap: skyboxes, and in RayTracer, the environment sampled by
/// rays that hit nothing in the scene.
class Cubemap {
public:
    /// Faces in OpenGL's own order: +X, -X, +Y, -Y, +Z, -Z.
    [[nodiscard]] static std::optional<Cubemap>
    fromFiles(const std::array<std::string, 6>& facePaths, std::string* error = nullptr);

    Cubemap(const Cubemap&) = delete;
    Cubemap& operator=(const Cubemap&) = delete;
    Cubemap(Cubemap&& other) noexcept;
    Cubemap& operator=(Cubemap&& other) noexcept;
    ~Cubemap();

    void bind(unsigned unit = 0) const;
    [[nodiscard]] unsigned handle() const noexcept { return m_handle; }

private:
    explicit Cubemap(unsigned handle) noexcept : m_handle(handle) {}
    void destroy() noexcept;

    unsigned m_handle = 0;
};

}  // namespace ysq
