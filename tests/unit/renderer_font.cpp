#include <Renderer/Font.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <vector>

/// Font's atlas/UV logic is pure data and arithmetic, no GL context needed;
/// tests/integration/renderer_text.cpp covers the part that actually draws.

TEST(RendererFont, AtlasPixelsHaveTheDeclaredSizeAndFormat) {
    const std::vector<std::uint8_t> pixels = ysq::font::buildAtlasPixels();
    const std::size_t expected = static_cast<std::size_t>(ysq::font::kAtlasWidth) *
                                 static_cast<std::size_t>(ysq::font::kAtlasHeight) * 4;
    EXPECT_EQ(pixels.size(), expected);
}

TEST(RendererFont, EveryPrintableCharacterHasAValidDistinctUV) {
    std::vector<ysq::font::GlyphUV> rects;
    for (int code = ysq::font::kFirstChar; code <= ysq::font::kLastChar; ++code) {
        const std::optional<ysq::font::GlyphUV> uv =
            ysq::font::glyphUV(static_cast<char>(code));
        ASSERT_TRUE(uv) << "code " << code;
        EXPECT_GE(uv->uvMin.x, 0.0f);
        EXPECT_GE(uv->uvMin.y, 0.0f);
        EXPECT_LE(uv->uvMax.x, 1.0f);
        EXPECT_LE(uv->uvMax.y, 1.0f);
        EXPECT_LT(uv->uvMin.x, uv->uvMax.x);
        EXPECT_LT(uv->uvMin.y, uv->uvMax.y);
        rects.push_back(*uv);
    }

    // No two glyphs share a cell: every rect's top-left corner is unique.
    for (std::size_t i = 0; i < rects.size(); ++i) {
        for (std::size_t j = i + 1; j < rects.size(); ++j) {
            const bool same = rects[i].uvMin.x == rects[j].uvMin.x &&
                              rects[i].uvMin.y == rects[j].uvMin.y;
            EXPECT_FALSE(same) << "glyphs " << i << " and " << j << " collide";
        }
    }
}

TEST(RendererFont, OutsidePrintableRangeIsNullopt) {
    EXPECT_FALSE(ysq::font::glyphUV('\x01'));
    EXPECT_FALSE(ysq::font::glyphUV(static_cast<char>(127)));
    EXPECT_FALSE(ysq::font::glyphUV(static_cast<char>(31)));
}

TEST(RendererFont, SpaceGlyphIsEntirelyTransparent) {
    const std::vector<std::uint8_t> pixels = ysq::font::buildAtlasPixels();
    const std::optional<ysq::font::GlyphUV> uv = ysq::font::glyphUV(' ');
    ASSERT_TRUE(uv);

    const int x0 =
        static_cast<int>(uv->uvMin.x * static_cast<float>(ysq::font::kAtlasWidth));
    const int y0 =
        static_cast<int>(uv->uvMin.y * static_cast<float>(ysq::font::kAtlasHeight));
    for (int y = y0; y < y0 + ysq::font::kGlyphHeight; ++y) {
        for (int x = x0; x < x0 + ysq::font::kGlyphWidth; ++x) {
            const std::size_t offset =
                (static_cast<std::size_t>(y) *
                     static_cast<std::size_t>(ysq::font::kAtlasWidth) +
                 static_cast<std::size_t>(x)) *
                4;
            EXPECT_EQ(pixels[offset + 3], 0);
        }
    }
}
