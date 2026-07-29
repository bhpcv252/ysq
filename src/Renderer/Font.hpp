#pragma once

#include <Math/Vector2.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace ysq {

/// A classic 8x8 monospace bitmap font, ASCII 32-126 (95 printable glyphs):
/// the widely-redistributed IBM PC/VGA ROM charset (CP437's printable ASCII
/// range), long-established public domain and reused verbatim across
/// countless open-source terminal emulators and font packs. Baked into a
/// small texture atlas at DebugDraw::create() time.
///
/// No font file, no TrueType parsing: short debug/label text is all this
/// engine's text needs, and a fixed bitmap this small is worth owning
/// outright rather than taking a dependency for. See DebugDraw::text().
namespace font {

inline constexpr int kGlyphWidth = 8;
inline constexpr int kGlyphHeight = 8;
inline constexpr int kFirstChar = 32;                           // ' '
inline constexpr int kLastChar = 126;                           // '~'
inline constexpr int kGlyphCount = kLastChar - kFirstChar + 1;  // 95

inline constexpr int kAtlasColumns = 16;
inline constexpr int kAtlasRows = 6;                             // 96 slots for 95 glyphs
inline constexpr int kAtlasWidth = kAtlasColumns * kGlyphWidth;  // 128
inline constexpr int kAtlasHeight = kAtlasRows * kGlyphHeight;   // 48

struct GlyphUV {
    Vec2f uvMin;
    Vec2f uvMax;
};

/// nullopt outside the printable ASCII range [kFirstChar, kLastChar].
[[nodiscard]] std::optional<GlyphUV> glyphUV(char c);

/// RGBA8, kAtlasWidth * kAtlasHeight * 4 bytes, row 0 at the top (matching
/// Texture::fromPixels' convention: tightly packed, rows top-to-bottom).
/// White RGB everywhere; alpha carries glyph coverage, so a shader
/// multiplies it by whatever color a label asks for.
[[nodiscard]] std::vector<std::uint8_t> buildAtlasPixels();

}  // namespace font

}  // namespace ysq
