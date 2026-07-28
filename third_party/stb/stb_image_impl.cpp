// stb_image ships as a header with an opt-in implementation, meant to be
// compiled exactly once. This is that one translation unit, so every
// consumer links the same stb_image target rather than each defining
// STB_IMAGE_IMPLEMENTATION itself and risking duplicate symbols.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
