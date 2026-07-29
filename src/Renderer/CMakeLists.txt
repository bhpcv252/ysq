# Renderer: camera, shaders, meshes, textures, and both a forward rasterizer
# and a fragment-shader ray tracer. Presentation layer, so ysq::warnings not
# _strict; see docs/architecture.md.

include(${CMAKE_SOURCE_DIR}/cmake/YsqEmbedShader.cmake)

set(_ysq_renderer_shader_dir ${YSQ_GENERATED_INCLUDE_DIR}/Renderer/shaders)

# One call per shader stage: source stays a real, editor-highlighted
# .vert/.frag file; the compiled binary embeds it at configure time with no
# runtime filesystem dependency. See cmake/YsqEmbedShader.cmake.
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/basic.vert
    ${_ysq_renderer_shader_dir}/Basic.vert.hpp kBasicVertSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/basic.frag
    ${_ysq_renderer_shader_dir}/Basic.frag.hpp kBasicFragSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/instanced.vert
    ${_ysq_renderer_shader_dir}/Instanced.vert.hpp kInstancedVertSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/debug.vert
    ${_ysq_renderer_shader_dir}/Debug.vert.hpp kDebugVertSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/debug.frag
    ${_ysq_renderer_shader_dir}/Debug.frag.hpp kDebugFragSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/skybox.vert
    ${_ysq_renderer_shader_dir}/Skybox.vert.hpp kSkyboxVertSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/skybox.frag
    ${_ysq_renderer_shader_dir}/Skybox.frag.hpp kSkyboxFragSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytrace.vert
    ${_ysq_renderer_shader_dir}/Raytrace.vert.hpp kRaytraceVertSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytrace.frag
    ${_ysq_renderer_shader_dir}/Raytrace.frag.hpp kRaytraceFragSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/text.vert
    ${_ysq_renderer_shader_dir}/Text.vert.hpp kTextVertSource)
ysq_embed_shader(${CMAKE_CURRENT_SOURCE_DIR}/shaders/text.frag
    ${_ysq_renderer_shader_dir}/Text.frag.hpp kTextFragSource)

add_library(ysq_renderer STATIC
    Camera.hpp
    Camera.cpp
    CameraController.hpp
    CameraController.cpp
    Light.hpp
    Material.hpp
    Shader.hpp
    Shader.cpp
    Mesh.hpp
    Mesh.cpp
    Font.hpp
    Font.cpp
    DebugDraw.hpp
    DebugDraw.cpp
    Texture.hpp
    Texture.cpp
    Renderer.hpp
    Renderer.cpp
    RayTracer.hpp
    RayTracer.cpp

    shaders/basic.vert
    shaders/basic.frag
    shaders/instanced.vert
    shaders/debug.vert
    shaders/debug.frag
    shaders/skybox.vert
    shaders/skybox.frag
    shaders/raytrace.vert
    shaders/raytrace.frag
    shaders/text.vert
    shaders/text.frag)
add_library(ysq::Renderer ALIAS ysq_renderer)

target_include_directories(ysq_renderer
    PUBLIC ${YSQ_SOURCE_INCLUDE_DIR}
    PRIVATE ${YSQ_GENERATED_INCLUDE_DIR})

# Math and Platform are PUBLIC: Camera/CameraController headers hand back
# Math types and CameraController::update() takes a Platform::InputState by
# reference, so a consumer of Renderer needs both. Core, glad and stb_image
# are implementation details, PRIVATE. See docs/architecture.md and
# src/Renderer/README.md.
target_link_libraries(ysq_renderer
    PUBLIC ysq::Math ysq::Platform
    PRIVATE ysq::Core glad stb_image ysq::warnings)
