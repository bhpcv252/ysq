# Embeds a GLSL source file as a constexpr std::string_view at configure
# time. Renderer's shaders live as real .vert/.frag files under
# src/Renderer/shaders/ (editor syntax highlighting, no C++ string-literal
# noise as source of truth) while the compiled binary stays self-contained
# with no runtime filesystem dependency to resolve a shader path at run
# time. A thin wrapper over the general-purpose YsqEmbedTextFile.cmake,
# fixed to the ysq::shaders namespace every shader is generated into.
#
# ysq_embed_shader(<source> <output_header> <variable_name>)
#   source         path to the .vert/.frag file
#   output_header  path to write the generated header to
#   variable_name  name of the generated constexpr std::string_view
include(${CMAKE_CURRENT_LIST_DIR}/YsqEmbedTextFile.cmake)

function(ysq_embed_shader source output_header variable_name)
    ysq_embed_text_file(${source} ${output_header} ${variable_name} ysq::shaders)
endfunction()
