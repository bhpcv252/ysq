# Embeds a GLSL source file as a constexpr std::string_view at configure
# time. Renderer's shaders live as real .vert/.frag files under
# src/Renderer/shaders/ (editor syntax highlighting, no C++ string-literal
# noise as source of truth) while the compiled binary stays self-contained
# with no runtime filesystem dependency to resolve a shader path at run
# time. Same shape as Core/Version.hpp.in generating Core/Version.hpp: a
# real source file, expanded into a generated header at configure time.
#
# ysq_embed_shader(<source> <output_header> <variable_name>)
#   source         path to the .vert/.frag file
#   output_header  path to write the generated header to
#   variable_name  name of the generated constexpr std::string_view
function(ysq_embed_shader source output_header variable_name)
    file(READ ${source} _shader_source)

    # A raw string literal is the whole trick: GLSL needs no escaping this
    # way, only a delimiter guaranteed not to appear in the source. No real
    # shader contains "ysq_shader" in running text, so collision is not a
    # practical concern.
    string(CONCAT _header_content
        "// Generated from ${source} by ysq_embed_shader(). Edit that file, not this one.\n"
        "#pragma once\n\n"
        "#include <string_view>\n\n"
        "namespace ysq::shaders {\n\n"
        "inline constexpr std::string_view ${variable_name} = R\"ysq_shader(\n"
        "${_shader_source}"
        ")ysq_shader\";\n\n"
        "}  // namespace ysq::shaders\n")

    file(WRITE ${output_header} "${_header_content}")
endfunction()
