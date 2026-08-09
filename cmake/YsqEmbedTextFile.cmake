# Embeds a text file as a constexpr std::string_view at configure time, so a
# real source file (a shader, a data table) stays a real file on disk --
# editor syntax highlighting, no C++ string-literal noise as source of truth
# -- while the compiled binary stays self-contained with no runtime
# filesystem dependency to resolve its path at run time. Same shape as
# Core/Version.hpp.in generating Core/Version.hpp: a real source file,
# expanded into a generated header at configure time.
#
# ysq_embed_text_file(<source> <output_header> <variable_name> <namespace>)
#   source         path to the text file to embed
#   output_header  path to write the generated header to
#   variable_name  name of the generated constexpr std::string_view
#   namespace      namespace (e.g. ysq::shaders) the variable is declared in
function(ysq_embed_text_file source output_header variable_name namespace)
    file(READ ${source} _file_content)

    # A raw string literal is the whole trick: the source needs no escaping
    # this way, only a delimiter guaranteed not to appear in it. No real
    # shader or data file contains "ysq_embed" in running text, so collision
    # is not a practical concern.
    string(CONCAT _header_content
        "// Generated from ${source}. Edit that file, not this one.\n"
        "#pragma once\n\n"
        "#include <string_view>\n\n"
        "namespace ${namespace} {\n\n"
        "inline constexpr std::string_view ${variable_name} = R\"ysq_embed(\n"
        "${_file_content}"
        ")ysq_embed\";\n\n"
        "}  // namespace ${namespace}\n")

    file(WRITE ${output_header} "${_header_content}")
endfunction()
