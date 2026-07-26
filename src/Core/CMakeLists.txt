configure_file(Version.hpp.in ${YSQ_GENERATED_INCLUDE_DIR}/Core/Version.hpp @ONLY)

add_library(ysq_core STATIC
    Version.cpp
    ${YSQ_GENERATED_INCLUDE_DIR}/Core/Version.hpp)
add_library(ysq::Core ALIAS ysq_core)

target_include_directories(ysq_core PUBLIC
    ${YSQ_SOURCE_INCLUDE_DIR}
    ${YSQ_GENERATED_INCLUDE_DIR})

target_link_libraries(ysq_core PRIVATE ysq::warnings)
