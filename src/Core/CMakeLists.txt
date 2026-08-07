configure_file(Version.hpp.in ${YSQ_GENERATED_INCLUDE_DIR}/Core/Version.hpp @ONLY)

add_library(ysq_core STATIC
    Clock.cpp
    Config.cpp
    Csv.cpp
    Event.cpp
    Logger.cpp
    UUID.cpp
    Version.cpp
    Clock.hpp
    Config.hpp
    Csv.hpp
    Event.hpp
    Logger.hpp
    Timer.hpp
    UUID.hpp
    ${YSQ_GENERATED_INCLUDE_DIR}/Core/Version.hpp)
add_library(ysq::Core ALIAS ysq_core)

target_include_directories(ysq_core PUBLIC
    ${YSQ_SOURCE_INCLUDE_DIR}
    ${YSQ_GENERATED_INCLUDE_DIR})

# spdlog is PRIVATE: Logger.hpp never names it, so consumers link it without
# compiling its headers and without depending on the backend staying spdlog.
target_link_libraries(ysq_core PRIVATE ysq::warnings spdlog::spdlog)
