set(TARGET_NAME "HostProgram")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)
file(
    GLOB_RECURSE SHADERS
    *.vert *.frag)

source_group(
    "" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "shaders" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/[^/]*\.(vert|frag)$")

file(
    GLOB_RECURSE EXT_SOURCES
    "../ext/glfw/include/*"
    "../ext/imgui/*"
    "../ext/tinyexr/*"
    "../ext/stb/stb_image.h"
    "../ext/stb/stb_image_write.h"
)
list(
    FILTER EXT_SOURCES INCLUDE REGEX
    ".*\.(h|hpp|cuh|c|cc|cpp|cu|vert|frag)$")
list(
    FILTER EXT_SOURCES EXCLUDE REGEX
    "../ext/prebuilt/.*")
list(
    FILTER EXT_SOURCES EXCLUDE REGEX
    "../ext/imgui/imgui_demo\.cpp$")
list(
    FILTER EXT_SOURCES EXCLUDE REGEX
    "../ext/imgui/(backends|misc|examples)/.*")
list(
    FILTER EXT_SOURCES EXCLUDE REGEX
    "../ext/tinyexr/deps/miniz/examples/.*")
list(
    FILTER EXT_SOURCES EXCLUDE REGEX
    "../ext/tinyexr/deps/ZFP/.*")
list(
    FILTER EXT_SOURCES EXCLUDE REGEX
    "../ext/tinyexr/(examples|experimental|test)/.*")
list(
    FILTER EXT_SOURCES EXCLUDE REGEX
    "../ext/tinyexr/test_tinyexr.cc$")
file(
    GLOB IMGUI_BACKEND_SOURCES
    "../ext/imgui/backends/imgui_impl_glfw*"
    "../ext/imgui/backends/imgui_impl_opengl3*"
)
set(
    EXT_SOURCES
    ${EXT_SOURCES}
    ${IMGUI_BACKEND_SOURCES}
)

source_group(
    "ext/gl3w" REGULAR_EXPRESSION
    "../ext/gl3w/.*$")
source_group(
    "ext/glfw" REGULAR_EXPRESSION
    "../ext/glfw/.*\.h$")
source_group(
    "ext/imgui/core" REGULAR_EXPRESSION
    "../ext/imgui/[^/]*\.(h|cpp)$")
source_group(
    "ext/imgui" REGULAR_EXPRESSION
    "../ext/imgui/backends/.*\.(h|cpp)$")
source_group(
    "ext/tinyexr" REGULAR_EXPRESSION
    "../ext/tinyexr/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "ext/tinyexr/miniz" REGULAR_EXPRESSION
    "../ext/tinyexr/deps/miniz/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "ext/tinyexr/nanozlib" REGULAR_EXPRESSION
    "../ext/tinyexr/deps/nanozlib/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "ext/stb" REGULAR_EXPRESSION
    "../ext/stb/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "ext" REGULAR_EXPRESSION
    "../ext/[^/]*\.(h|hpp|c|cc|cpp)$")



copy_files(
    SOURCES ${SHADERS}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/shaders"
    GENERATED_FILES COPIED_SHADERS
)



add_executable(
    "${TARGET_NAME}"
    ${EXT_SOURCES}
    ${SOURCES}
    ${SHADERS}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
)
target_compile_options(
    "${TARGET_NAME}" PRIVATE
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/MP);
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/Zc:__cplusplus);
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/Zc:__cplusplus>"
)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "../ext/tinyexr"
    "../ext/tinyexr/deps/miniz"
    "../ext/glfw/include"
    "../ext/imgui"
    "../ext/assimp/include"
    "../ext/stb"
    "${CMAKE_BINARY_DIR}/ext/assimp/include"
    "${CMAKE_BINARY_DIR}/ext/gl3w/include"
    "${CMAKE_SOURCE_DIR}/libVLR/include"
)
target_link_libraries(
    "${TARGET_NAME}"
    CUDA::cuda_driver
    opengl32
    gl3w
    glfw
    assimp::assimp
    libvlr
)
