set(TARGET_NAME "libvlr")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

set(
    GL3W_SOURCES
    "${CMAKE_BINARY_DIR}/ext/gl3w/include/GL/gl3w.h"
    "${CMAKE_BINARY_DIR}/ext/gl3w/include/GL/glcorearb.h"
    "${CMAKE_BINARY_DIR}/ext/gl3w/include/KHR/khplatform.h"
    "${CMAKE_BINARY_DIR}/ext/gl3w/src/gl3w.c"
)
set_source_files_properties(
    ${GL3W_SOURCES} PROPERTIES
    GENERATED TRUE
)

set(
    CUDA_KERNELS
    "GPU_kernels/aux_buffer_generator.cu"
    "GPU_kernels/post_process.cu"
    "GPU_kernels/setup_scene.cu"
)

set(
    OPTIX_KERNELS
    "GPU_kernels/materials.cu"
    "GPU_kernels/shader_nodes.cu"
    "GPU_kernels/triangle.cu"
    "GPU_kernels/point.cu"
    "GPU_kernels/infinite_sphere.cu"
    "GPU_kernels/cameras.cu"
    "GPU_kernels/debug_rendering.cu"
    "GPU_kernels/path_tracing.cu"
    "GPU_kernels/light_tracing.cu"
    "GPU_kernels/lvc_bpt.cu"
)

set(
    GPU_KERNEL_DEPENDENCIES
    "utils/cuda_util.h"
    "utils/optix_util.h"
    "utils/optixu_on_cudau.h"
    "include/vlr/common.h"
    "include/vlr/basic_types.h"
    "shared/common_internal.h"
    "shared/basic_types_internal.h"
    "shared/spectrum_base.h"
    "shared/rgb_spectrum_types.h"
    "shared/spectrum_types.h"
    "shared/random_distributions.h"
    "shared/shared.h"
    "shared/kernel_common.h"
    "shared/renderer_common.h"
    "shared/light_transport_common.h"
)

nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819 /Zc:__cplusplus\""
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-G>"
    "--gpu-architecture=compute_52"
    "-std=c++20"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
    "-I\"include/vlr\""
)

nvcuda_compile_ptx(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819 /Zc:__cplusplus\""
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-lineinfo>"
    "--gpu-architecture=compute_52"
    "-std=c++20"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
    "-I\"include/vlr\""
)



source_group(
    "Host" REGULAR_EXPRESSION 
    ".*\.(h|c|hpp|cpp)")
source_group(
    "include" REGULAR_EXPRESSION 
    "include/vlr/.*\.(h|c|hpp|cpp)")
source_group(
    "Shared" REGULAR_EXPRESSION 
    "shared/.*\.(h|c|hpp|cpp)")
source_group(
    "GPU Kernels" REGULAR_EXPRESSION 
    "GPU_kernels/.*\.(cuh|cu)")
source_group(
    "Utilities" REGULAR_EXPRESSION 
    "utils/.*\.(h|c|hpp|cpp)")
source_group(
    "ext/gl3w" REGULAR_EXPRESSION
    "../ext/gl3w/.*$")


file(
    GLOB_RECURSE SPECTRAL_UPSAMPLING_TABLES
    "spectral_upsampling_tables/*"
)

copy_files(
    SOURCES ${SPECTRAL_UPSAMPLING_TABLES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/spectral_upsampling_tables"
    GENERATED_FILES COPIED_SPECTRAL_UPSAMPLING_TABLES
)



add_library(
    "${TARGET_NAME}" SHARED
    ${SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
    ${SPECTRAL_UPSAMPLING_TABLES}
)
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
    "VLR_API_EXPORTS"
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
    "include/vlr"
    "${CMAKE_BINARY_DIR}/ext/gl3w/include"
    "${CMAKE_SOURCE_DIR}/ext/half"
    "${OPTIX_INCLUDE_DIR}"
)
target_link_libraries(
    "${TARGET_NAME}" PRIVATE
    CUDA::cuda_driver
    gl3w
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)
