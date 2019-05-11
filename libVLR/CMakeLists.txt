# ----------------------------------------------------------------
# JP: PTX生成用ターゲット
# EN: Target for PTX generation

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30 -cudart shared -rdc=true")

file(GLOB_RECURSE libVLR_PTX_Headers
     *.cuh)
file(GLOB_RECURSE libVLR_PTX_Sources
     *.cu)

add_library(VLR_PTX OBJECT ${libVLR_PTX_Headers} ${libVLR_PTX_Sources})
target_include_directories(VLR_PTX PUBLIC "\
${OptiX_SDK}/include;\
${CMAKE_CURRENT_SOURCE_DIR}/include/VLR\
")
target_compile_features(VLR_PTX PUBLIC cxx_std_11)
set_property(TARGET VLR_PTX PROPERTY CUDA_PTX_COMPILATION ON)

# END: Target for PTX generation
# ----------------------------------------------------------------



set(include_dirs "\
${OptiX_SDK}/include;\
${CMAKE_CURRENT_SOURCE_DIR}/include/VLR\
")
set(lib_dirs "${OptiX_SDK}/lib64")
set(libs "optix.6.0.0")

if(MSVC)
    add_definitions(-DVLR_API_EXPORTS)
endif()

file(GLOB_RECURSE libVLR_Sources
     *.h
     *.c
     *.hpp
     *.cpp)

source_group("" REGULAR_EXPRESSION 
             ".*\.(h|c|hpp|cpp)")
source_group("include" REGULAR_EXPRESSION 
             "include/VLR/.*\.(h|c|hpp|cpp)")
source_group("Shared" REGULAR_EXPRESSION 
             "shared/.*\.(h|c|hpp|cpp)")
source_group("GPU Kernels" REGULAR_EXPRESSION 
             "GPU_kernels/.*\.(cuh|cu)")
source_group("ext" REGULAR_EXPRESSION 
             "ext/include/.*\.(h|c|hpp|cpp)")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(VLR SHARED ${libVLR_Sources})
add_dependencies(VLR VLR_PTX)
target_include_directories(VLR PUBLIC ${include_dirs})
foreach(lib_dir ${lib_dirs})
    target_link_directories(VLR PRIVATE ${lib_dir})
endforeach()
foreach(lib ${libs})
    target_link_libraries(VLR PRIVATE ${lib})
endforeach()

set(PTXes_to_copy)
foreach(file ${libVLR_PTX_Sources})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/GPU_kernels/" "" ptx ${file})
    string(REPLACE ".cu" ".ptx" ptx ${ptx})
    set(PTXes_to_copy ${PTXes_to_copy} ${ptx})
endforeach()

# foreach(ptx ${PTXes_to_copy})
#     message(${ptx})
# endforeach()


# ----------------------------------------------------------------
# Post-build events

# copy ptxes
set(ptx_destination_dir "$<TARGET_FILE_DIR:VLR>/ptxes")
add_custom_command(TARGET VLR POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory
                   ${ptx_destination_dir})
add_custom_command(TARGET VLR POST_BUILD COMMAND ${CMAKE_COMMAND} -E chdir
                   "${CMAKE_CURRENT_BINARY_DIR}/VLR_PTX.dir/${CMAKE_CFG_INTDIR}" 
                   ${CMAKE_COMMAND} -E 
                   copy ${PTXes_to_copy} ${ptx_destination_dir})

# copy resouces
add_custom_command(TARGET VLR POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${CMAKE_CURRENT_SOURCE_DIR}/spectral_upsampling_tables" "$<TARGET_FILE_DIR:VLR>/spectral_upsampling_tables")

# copy dlls
if(MSVC)
    add_custom_command(TARGET VLR POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                       "${OptiX_SDK}/bin64/optix.6.0.0.dll" "$<TARGET_FILE_DIR:VLR>")
endif()

# END: Post-build events
# ----------------------------------------------------------------

install(TARGETS VLR CONFIGURATIONS Debug DESTINATION "${CMAKE_BINARY_DIR}/bin/Debug")
install(TARGETS VLR CONFIGURATIONS Release DESTINATION "${CMAKE_BINARY_DIR}/bin/Release")
