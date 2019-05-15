set(include_dirs "\
${OpenEXR_include};\
${Assimp_include};\
${CMAKE_SOURCE_DIR}/libVLR/include;\
ext/include;\
ext/include/imGui\
")
set(lib_dirs "\
${OpenEXR_lib};\
${Assimp_lib};\
${CMAKE_CURRENT_SOURCE_DIR}/ext/lib\
")
set(libs "\
VLR;\
Half;Iex-2_2;IexMath-2_2;IlmImf-2_2;IlmImfUtil-2_2;IlmThread-2_2;Imath-2_2;zlib;\
assimp-vc140-mt;\
glfw3;opengl32\
")

if(MSVC)
    add_definitions(-DOPENEXR_DLL)
endif()

file(GLOB_RECURSE HostProgram_Sources
     *.h
     *.c
     *.cpp
     *.vert
     *.frag)

source_group("" REGULAR_EXPRESSION 
             ".*\.(h|c|hpp|cpp)")
source_group("gl3w" REGULAR_EXPRESSION 
             "ext/(include/GL|src/gl3w)/.*\.(h|c|hpp|cpp)")
source_group("shaders" REGULAR_EXPRESSION 
             "shaders/.*\.(vert|frag)")
source_group("imGui" REGULAR_EXPRESSION 
             "ext/(include/imGui|src/imGui)/.*\.(h|c|hpp|cpp)")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(HostProgram ${HostProgram_Sources})
target_include_directories(HostProgram PUBLIC ${include_dirs})
foreach(lib_dir ${lib_dirs})
    target_link_directories(HostProgram PRIVATE ${lib_dir})
endforeach()
foreach(lib ${libs})
    target_link_libraries(HostProgram PRIVATE ${lib})
endforeach()

# ----------------------------------------------------------------
# Post-build events

# copy shaders
add_custom_command(TARGET HostProgram POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${CMAKE_CURRENT_SOURCE_DIR}/shaders" "$<TARGET_FILE_DIR:HostProgram>/shaders")

# copy dlls
if(MSVC)
    add_custom_command(TARGET HostProgram POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                       "${Assimp_bin}/assimp-vc140-mt.dll" "$<TARGET_FILE_DIR:HostProgram>")
    add_custom_command(TARGET HostProgram POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                       "${OpenEXR_bin}/Half.dll"
                       "${OpenEXR_bin}/Iex-2_2.dll"
                       "${OpenEXR_bin}/IexMath-2_2.dll"
                       "${OpenEXR_bin}/IlmImf-2_2.dll"
                       "${OpenEXR_bin}/IlmImfUtil-2_2.dll"
                       "${OpenEXR_bin}/IlmThread-2_2.dll"
                       "${OpenEXR_bin}/Imath-2_2.dll"
                       "$<TARGET_FILE_DIR:HostProgram>")
    add_custom_command(TARGET HostProgram POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                       "${zlib_bin}/zlib.dll" "$<TARGET_FILE_DIR:HostProgram>")
endif()

# END: Post-build events
# ----------------------------------------------------------------

set_target_properties(HostProgram PROPERTIES INSTALL_RPATH "@executable_path")
install(TARGETS HostProgram CONFIGURATIONS Debug DESTINATION "${CMAKE_BINARY_DIR}/bin/Debug")
install(TARGETS HostProgram CONFIGURATIONS Release DESTINATION "${CMAKE_BINARY_DIR}/bin/Release")
