# Current CMakeLists has been tested only on Windows/MSVC.
# Manual copying executables and other files are required for other platforms.

cmake_minimum_required(VERSION 3.14)
set(CMAKE_CONFIGURATION_TYPES Debug Release)
project(VLR)

if(NOT MSVC)
    option(USE_LIBCPP "Use libc++ instead of libstdc++." ON)
endif()

# macro (set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE)
# set_property (TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY}
#     ${XCODE_VALUE})
# endmacro (set_xcode_property)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    set(CMAKE_C_FLAGS_DEBUG "/Od /DDEBUG /ZI /MP")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /DDEBUG /ZI /MP")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG /Zi /MP")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /Zi /MP")

    set_property(GLOBAL APPEND PROPERTY LINK_FLAGS_DEBUG /DEBUG:FASTLINK)
    set_property(GLOBAL APPEND PROPERTY LINK_FLAGS_RELEASE /DEBUG:FASTLINK)
else()
    set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

set(Assimp_include "/usr/local/include/" CACHE PATH "Assimp include directory")
set(Assimp_lib "/usr/local/lib/" CACHE PATH "Assimp library directory")
set(Assimp_bin "/usr/local/bin/" CACHE PATH "Assimp binary directory")
set(Imath_include "/usr/local/include/Imath" CACHE PATH "Imath 3.1 include directory")
set(OpenEXR_include "/usr/local/include/OpenEXR" CACHE PATH "OpenEXR 3.1 include directory")
set(OpenEXR_lib "/usr/local/lib/" CACHE PATH "OpenEXR library directory")
set(OpenEXR_bin "/usr/local/bin/" CACHE PATH "OpenEXR binary directory")
if(MSVC)
    set(OptiX_SDK "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0" CACHE PATH "OptiX SDK directory")
else()
    set(OptiX_SDK "NVIDIA Corporation/OptiX SDK 7.4.0" CACHE PATH "OptiX SDK directory")
endif()

if(MSVC)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif()

# OS Xにおけるrun path処理の有効化
set(CMAKE_MACOSX_RPATH 1)

# 各プロジェクトのCMakeLists.txtを呼び出す。
add_subdirectory(libVLR)
add_subdirectory(HostProgram)

# ビルド依存関係を設定
add_dependencies(HostProgram VLR)
