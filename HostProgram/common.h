#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstdarg>
#include <iostream>
#include <string>
#include <set>
#include <functional>
#include <sstream>
#include <fstream>

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define HP_Platform_Windows
#    if defined(_MSC_VER)
#        define HP_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define HP_Platform_macOS
#endif

#if defined(HP_Platform_Windows_MSVC)
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   if defined(VLR_API_EXPORTS)
#       define VLR_API __declspec(dllexport)
#   else
#       define VLR_API __declspec(dllimport)
#   endif
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#else
#   define VLR_API
#endif



#ifdef DEBUG
#   define ENABLE_ASSERT
#endif

#ifdef HP_Platform_Windows_MSVC
static void devPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[1024];
    vsprintf_s(str, fmt, args);
    va_end(args);
    OutputDebugString(str);
}
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#if 1
#   define hpprintf devPrintf
#else
#   define hpprintf printf
#endif

#ifdef ENABLE_ASSERT
#   define Assert(expr, fmt, ...) if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_ShouldNotBeCalled() Assert(false, "Should not be called!")
#define Assert_NotImplemented() Assert(false, "Not implemented yet!")

template <typename T, size_t size>
constexpr size_t lengthof(const T(&array)[size]) {
    return size;
}
