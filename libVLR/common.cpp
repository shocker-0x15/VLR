#include "shared/common_internal.h"

#if defined(VLR_Platform_Windows_MSVC)
VLR_CPP_API void vlrDevPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[4096];
    vsprintf_s(str, fmt, args);
    va_end(args);
    OutputDebugString(str);
}
#endif

VLR_CPP_API void vlrprintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[4096];
    vsprintf_s(str, fmt, args);
    va_end(args);

#   if defined(VLR_USE_DEVPRINTF) && defined(VLR_Platform_Windows_MSVC)
    vlrDevPrintf("%s", str);
#   endif
    printf("%s", str);
}

namespace VLR {
    // TODO: Make this function thread-safe.
    std::filesystem::path getExecutableDirectory() {
        static std::filesystem::path ret;

        static bool done = false;
        if (!done) {
#if defined(VLR_Platform_Windows_MSVC)
            TCHAR filepath[1024];
            auto length = GetModuleFileName(NULL, filepath, 1024);
            VLRAssert(length > 0, "Failed to query the executable path.");

            ret = filepath;
#else
            static_assert(false, "Not implemented");
#endif
            ret = ret.remove_filename();

            done = true;
        }

        return ret;
    }
}
