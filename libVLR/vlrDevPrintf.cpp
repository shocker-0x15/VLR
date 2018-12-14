#include "common.h"

#if defined(VLR_Platform_Windows_MSVC)
VLR_API void vlrDevPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[1024];
    vsprintf_s(str, fmt, args);
    va_end(args);
    OutputDebugString(str);
}
#endif
