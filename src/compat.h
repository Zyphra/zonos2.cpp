// Cross-platform shims for POSIX file calls used by the GGUF loaders.
// MSVC's CRT lacks fseeko/ftello/popen; map them to the _-prefixed equivalents.
// _fseeki64/_ftelli64 take/return 64-bit offsets -- required for the multi-GB
// GGUFs (a plain fseek would truncate on Win32, where long is 32-bit).
#pragma once

#ifdef _WIN32
#include <stdio.h>
#define fseeko(stream, offset, whence) _fseeki64((stream), (long long) (offset), (whence))
#define ftello(stream)                 _ftelli64(stream)
#define popen                          _popen
#define pclose                         _pclose
#endif
