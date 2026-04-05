#pragma once
#define __mem_alloc_h

#if defined WIN32 || defined _WIN32 || defined __WIN32__
#include <stdlib.h>
#include <intrin.h>
#include <malloc.h>
//#include <windows.h>
#endif

#include <stddef.h>
#include <stdlib.h>
#ifdef __linux__
#include <malloc.h>
#endif
#include "pll.hpp"

#if defined WIN32 || defined _WIN32 || defined __WIN32__
    #if (defined(__MINGW32__) || defined(__clang__)) && defined(BINARY32)
        #define rax_posix_memalign(ptr,alignment,size) *(ptr) = __mingw_aligned_malloc((size),(alignment))
        #define rax_malloc(size) __mingw_aligned_malloc((size), PLL_BYTE_ALIGNMENT)
        void *rax_calloc(size_t count, size_t size);
        #define rax_free __mingw_aligned_free
    #else
        #define rax_posix_memalign(ptr,alignment,size) *(ptr) = _aligned_malloc((size),(alignment))
        #define rax_malloc(size) _aligned_malloc((size), PLL_BYTE_ALIGNMENT)
        void *rax_calloc(size_t count, size_t size);
        #define rax_free _aligned_free
    #endif
#elif (defined(__SSE3) || defined(__AVX))
    #define rax_posix_memalign posix_memalign
    #define rax_malloc malloc
    #define rax_calloc calloc
    #define rax_free free
#else
    void *rax_memalign(size_t alignment, size_t size);
    void *rax_malloc(size_t size);
    void *rax_realloc(void *p, size_t size);
    void *rax_calloc(size_t n, size_t size);
    void  rax_free(void *p);
    int   rax_posix_memalign(void **p, size_t alignment, size_t size);
    void *rax_malloc_aligned(size_t size);
#endif