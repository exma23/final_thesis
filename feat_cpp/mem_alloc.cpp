#include "mem_alloc.hpp"

#include <malloc.h>
#include <stdlib.h>

void *rax_memalign(size_t align, size_t size) { return memalign(align, size); }

void *rax_malloc(size_t size) { return malloc(size); }

void *rax_realloc(void *p, size_t size) { return realloc(p, size); }

void *rax_calloc(size_t n, size_t size) { return calloc(n, size); }

void rax_free(void *p) { free(p); }

int rax_posix_memalign(void **p, size_t align, size_t size) {
  return posix_memalign(p, align, size);
}

void *rax_malloc_aligned(size_t size) { return rax_memalign(32, size); }