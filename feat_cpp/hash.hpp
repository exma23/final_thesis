#pragma once
#include "pll.hpp"

// cấp phát bảng băm chọn size là số nguyên tố
pllHashTable *pllHashInit(unsigned int n);

// Biến string → index
unsigned int pllHashString(const char *s, unsigned int size);

// Thêm phần tử vào hash
int pllHashAdd(struct pllHashTable *hTable, unsigned int hash, const char *s,
               void *item);
int pllHashSearch(struct pllHashTable *hTable, char *s, void **item);
void pllHashDestroy(struct pllHashTable **hTable, int);