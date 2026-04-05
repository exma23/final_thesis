#pragma once
#include "pll.hpp"

pllHashTable * pllHashInit (unsigned int n);

unsigned int pllHashString (const char * s, unsigned int size);
int pllHashAdd(struct pllHashTable * hTable, unsigned int hash, const char * s, void * item);
int pllHashSearch (struct pllHashTable * hTable, char * s, void ** item);
void pllHashDestroy (struct pllHashTable ** hTable, int);