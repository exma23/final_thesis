#include "mem_alloc.hpp"
#include "pll.hpp"
#include <stdio.h>
#include <string.h>

static const unsigned int initTable[] = {
    53,        97,         193,        389,       769,       1543,
    3079,      6151,       12289,      24593,     49157,     98317,
    196613,    393241,     786433,     1572869,   3145739,   6291469,
    12582917,  25165843,   50331653,   100663319, 201326611, 402653189,
    805306457, 1610612741, 3221225473, 4294967291};

pllHashTable *pllHashInit(unsigned int n) {
  pllHashTable *hTable;
  unsigned int i;
  unsigned int primeTableLength;

  hTable = (pllHashTable *)rax_malloc(sizeof(pllHashTable));
  if (!hTable)
    return (NULL);

  primeTableLength = sizeof(initTable) / sizeof(initTable[0]);

  i = 0;

  while (i < primeTableLength && initTable[i] < n)
    ++i;

  n = initTable[i];

  hTable->Items = (pllHashItem **)rax_calloc(n, sizeof(pllHashItem *));
  if (!hTable->Items) {
    rax_free(hTable);
    return (NULL);
  }
  hTable->size = n;
  hTable->entries = 0;

  return (hTable);
}

unsigned int pllHashString(const char *s, unsigned int size) {
  unsigned int hash = 0;

  for (; *s; ++s)
    hash = (hash << 5) - hash + (unsigned int)*s;

  return (hash % size);
}

int pllHashSearch(pllHashTable *hTable, char *s, void **item) {
  unsigned int pos;
  pllHashItem *hItem;

  if (!hTable || !s || !item)
    return (PLL_FALSE);

  pos = pllHashString(s, hTable->size);
  hItem = hTable->Items[pos];

  for (; hItem; hItem = hItem->next) {
    if (hItem->str && !strcmp(s, hItem->str)) {
      *item = hItem->data;
      return (PLL_TRUE);
    }
  }

  return (PLL_FALSE);
}

int pllHashAdd(pllHashTable *hTable, unsigned int hash, const char *s,
               void *item) {
  pllHashItem *hItem;

  if (!hTable) {
    printf("in pllHashAdd, hTable is null");
    return (PLL_FALSE);
  }

  hItem = hTable->Items[hash];

  /* If a string was given, check whether the record already exists */
  if (s) {
    for (; hItem; hItem = hItem->next) {
      if (hItem->str && !strcmp(s, hItem->str))
        return (PLL_FALSE);
    }
  }

  hItem = (pllHashItem *)rax_malloc(sizeof(pllHashItem));

  /* store the string together with the element if given */
  if (s) {
    hItem->str = (char *)rax_malloc((strlen(s) + 1) * sizeof(char));
    if (!hItem->str) {
      rax_free(hItem);
      return (PLL_FALSE);
    }
    strcpy(hItem->str, s);
  } else
    hItem->str = NULL;

  hItem->data = item;

  hItem->next = hTable->Items[hash];
  hTable->Items[hash] = hItem;
  hTable->entries += 1;

  return (PLL_TRUE);
}

void pllHashDestroy(pllHashTable **hTable, int cbDealloc) {
  unsigned int i;
  pllHashItem *hItem;
  pllHashItem *tmp;

  (void)cbDealloc;

  for (i = 0; i < (*hTable)->size; ++i) {
    hItem = (*hTable)->Items[i];
    while (hItem) {
      tmp = hItem;
      hItem = hItem->next;
      if (tmp->str)
        rax_free(tmp->str);
      rax_free(tmp);
    }
  }
  rax_free((*hTable)->Items);
  rax_free(*hTable);
  *hTable = NULL;
}