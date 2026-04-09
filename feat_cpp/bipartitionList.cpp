#include "globalVariables.hpp"
#include "hash.hpp"
#include "mem_alloc.hpp"
#include "pll.hpp"
#include "pllInternal.hpp"
#include <assert.h>
#include <cstring>

static void getxnodeBips(nodeptr p) {
  nodeptr s;

  if ((s = p->next)->xBips || (s = s->next)->xBips) {
    p->xBips = s->xBips;
    s->xBips = 0;
  }

  assert(p->xBips);
}

static pllBipartitionEntry *initEntry(void) {
  pllBipartitionEntry *e =
      (pllBipartitionEntry *)rax_malloc(sizeof(pllBipartitionEntry));

  e->bitVector = (unsigned int *)NULL;
  e->treeVector = (unsigned int *)NULL;
  e->supportVector = (int *)NULL;
  e->bipNumber = 0;
  e->bipNumber2 = 0;
  e->supportFromTreeset[0] = 0;
  e->supportFromTreeset[1] = 0;
  e->next = (pllBipartitionEntry *)NULL;

  return e;
}

static void insertHashRF(unsigned int *bitVector, pllHashTable *h,
                         unsigned int vectorLength, int treeNumber,
                         int treeVectorLength, hashNumberType position,
                         int support, pllBoolean computeWRF) {
  pllBipartitionEntry *e;
  pllHashItem *hitem;

  if (h->Items[position] != NULL) {
    for (hitem = h->Items[position]; hitem; hitem = hitem->next) {
      e = (pllBipartitionEntry *)(hitem->data);

      if (!memcmp(bitVector, e->bitVector,
                  vectorLength * sizeof(unsigned int))) {
        e->treeVector[treeNumber / PLL_MASK_LENGTH] |=
            mask32[treeNumber % PLL_MASK_LENGTH];
        if (computeWRF) {
          e->supportVector[treeNumber] = support;
          assert(0 <= treeNumber &&
                 treeNumber < treeVectorLength * PLL_MASK_LENGTH);
        }
        return;
      }
    }
  }
  e = initEntry();

  e->bitVector =
      (unsigned int *)rax_calloc((size_t)vectorLength, sizeof(unsigned int));
  e->treeVector = (unsigned int *)rax_calloc((size_t)treeVectorLength,
                                             sizeof(unsigned int));
  if (computeWRF)
    e->supportVector = (int *)rax_calloc(
        (size_t)treeVectorLength * PLL_MASK_LENGTH, sizeof(int));

  e->treeVector[treeNumber / PLL_MASK_LENGTH] |=
      mask32[treeNumber % PLL_MASK_LENGTH];
  if (computeWRF) {
    e->supportVector[treeNumber] = support;

    assert(0 <= treeNumber && treeNumber < treeVectorLength * PLL_MASK_LENGTH);
  }

  memcpy(e->bitVector, bitVector, sizeof(unsigned int) * vectorLength);

  pllHashAdd(h, position, NULL, (void *)e);
}

static void newviewBipartitions(unsigned int **bitVectors, nodeptr p, int numsp,
                                unsigned int vectorLength, int processID) {
  if (isTip(p->number, numsp))
    return;

  nodeptr q = p->next->back;
  nodeptr r = p->next->next->back;
  if (!q || !r) {
    while (!p->xBips)
      getxnodeBips(p);
    return;
  }
  {
    nodeptr q = p->next->back, r = p->next->next->back;

    unsigned int *vector = bitVectors[p->number], *left = bitVectors[q->number],
                 *right = bitVectors[r->number];
    unsigned int i;

    assert(processID == 0);

    while (!p->xBips) {
      if (!p->xBips)
        getxnodeBips(p);
    }

    p->hash = q->hash ^ r->hash;

    if (isTip(q->number, numsp) && isTip(r->number, numsp)) {
      for (i = 0; i < vectorLength; i++)
        vector[i] = left[i] | right[i];
    } else {
      if (isTip(q->number, numsp) || isTip(r->number, numsp)) {
        if (isTip(r->number, numsp)) {
          nodeptr tmp = r;
          r = q;
          q = tmp;
        }

        while (!r->xBips) {
          if (!r->xBips)
            newviewBipartitions(bitVectors, r, numsp, vectorLength, processID);
        }

        for (i = 0; i < vectorLength; i++)
          vector[i] = left[i] | right[i];
      } else {
        while ((!r->xBips) || (!q->xBips)) {
          if (!q->xBips)
            newviewBipartitions(bitVectors, q, numsp, vectorLength, processID);
          if (!r->xBips)
            newviewBipartitions(bitVectors, r, numsp, vectorLength, processID);
        }

        for (i = 0; i < vectorLength; i++)
          vector[i] = left[i] | right[i];
      }
    }
  }
}

void bitVectorInitravSpecial(unsigned int **bitVectors, nodeptr p, int numsp,
                             unsigned int vectorLength, pllHashTable *h,
                             int treeNumber, int function, branchInfo *bInf,
                             int *countBranches, int treeVectorLength,
                             pllBoolean traverseOnly, pllBoolean computeWRF,
                             int processID) {
  if (isTip(p->number, numsp))
    return;
  else {
    nodeptr q = p->next;

    do {
      if (q->back) {
        bitVectorInitravSpecial(bitVectors, q->back, numsp, vectorLength, h,
                                treeNumber, function, bInf, countBranches,
                                treeVectorLength, traverseOnly, computeWRF,
                                processID);
      }
      q = q->next;
    } while (q != p);

    if (!p->next->back || !p->next->next->back)
      return;
    newviewBipartitions(bitVectors, p, numsp, vectorLength, processID);

    assert(p->xBips);

    assert(!traverseOnly);

    if (!(isTip(p->back->number, numsp))) {
      unsigned int *toInsert = bitVectors[p->number];

      hashNumberType position = p->hash % h->size;

      assert(!(toInsert[0] & 1));
      assert(!computeWRF);

      switch (function) {
      case PLL_BIPARTITIONS_RF:
        insertHashRF(toInsert, h, vectorLength, treeNumber, treeVectorLength,
                     position, 0, computeWRF);
        *countBranches = *countBranches + 1;
        break;
      default:
        assert(0);
      }
    }
  }
}

unsigned int **initBitVector(int mxtips, unsigned int *vectorLength) {
  unsigned int **bitVectors =
      (unsigned int **)rax_malloc(sizeof(unsigned int *) * 2 * (size_t)mxtips);

  int i;

  if (mxtips % PLL_MASK_LENGTH == 0)
    *vectorLength = mxtips / PLL_MASK_LENGTH;
  else
    *vectorLength = 1 + (mxtips / PLL_MASK_LENGTH);

  for (i = 1; i <= mxtips; i++) {
    bitVectors[i] = (unsigned int *)rax_calloc((size_t)(*vectorLength),
                                               sizeof(unsigned int));
    assert(bitVectors[i]);
    bitVectors[i][(i - 1) / PLL_MASK_LENGTH] |=
        mask32[(i - 1) % PLL_MASK_LENGTH];
  }

  for (i = mxtips + 1; i < 2 * mxtips; i++) {
    bitVectors[i] = (unsigned int *)rax_malloc(sizeof(unsigned int) *
                                               (size_t)(*vectorLength));
    assert(bitVectors[i]);
  }

  return bitVectors;
}

double convergenceCriterion(pllHashTable *h, int mxtips) {
  int rf = 0;

  unsigned int k = 0, entryCount = 0;

  double rrf;

  pllHashItem *hitem;

  for (k = 0, entryCount = 0; k < h->size; k++) {
    for (hitem = h->Items[k]; hitem; hitem = hitem->next) {
      pllBipartitionEntry *e = (pllBipartitionEntry *)hitem->data;
      unsigned int *vector = e->treeVector;

      if (((vector[0] & 1) > 0) + ((vector[0] & 2) > 0) == 1)
        rf++;

      entryCount++;
      e = e->next;
    }
  }

  assert(entryCount == h->entries);
  rrf = (double)rf / ((double)(2 * (mxtips - 3)));
  return rrf;
}