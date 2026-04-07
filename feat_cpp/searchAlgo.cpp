#include "pllInternal.hpp"
#include "utils.hpp"
#include <cassert>
#include <cmath>
#include <cstring>

/* === isTip: from utils.c === */
pllBoolean isTip(int number, int maxTips) {
  assert(number > 0);
  return (number <= maxTips) ? PLL_TRUE : PLL_FALSE;
}

/* === removeNodeBIG (simplified: no makenewzGeneric, no tr->zqr) ===
   Prune subtree rooted at p: connect p->next->back and p->next->next->back
   directly, detach p from the tree. */
nodeptr removeNodeBIG(pllInstance *tr, partitionList *pr, nodeptr p,
                      int numBranches) {
  nodeptr q, r;
  int i;

  q = p->next->back;
  r = p->next->next->back;

  /* Original uses makenewzGeneric to optimise the new branch length;
     simplified version just takes the product of the two old branch props. */
  double z[PLL_NUM_BRANCHES];
  for (i = 0; i < numBranches; i++)
    z[i] = q->z[i] * r->z[i];

  hookup(q, r, z, numBranches);

  p->next->next->back = p->next->back = (node *)NULL;

  return q;
}

/* === insertBIG (simplified: non-thorough only, no pllUpdatePartials /
   localSmooth) === Insert subtree p into the edge (q -- q->back). Edge q--r is
   removed; two new edges p->next--q and p->next->next--r are created. */
pllBoolean insertBIG(pllInstance *tr, partitionList *pr, nodeptr p, nodeptr q) {
  nodeptr r;
  int i;
  int numBranches = pr->perGeneBranchLengths ? pr->numberOfPartitions : 1;

  r = q->back;

  /* Simple branch-length split: sqrt of the original edge prop. */
  double z[PLL_NUM_BRANCHES];
  for (i = 0; i < numBranches; i++) {
    z[i] = sqrt(q->z[i]);
    if (z[i] < PLL_ZMIN)
      z[i] = PLL_ZMIN;
    if (z[i] > PLL_ZMAX)
      z[i] = PLL_ZMAX;
  }

  hookup(p->next, q, z, numBranches);
  hookup(p->next->next, r, z, numBranches);

  return PLL_TRUE;
}

/* === testInsertBIG (simplified: insert then immediately undo) ===
   Original inserts, evaluates likelihood, compares with bestOfNode/endLH,
   then restores. Simplified version just does the topology insert + restore. */
pllBoolean testInsertBIG(pllInstance *tr, partitionList *pr, nodeptr p,
                         nodeptr q) {
  int numBranches = pr->perGeneBranchLengths ? pr->numberOfPartitions : 1;
  double qz[PLL_NUM_BRANCHES];
  nodeptr r;
  int i;

  r = q->back;
  for (i = 0; i < numBranches; i++)
    qz[i] = q->z[i];

  if (!insertBIG(tr, pr, p, q))
    return PLL_FALSE;

  /* Restore original topology */
  hookup(q, r, qz, numBranches);
  p->next->next->back = p->next->back = (nodeptr)NULL;

  return PLL_TRUE;
}