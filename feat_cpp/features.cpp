#include "features.hpp"
#include "mem_alloc.hpp"
#include "pllInternal.hpp"
#include "utils.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

static inline double branchLength(double z) {
  if (z < PLL_ZMIN)
    z = PLL_ZMIN;
  if (z > PLL_ZMAX)
    z = PLL_ZMAX;
  return -log(z);
}

static int countLeaves(nodeptr p, int mxtips) {
  if (!p || p->z[0] <= 0.0) {
    return 0;
  }

  /* leaf node */
  if (p->number <= mxtips) {
    return 1;
  }

  /* Validate internal node structure */
  if (!p->next || !p->next->back || !p->next->next || !p->next->next->back) {
    return 0;
  }

  return countLeaves(p->next->back, mxtips) +
         countLeaves(p->next->next->back, mxtips);
}

/* Sum of edge BLs in subtree below p (NOT including edge p↔p->back) */
static double subtreeBLSum(nodeptr p, int mxtips) {
  if (!p || p->z[0] <= 0.0) { // Skip uninitialized nodes
    return 0.0;
  }

  if (p->number <= mxtips) /* leaf */
    return 0.0;

  /* Check node structure */
  if (!p->next || !p->next->back || !p->next->next || !p->next->next->back) {
    return 0.0;
  }

  double bl1 = branchLength(p->next->back->z[0]);
  double bl2 = branchLength(p->next->next->back->z[0]);

  return bl1 + subtreeBLSum(p->next->back, mxtips) + bl2 +
         subtreeBLSum(p->next->next->back, mxtips);
}

/* Longest edge BL in subtree below p (NOT including edge p↔p->back) */
static double subtreeLongestBL(nodeptr p, int mxtips) {
  if (!p || p->z[0] <= 0.0) { // Skip uninitialized nodes
    return 0.0;
  }

  if (p->number <= mxtips)
    return 0.0;

  if (!p->next || !p->next->back || !p->next->next || !p->next->next->back)
    return 0.0;

  double bl1 = branchLength(p->next->back->z[0]);
  double bl2 = branchLength(p->next->next->back->z[0]);
  double best = (bl1 > bl2) ? bl1 : bl2;
  double s1 = subtreeLongestBL(p->next->back, mxtips);
  double s2 = subtreeLongestBL(p->next->next->back, mxtips);
  if (s1 > best)
    best = s1;
  if (s2 > best)
    best = s2;
  return best;
}

/* Whole-tree stats starting from tr->start (a tip) */
static void wholeTreeStats(pllInstance *tr, double &totalBL,
                           double &longestBL) {
  double rootEdge = branchLength(tr->start->z[0]);
  totalBL = rootEdge + subtreeBLSum(tr->start->back, tr->mxtips);
  longestBL = rootEdge;
  double sub = subtreeLongestBL(tr->start->back, tr->mxtips);
  if (sub > longestBL)
    longestBL = sub;
}

/* ============================================================
 * Remove / Restore  (branch-length preserving)
 * ============================================================ */

static nodeptr removeNodeBL(nodeptr p, double savedZ1[PLL_NUM_BRANCHES],
                            double savedZ2[PLL_NUM_BRANCHES]) {
  nodeptr q = p->next->back;
  nodeptr r = p->next->next->back;
  for (int i = 0; i < PLL_NUM_BRANCHES; i++) {
    savedZ1[i] = q->z[i];
    savedZ2[i] = r->z[i];
  }
  // reconnect q↔r  with z = product  (BL_new ≈ BL_q + BL_r)
  double z[PLL_NUM_BRANCHES];
  for (int i = 0; i < PLL_NUM_BRANCHES; i++)
    z[i] = q->z[i] * r->z[i];
  hookup(q, r, z, PLL_NUM_BRANCHES);
  p->next->back = p->next->next->back = (node *)NULL;
  return q;
}

static void restoreNodeBL(nodeptr p, nodeptr p1, nodeptr p2,
                          double savedZ1[PLL_NUM_BRANCHES],
                          double savedZ2[PLL_NUM_BRANCHES]) {
  hookup(p->next, p1, savedZ1, PLL_NUM_BRANCHES);
  hookup(p->next->next, p2, savedZ2, PLL_NUM_BRANCHES);
}

/* ============================================================
 * Recursive traversal:  for each regraft target, compute features
 * ============================================================ */

static void
addTraverseFeatures(pllInstance *tr, nodeptr q, /* current regraft candidate */
                    int mintrav, int maxtrav,
                    int depth, /* topo distance from prune node to here */
                    double pathBLSum, /* sum of intermediate edge BLs on path */
                    double pruneBL, int n_leaves_a, double total_bl_a,
                    double longest_bl_a, double treeTotalBL,
                    double treeLongestBL, int totalLeaves,
                    std::vector<SPRFeatures> &results) {
  if (!q)
    return;
  if (--mintrav <= 0) {
    SPRFeatures f = {};
    /* global tree features (original tree) */
    f.total_branch_lengths = treeTotalBL;
    f.longest_branch = treeLongestBL;

    f.prune_branch_len = pruneBL;

    double regraftBL = branchLength(q->z[0]);
    f.regraft_branch_len = regraftBL;
    f.topo_distance = depth;
    f.branch_len_distance = pathBLSum;
    f.new_branch_len = regraftBL / 2.0; /* insertBIG uses sqrt(z) ⇒ BL/2 */

    /* subtree a (pruned subtree – precomputed) */
    f.n_leaves_a = n_leaves_a;
    f.total_bl_a = total_bl_a;
    f.longest_bl_a = longest_bl_a;

    /* subtree b  split at regraft edge into b1 (q side) + b2 (q->back side) */
    f.n_leaves_b1 = countLeaves(q, tr->mxtips);

    f.n_leaves_b2 = countLeaves(q->back, tr->mxtips);
    f.n_leaves_b = f.n_leaves_b1 + f.n_leaves_b2;

    f.total_bl_b1 = subtreeBLSum(q, tr->mxtips);
    f.total_bl_b2 = subtreeBLSum(q->back, tr->mxtips);
    f.total_bl_b = f.total_bl_b1 + f.total_bl_b2 + regraftBL;

    f.longest_bl_b1 = subtreeLongestBL(q, tr->mxtips);
    f.longest_bl_b2 = subtreeLongestBL(q->back, tr->mxtips);
    f.longest_bl_b = regraftBL;
    if (f.longest_bl_b1 > f.longest_bl_b)
      f.longest_bl_b = f.longest_bl_b1;
    if (f.longest_bl_b2 > f.longest_bl_b)
      f.longest_bl_b = f.longest_bl_b2;

    results.push_back(f);
  }

  if ((q->number > tr->mxtips) && (--maxtrav > 0)) {
    /* current regraft edge becomes an intermediate edge */
    double newPathBLSum = pathBLSum + branchLength(q->z[0]);

    addTraverseFeatures(tr, q->next->back, mintrav, maxtrav, depth + 1,
                        newPathBLSum, pruneBL, n_leaves_a, total_bl_a,
                        longest_bl_a, treeTotalBL, treeLongestBL, totalLeaves,
                        results);
    addTraverseFeatures(tr, q->next->next->back, mintrav, maxtrav, depth + 1,
                        newPathBLSum, pruneBL, n_leaves_a, total_bl_a,
                        longest_bl_a, treeTotalBL, treeLongestBL, totalLeaves,
                        results);
  }
}

/* ============================================================
 * Per-node SPR enumeration  (mirrors rearrangeParsimony layout)
 * ============================================================ */

static void rearrangeFeatures(pllInstance *tr, nodeptr p, int mintrav,
                              int maxtrav, double treeTotalBL,
                              double treeLongestBL, int totalLeaves,
                              std::vector<SPRFeatures> &results) {
  if (maxtrav > totalLeaves - 3)
    maxtrav = totalLeaves - 3;
  if (maxtrav < mintrav)
    return;
  nodeptr q = p->back;
  /* === P side: prune subtree hanging from p->back === */
  if (p->number > tr->mxtips) {
    nodeptr p1 = p->next->back;
    nodeptr p2 = p->next->next->back;
    if ((p1->number > tr->mxtips) || (p2->number > tr->mxtips)) {
      /* subtree-a stats (computed on original tree before pruning) */

      double pruneBL = branchLength(p->z[0]);
      int n_a = countLeaves(p, tr->mxtips);
      double tbl_a = subtreeBLSum(p, tr->mxtips);
      double lbl_a = subtreeLongestBL(p, tr->mxtips);

      double savedZ1[PLL_NUM_BRANCHES], savedZ2[PLL_NUM_BRANCHES];
      removeNodeBL(p, savedZ1, savedZ2);

      /* initial path BL = edge from prune node p to neighbor p1 or p2 */
      double bl_to_p1 = branchLength(savedZ1[0]);
      double bl_to_p2 = branchLength(savedZ2[0]);

      if (p1->number > tr->mxtips) {
        addTraverseFeatures(tr, p1->next->back, mintrav, maxtrav, 1, bl_to_p1,
                            pruneBL, n_a, tbl_a, lbl_a, treeTotalBL,
                            treeLongestBL, totalLeaves, results);
        addTraverseFeatures(tr, p1->next->next->back, mintrav, maxtrav, 1,
                            bl_to_p1, pruneBL, n_a, tbl_a, lbl_a, treeTotalBL,
                            treeLongestBL, totalLeaves, results);
      }
      if (p2->number > tr->mxtips) {
        addTraverseFeatures(tr, p2->next->back, mintrav, maxtrav, 1, bl_to_p2,
                            pruneBL, n_a, tbl_a, lbl_a, treeTotalBL,
                            treeLongestBL, totalLeaves, results);
        addTraverseFeatures(tr, p2->next->next->back, mintrav, maxtrav, 1,
                            bl_to_p2, pruneBL, n_a, tbl_a, lbl_a, treeTotalBL,
                            treeLongestBL, totalLeaves, results);
      }

      restoreNodeBL(p, p1, p2, savedZ1, savedZ2);
    }
  }

  /* === Q side: prune subtree hanging from q->back = p === */
  if ((q->number > tr->mxtips) && (maxtrav > 0)) {

    nodeptr q1 = q->next->back;
    nodeptr q2 = q->next->next->back;
    if (!q1 || !q2) {
      fprintf(stderr,
              "  [Q-SIDE] SKIP node %d: q=%d has NULL neighbor (q1=%p q2=%p)\n",
              p->number, q->number, q1, q2);
      return;
    }

    if (((q1->number > tr->mxtips) && q1->next && q1->next->back &&
         q1->next->next && q1->next->next->back &&
         ((q1->next->back->number > tr->mxtips) ||
          (q1->next->next->back->number > tr->mxtips))) ||
        ((q2->number > tr->mxtips) && q2->next && q2->next->back &&
         q2->next->next && q2->next->next->back &&
         ((q2->next->back->number > tr->mxtips) ||
          (q2->next->next->back->number > tr->mxtips)))) {
      double pruneBL = branchLength(q->z[0]);

      int n_a = countLeaves(q, tr->mxtips);
      double tbl_a = subtreeBLSum(q, tr->mxtips);
      double lbl_a = subtreeLongestBL(q, tr->mxtips);

      double savedZ1[PLL_NUM_BRANCHES], savedZ2[PLL_NUM_BRANCHES];

      removeNodeBL(q, savedZ1, savedZ2);

      int mintrav2 = mintrav > 2 ? mintrav : 2;

      double bl_to_q1 = branchLength(savedZ1[0]);
      double bl_to_q2 = branchLength(savedZ2[0]);

      if (q1->number > tr->mxtips) {

        addTraverseFeatures(tr, q1->next->back, mintrav2, maxtrav, 1, bl_to_q1,
                            pruneBL, n_a, tbl_a, lbl_a, treeTotalBL,
                            treeLongestBL, totalLeaves, results);
        addTraverseFeatures(tr, q1->next->next->back, mintrav2, maxtrav, 1,
                            bl_to_q1, pruneBL, n_a, tbl_a, lbl_a, treeTotalBL,
                            treeLongestBL, totalLeaves, results);
      }
      if (q2->number > tr->mxtips) {

        addTraverseFeatures(tr, q2->next->back, mintrav2, maxtrav, 1, bl_to_q2,
                            pruneBL, n_a, tbl_a, lbl_a, treeTotalBL,
                            treeLongestBL, totalLeaves, results);
        addTraverseFeatures(tr, q2->next->next->back, mintrav2, maxtrav, 1,
                            bl_to_q2, pruneBL, n_a, tbl_a, lbl_a, treeTotalBL,
                            treeLongestBL, totalLeaves, results);
      }

      restoreNodeBL(q, q1, q2, savedZ1, savedZ2);
    }
  }
}

std::vector<SPRFeatures> computeAllSPRFeatures(pllInstance *tr,
                                               partitionList *pr, int sprDist) {
  std::vector<SPRFeatures> results;

  double treeTotalBL, treeLongestBL;
  wholeTreeStats(tr, treeTotalBL, treeLongestBL);
  int totalLeaves = tr->mxtips;

  for (int i = 1; i <= 2 * tr->mxtips - 2; i++) {
    if (!tr->nodep[i] || tr->nodep[i]->z[0] <= 0.0)
      continue; // Skip NULL or uninitialized nodes

    rearrangeFeatures(tr, tr->nodep[i], 1, sprDist, treeTotalBL, treeLongestBL,
                      totalLeaves, results);
  }

  return results;
}