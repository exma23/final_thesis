#pragma once
#include "pll.hpp"
#include "pllInternal.hpp"
#include "vector"

struct SPRFeatures {
  double total_branch_lengths; /* feat  1 */
  double longest_branch;       /* feat  2 */
  double prune_branch_len;     /* feat  3 */
  double regraft_branch_len;   /* feat  4 */
  int topo_distance;           /* feat  5 */
  double branch_len_distance;  /* feat  6 */
  double new_branch_len;       /* feat  7 */
  int n_leaves_a;              /* feat  8 */
  int n_leaves_b;              /* feat  9 */
  int n_leaves_b1;             /* feat 10 */
  int n_leaves_b2;             /* feat 11 */
  double total_bl_a;           /* feat 12 */
  double total_bl_b;           /* feat 13 */
  double total_bl_b1;          /* feat 14 */
  double total_bl_b2;          /* feat 15 */
  double longest_bl_a;         /* feat 16 */
  double longest_bl_b;         /* feat 17 */
  double longest_bl_b1;        /* feat 18 */
  double longest_bl_b2;        /* feat 19 */
};

struct actionXy {
  int pruned;       /* id of pruned node */
  int regraft;      /* id of regrafted node */
  SPRFeatures feat; /* encoded state + action */
  int pruned_back;  // thêm: p->back->number để xác định rotation
  int reward; /* minus RF distance from the truth tree to state after apply the
                 SPR move */
};

std::vector<SPRFeatures> computeAllSPRFeatures(pllInstance *tr, partitionList *pr, int sprDist);
std::vector<actionXy> computeAllActions(pllInstance *tr, pllInstance *gt_tr, partitionList *pr, int sprDist);