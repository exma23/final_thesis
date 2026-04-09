#include "features.hpp"
#include "mem_alloc.hpp"
#include "newick.hpp"
#include "pll.hpp"
#include "pllInternal.hpp"
#include "treeIO.hpp"
#include "utils.hpp"
#include "hash.hpp"
#include <cmath>
#include <cstring>
#include <vector>

static const int FEAT_DIM = 19;

/* ── helpers ─────────────────────────────────────────────────── */

static pllInstance *loadTree(const char *newick_str) {
  pllNewickTree *nw = pllNewickParseString(newick_str);
  if (!nw)
    return nullptr;

  if (!pllValidateNewick(nw))
    pllNewickUnroot(nw);

  pllInstance *tr = (pllInstance *)rax_calloc(1, sizeof(pllInstance));
  if (!tr) {
    pllNewickParseDestroy(&nw);
    return nullptr;
  }

  pllTreeInitTopologyNewick(tr, nw, PLL_FALSE);
  pllNewickParseDestroy(&nw);
  return tr;
}

static void freeTree(pllInstance *tr) {
  if (!tr)
    return;
  if (tr->nameList) {
    for (int i = 1; i <= tr->mxtips; i++)
      rax_free(tr->nameList[i]);
    rax_free(tr->nameList);
  }
  if (tr->nameHash)
    pllHashDestroy(&tr->nameHash, 0);
  rax_free(tr->nodeBaseAddress);
  rax_free(tr->nodep);
  rax_free(tr->tree_string);
  rax_free(tr->tree0);
  rax_free(tr->tree1);
  rax_free(tr);
}

static bool applySPR(pllInstance *tr, partitionList *pr, int pruned,
                     int pruned_back, int regraft_num) {
  if (pruned < 1 || pruned > 2 * tr->mxtips - 2)
    return false;
  if (regraft_num < 1 || regraft_num > 2 * tr->mxtips - 2)
    return false;

  /* find the correct rotation of the pruned node */
  nodeptr prune = nullptr;
  nodeptr p = tr->nodep[pruned];

  if (p->number > tr->mxtips) {
    nodeptr q = p;
    do {
      if (q->back && q->back->number == pruned_back) {
        prune = q;
        break;
      }
      q = q->next;
    } while (q && q != p);
    if (!prune)
      prune = p;
  } else {
    prune = p;
  }

  if (!prune || !prune->back)
    return false;
  if (prune->number <= tr->mxtips)
    return false;
  if (!prune->next || !prune->next->back || !prune->next->next ||
      !prune->next->next->back)
    return false;

  /* save state for rollback */
  nodeptr p1 = prune->next->back;
  nodeptr p2 = prune->next->next->back;
  double savedZ1[PLL_NUM_BRANCHES], savedZ2[PLL_NUM_BRANCHES];
  for (int i = 0; i < PLL_NUM_BRANCHES; i++) {
    savedZ1[i] = prune->next->z[i];
    savedZ2[i] = prune->next->next->z[i];
  }

  removeNodeBIG(tr, pr, prune, 1);

  nodeptr reg = tr->nodep[regraft_num];
  if (!reg || !reg->back) {
    /* rollback */
    hookup(prune->next, p1, savedZ1, PLL_NUM_BRANCHES);
    hookup(prune->next->next, p2, savedZ2, PLL_NUM_BRANCHES);
    return false;
  }

  insertBIG(tr, pr, prune, reg);
  return true;
}

/* ── C API ───────────────────────────────────────────────────── */

extern "C" {

void get_state_action_c(const char *newick_str,
                        int action_idx,            /* index into previous action list, or -1 */
                        const char *gt_newick_str,
                        char *out_newick,
                        int out_newick_cap,
                        int *out_actions,        /* flat n*3 */
                        double *out_feats,       /* flat n*19 */
                        double *out_rewards,     /* n */
                        int *out_n_actions) {
  *out_n_actions = 0;
  out_newick[0] = '\0';

  /* 1. parse trees */
  pllInstance *tr = loadTree(newick_str);
  pllInstance *gt = loadTree(gt_newick_str);
  if (!tr || !gt) {
    freeTree(tr);
    freeTree(gt);
    return;
  }

  partitionList pr = {};
  pr.numberOfPartitions = 1;
  pr.perGeneBranchLengths = PLL_FALSE;

  /* 2. apply SPR if action is valid */
  if (action_idx >= 0) {
    int sprDist = tr->mxtips;
    std::vector<actionXy> prev = computeAllActions(tr, gt, &pr, sprDist);
    if (action_idx < (int)prev.size()) {
      applySPR(tr, &pr, prev[action_idx].pruned,
               prev[action_idx].pruned_back,
               prev[action_idx].regraft);
    }
  }

  /* 3. write current tree to output buffer */
  out_newick[0] = '\0';
  pllTreeToNewick(out_newick, tr, tr->start->back, PLL_TRUE, PLL_TRUE);
  /* strip trailing newline */
  size_t len = strlen(out_newick);
  while (len > 0 && (out_newick[len - 1] == '\n' || out_newick[len - 1] == '\r'))
    out_newick[--len] = '\0';

  /* 4. enumerate all SPR moves with features + RF reward */
  int sprDist = tr->mxtips;
  std::vector<actionXy> actions = computeAllActions(tr, gt, &pr, sprDist);

  /* 5. copy to output arrays */
  int n = (int)actions.size();
  *out_n_actions = n;

  for (int i = 0; i < n; i++) {
    out_actions[i * 3 + 0] = actions[i].pruned;
    out_actions[i * 3 + 1] = actions[i].pruned_back;
    out_actions[i * 3 + 2] = actions[i].regraft;

    const SPRFeatures &f = actions[i].feat;
    double *fp = &out_feats[i * FEAT_DIM];
    fp[0]  = f.total_branch_lengths;
    fp[1]  = f.longest_branch;
    fp[2]  = f.prune_branch_len;
    fp[3]  = f.regraft_branch_len;
    fp[4]  = (double)f.topo_distance;
    fp[5]  = f.branch_len_distance;
    fp[6]  = f.new_branch_len;
    fp[7]  = (double)f.n_leaves_a;
    fp[8]  = (double)f.n_leaves_b;
    fp[9]  = (double)f.n_leaves_b1;
    fp[10] = (double)f.n_leaves_b2;
    fp[11] = f.total_bl_a;
    fp[12] = f.total_bl_b;
    fp[13] = f.total_bl_b1;
    fp[14] = f.total_bl_b2;
    fp[15] = f.longest_bl_a;
    fp[16] = f.longest_bl_b;
    fp[17] = f.longest_bl_b1;
    fp[18] = f.longest_bl_b2;

    out_rewards[i] = (double)actions[i].reward;
  }

  /* 6. cleanup */
  freeTree(tr);
  freeTree(gt);
}

} /* extern "C" */