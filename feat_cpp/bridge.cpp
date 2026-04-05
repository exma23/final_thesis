#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <math.h>
#include "pll.hpp"
#include "mem_alloc.hpp"
#include "newick.hpp"
#include "hash.hpp"
#include "features.hpp"
#include "utils.hpp"

static partitionList * createDummyPartitionList()
{
    partitionList *pr = (partitionList *) rax_calloc(1, sizeof(partitionList));
    pr->numberOfPartitions  = 1;
    pr->perGeneBranchLengths = PLL_FALSE;
    pr->dirty                = PLL_FALSE;
    pr->partitionData        = nullptr;
    pr->alphaList            = nullptr;
    pr->rateList             = nullptr;
    pr->freqList             = nullptr;
    return pr;
}

static void destroyDummyPartitionList(partitionList *pr)
{
    rax_free(pr);
}

/* ── pllCreateRearrangeList / pllDestroyRearrangeList ──────────────────── */

pllRearrangeList * pllCreateRearrangeList(int max)
{
    pllRearrangeList *bl = (pllRearrangeList *) malloc(sizeof(pllRearrangeList));
    bl->max_entries = max;
    bl->entries     = 0;
    bl->rearr       = (pllRearrangeInfo *) malloc(max * sizeof(pllRearrangeInfo));
    return bl;
}

void pllDestroyRearrangeList(pllRearrangeList **bestList)
{
    free((*bestList)->rearr);
    free(*bestList);
    *bestList = nullptr;
}

/* ── tree ↔ newick helpers ────────────────────────────────────────────── */

static pllInstance * newickToTree(const char *newick_str)
{
    pllNewickTree *newick = pllNewickParseString(newick_str);
    if (!newick) return nullptr;

    pllInstance *tr = (pllInstance *) rax_calloc(1, sizeof(pllInstance));
    tr->fracchange = 1.0;
    pllTreeInitTopologyNewick(tr, newick, PLL_FALSE);
    pllNewickParseDestroy(&newick);
    return tr;
}

static void destroyTree(pllInstance *tr)
{
    if (!tr) return;
    for (int i = 1; i <= tr->mxtips; ++i)
        free(tr->nameList[i]);
    rax_free(tr->nameList);
    rax_free(tr->nodep);
    rax_free(tr->nodeBaseAddress);
    rax_free(tr->constraintVector);
    rax_free(tr->tree_string);
    rax_free(tr->tree0);
    rax_free(tr->tree1);
    rax_free(tr->td[0].ti);
    rax_free(tr->td[0].parameterValues);
    rax_free(tr->td[0].executeModel);
    if (tr->nameHash) pllHashDestroy(&tr->nameHash, PLL_FALSE);
    rax_free(tr);
}

/* Simple recursive newick serializer (topology-only with branch lengths) */
static char * treeToNewickRec(char *buf, pllInstance *tr, nodeptr p)
{
    if (isTip(p->number, tr->mxtips))
    {
        buf += sprintf(buf, "%s", tr->nameList[p->number]);
    }
    else
    {
        *buf++ = '(';
        buf = treeToNewickRec(buf, tr, p->next->back);
        *buf++ = ',';
        buf = treeToNewickRec(buf, tr, p->next->next->back);
        if (p == tr->start->back)
        {
            *buf++ = ',';
            buf = treeToNewickRec(buf, tr, p->back);
        }
        *buf++ = ')';
    }

    if (p == tr->start->back)
        buf += sprintf(buf, ";");
    else
        buf += sprintf(buf, ":%.5f", -log(p->z[0]) * tr->fracchange);

    return buf;
}

static void treeToNewick(pllInstance *tr, char *buf, int cap)
{
    nodeptr p = tr->start->back;
    char *end = treeToNewickRec(buf, tr, p);
    (void)end;
    (void)cap;
}

/* ── RF distance (bitmask-based, exact for small trees) ───────────────── */

typedef unsigned int bitmask_t;

static bitmask_t collectBipartition(nodeptr p, int mxtips)
{
    if (isTip(p->number, mxtips))
        return 1u << (p->number - 1);

    return collectBipartition(p->next->back, mxtips) |
           collectBipartition(p->next->next->back, mxtips);
}

static int collectAllBipartitions(pllInstance *tr, bitmask_t *bips, int maxBips)
{
    int count = 0;
    bitmask_t allTaxa = (1u << tr->mxtips) - 1;

    for (int i = tr->mxtips + 1; i <= 2 * tr->mxtips - 2; ++i)
    {
        nodeptr p = tr->nodep[i];
        for (int j = 0; j < 3; ++j, p = p->next)
        {
            if (!p->back) continue;
            bitmask_t mask = collectBipartition(p->back, tr->mxtips);
            if (mask & 1u) mask = allTaxa ^ mask;
            if (mask == 0 || mask == allTaxa) continue;
            if (__builtin_popcount(mask) <= 1) continue;
            bool dup = false;
            for (int k = 0; k < count; ++k)
                if (bips[k] == mask) { dup = true; break; }
            if (!dup && count < maxBips) bips[count++] = mask;
        }
    }
    return count;
}

static int computeRF(pllInstance *t1, pllInstance *t2)
{
    bitmask_t bips1[256], bips2[256];
    int n1 = collectAllBipartitions(t1, bips1, 256);
    int n2 = collectAllBipartitions(t2, bips2, 256);

    int diff = 0;
    for (int i = 0; i < n1; ++i)
    {
        int found = 0;
        for (int j = 0; j < n2; ++j)
            if (bips1[i] == bips2[j]) { found = 1; break; }
        if (!found) ++diff;
    }
    for (int i = 0; i < n2; ++i)
    {
        int found = 0;
        for (int j = 0; j < n1; ++j)
            if (bips2[i] == bips1[j]) { found = 1; break; }
        if (!found) ++diff;
    }
    return diff;
}

/* ── apply an SPR move on the tree ────────────────────────────────────── */

static void applySPR(pllInstance *tr, partitionList *pr,
                     int rmNumber, int insNumber)
{
    nodeptr p = tr->nodep[rmNumber];
    nodeptr q = tr->nodep[insNumber];
    int numBranches = 1;

    removeNodeBIG(tr, pr, p, numBranches);
    insertNodeBIG(p, q, numBranches);
}

/* ── main C API ───────────────────────────────────────────────────────── */

extern "C" {

void get_state_action_c(
    const char *newick_str,
    const int  *action,
    const char *gt_newick_str,
    char       *out_newick,
    int         out_newick_cap,
    int        *out_actions,
    double     *out_feats,
    double     *out_rewards,
    int        *out_n_actions
)
{
    const int FEAT_DIM = 19;

    /* 1. parse current tree */
    pllInstance *tr = newickToTree(newick_str);
    partitionList *pr = createDummyPartitionList();

    /* 2. apply chosen action (if not initial) */
    if (action[0] >= 0)
        applySPR(tr, pr, action[0], action[1]);

    /* 3. serialize current tree state to output */
    treeToNewick(tr, out_newick, out_newick_cap);

    /* 4. parse ground truth tree */
    pllInstance *gt_tr = newickToTree(gt_newick_str);

    /* 5. compute current RF distance (before any candidate move) */
    int rf_before = computeRF(tr, gt_tr);

    /* 6. collect all SPR moves + compute features */
    int n_moves = 0;
    SPRMoveRecordList *moveList =
        collectAllSPRMoves(tr, pr, 1, 20, 10000);
    n_moves = moveList->entries;

    /* global tree stats (computed once) */
    double tree_total_bl = 0.0, tree_longest_bl = 0.0;
    {
        double sum = 0.0, maxbl = 0.0;
        /* tips: one edge each */
        for (int i = 1; i <= tr->mxtips; ++i)
        {
            double bl = tr->nodep[i]->z[0];
            sum += bl;
            if (bl > maxbl) maxbl = bl;
        }
        /* internal nodes: three edges each (via ring) */
        for (int i = tr->mxtips + 1; i <= 2 * tr->mxtips - 2; ++i)
        {
            nodeptr p = tr->nodep[i];
            for (int j = 0; j < 3; ++j, p = p->next)
            {
                double bl = p->z[0];
                sum += bl;
                if (bl > maxbl) maxbl = bl;
            }
        }
        tree_total_bl  = sum / 2.0;
        tree_longest_bl = maxbl;
    }

    /* 7. for each move: compute features + reward (apply→RF→undo) */
    for (int m = 0; m < n_moves; ++m)
    {
        SPRMoveRecord *mv = &moveList->moves[m];

        /* --- features (read-only) --- */
        SPRFeatures f = computeSPRFeatures(tr, mv, tree_total_bl, tree_longest_bl);
        double *fp = out_feats + m * FEAT_DIM;
        fp[ 0] = f.total_branch_lengths;
        fp[ 1] = f.longest_branch;
        fp[ 2] = f.prune_branch_len;
        fp[ 3] = f.regraft_branch_len;
        fp[ 4] = (double) f.topo_distance;
        fp[ 5] = f.branch_len_distance;
        fp[ 6] = f.new_branch_len;
        fp[ 7] = (double) f.n_leaves_a;
        fp[ 8] = (double) f.n_leaves_b;
        fp[ 9] = (double) f.n_leaves_b1;
        fp[10] = (double) f.n_leaves_b2;
        fp[11] = f.total_bl_a;
        fp[12] = f.total_bl_b;
        fp[13] = f.total_bl_b1;
        fp[14] = f.total_bl_b2;
        fp[15] = f.longest_bl_a;
        fp[16] = f.longest_bl_b;
        fp[17] = f.longest_bl_b1;
        fp[18] = f.longest_bl_b2;

        /* --- actions --- */
        out_actions[m * 2 + 0] = mv->removeNode->number;
        out_actions[m * 2 + 1] = mv->insertNode->number;

        /* --- reward: apply move → compute RF → undo --- */
        nodeptr rm_p   = mv->removeNode;
        nodeptr ins_q  = mv->insertNode;
        nodeptr ins_r  = ins_q->back;

        /* save topology around prune point */
        nodeptr p1 = rm_p->next->back;
        nodeptr p2 = rm_p->next->next->back;
        double  p1z[PLL_NUM_BRANCHES], p2z[PLL_NUM_BRANCHES];
        double  qrz[PLL_NUM_BRANCHES];
        double  pz[PLL_NUM_BRANCHES], pbz[PLL_NUM_BRANCHES];   /* <-- NEW */
        for (int i = 0; i < PLL_NUM_BRANCHES; ++i)
        {
            p1z[i] = p1->z[i];
            p2z[i] = p2->z[i];
            qrz[i] = ins_q->z[i];
            pz[i]  = rm_p->z[i];              /* <-- NEW */
            pbz[i] = rm_p->back->z[i];        /* <-- NEW */
        }

        /* apply */
        removeNodeBIG(tr, pr, rm_p, 1);
        insertNodeBIG(rm_p, ins_q, 1);

        /* compute RF after move */
        int rf_after = computeRF(tr, gt_tr);
        out_rewards[m] = (double)(rf_before - rf_after);

        /* undo: disconnect rm_p from ins_q/ins_r, reconnect ins_q<->ins_r,
                 reconnect rm_p into original position */
        hookup(ins_q, ins_r, qrz, 1);
        rm_p->next->next->back = rm_p->next->back = nullptr;
        hookup(rm_p->next,       p1, p1z, 1);
        hookup(rm_p->next->next, p2, p2z, 1);
        /* restore the subtree-root edge that insertNodeBIG overwrote */
        for (int i = 0; i < PLL_NUM_BRANCHES; ++i)    /* <-- NEW */
        {
            rm_p->z[i]       = pz[i];
            rm_p->back->z[i] = pbz[i];
        }
    }

    *out_n_actions = n_moves;

    /* cleanup */
    destroySPRMoveRecordList(&moveList);
    destroyTree(gt_tr);
    destroyDummyPartitionList(pr);
    destroyTree(tr);
}

} /* extern "C" */