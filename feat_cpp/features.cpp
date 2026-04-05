#include <cstring>
#include <cstdlib>
#include "features.hpp"
#include "pllInternal.hpp"
#include "utils.hpp"

/* ── subtree stats (recursive, read-only) ───────────────────────────────── */

static void subtreeStats(nodeptr p, int mxtips,
                         int *leaves, double *total_bl, double *longest_bl)
{
    double bl = p->z[0];
    if (bl > *longest_bl) *longest_bl = bl;
    *total_bl += bl;

    if (isTip(p->number, mxtips))
    {
        *leaves += 1;
        return;
    }
    subtreeStats(p->next->back,       mxtips, leaves, total_bl, longest_bl);
    subtreeStats(p->next->next->back, mxtips, leaves, total_bl, longest_bl);
}

/* ── global tree stats ───────────────────────────────────────────────────── */

static void treeGlobalStats(pllInstance *tr, double *total_bl, double *longest_bl)
{
    double sum = 0.0, maxbl = 0.0;

    /* tips: each tip node has exactly one edge */
    for (int i = 1; i <= tr->mxtips; ++i)
    {
        double bl = tr->nodep[i]->z[0];
        sum += bl;
        if (bl > maxbl) maxbl = bl;
    }

    /* internal nodes: each has 3 edges via the ring (p, p->next, p->next->next) */
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

    /* every edge is counted twice (p->z == p->back->z), so halve the sum */
    *total_bl   = sum / 2.0;
    *longest_bl = maxbl;
}

/* ── collection: records topo_depth and path_bl_sum at traversal time ───── */

struct CollectCtx
{
    SPRMoveRecordList *list;
    int                mxtips;
};

/* depth = number of edges already stepped past toward insertNode.
   At the first valid call site (mintrav==1 → depth == original_mintrav-1),
   depth is passed explicitly so no post-hoc path-walking is needed. */
static void traverseCollectExact(nodeptr p, nodeptr q,
                                 int mintrav, int maxtrav,
                                 int depth,            /* edges walked so far */
                                 double path_bl_sum,   /* sum of z walked so far */
                                 CollectCtx *ctx)
{
    /* advance one step */
    double edge_bl = q->z[0];
    ++depth;
    path_bl_sum += edge_bl;
    --mintrav;

    if (mintrav <= 0)
    {
        /* record this move */
        SPRMoveRecordList *lst = ctx->list;
        if (lst->entries < lst->max_entries)
        {
            SPRMoveRecord *rec = &lst->moves[lst->entries++];
            rec->removeNode  = p;
            rec->insertNode  = q;
            rec->topo_depth  = depth;
            rec->path_bl_sum = path_bl_sum;
        }
    }

    --maxtrav;
    if (!isTip(q->number, ctx->mxtips) && maxtrav > 0)
    {
        traverseCollectExact(p, q->next->back,       mintrav, maxtrav, depth, path_bl_sum, ctx);
        traverseCollectExact(p, q->next->next->back, mintrav, maxtrav, depth, path_bl_sum, ctx);
    }
}

static int collectSPRExact(pllInstance *tr, partitionList *pr, nodeptr p,
                           int mintrav, int maxtrav, CollectCtx *ctx)
{
    nodeptr p1, p2, q, q1, q2;
    double  p1z[PLL_NUM_BRANCHES], p2z[PLL_NUM_BRANCHES];
    double  q1z[PLL_NUM_BRANCHES], q2z[PLL_NUM_BRANCHES];
    int     mintrav2, i;
    int     numBranches = pr->perGeneBranchLengths ? pr->numberOfPartitions : 1;

    if (maxtrav < 1 || mintrav > maxtrav) return PLL_FALSE;
    q = p->back;

    /* === prune side p === */
    if (!isTip(p->number, tr->mxtips))
    {
        p1 = p->next->back;
        p2 = p->next->next->back;

        if (!isTip(p1->number, tr->mxtips) || !isTip(p2->number, tr->mxtips))
        {
            for (i = 0; i < numBranches; ++i)
              { p1z[i] = p1->z[i]; p2z[i] = p2->z[i]; }

            removeNodeBIG(tr, pr, p, numBranches);

            /* path_bl_sum starts at 0 (we don't count the prune edge itself,
               matching feature definition: "not including these branches") */
            if (!isTip(p1->number, tr->mxtips))
            {
                traverseCollectExact(p, p1->next->back,       mintrav, maxtrav, 0, 0.0, ctx);
                traverseCollectExact(p, p1->next->next->back, mintrav, maxtrav, 0, 0.0, ctx);
            }
            if (!isTip(p2->number, tr->mxtips))
            {
                traverseCollectExact(p, p2->next->back,       mintrav, maxtrav, 0, 0.0, ctx);
                traverseCollectExact(p, p2->next->next->back, mintrav, maxtrav, 0, 0.0, ctx);
            }

            hookup(p->next,       p1, p1z, numBranches);
            hookup(p->next->next, p2, p2z, numBranches);
        }
    }

    /* === prune side q === */
    if (!isTip(q->number, tr->mxtips) && maxtrav > 0)
    {
        q1 = q->next->back;
        q2 = q->next->next->back;

        if (
            (!isTip(q1->number, tr->mxtips) &&
             (!isTip(q1->next->back->number,       tr->mxtips) ||
              !isTip(q1->next->next->back->number, tr->mxtips)))
            ||
            (!isTip(q2->number, tr->mxtips) &&
             (!isTip(q2->next->back->number,       tr->mxtips) ||
              !isTip(q2->next->next->back->number, tr->mxtips)))
           )
        {
            for (i = 0; i < numBranches; ++i)
              { q1z[i] = q1->z[i]; q2z[i] = q2->z[i]; }

            removeNodeBIG(tr, pr, q, numBranches);

            mintrav2 = mintrav > 2 ? mintrav : 2;

            if (!isTip(q1->number, tr->mxtips))
            {
                traverseCollectExact(q, q1->next->back,       mintrav2, maxtrav, 0, 0.0, ctx);
                traverseCollectExact(q, q1->next->next->back, mintrav2, maxtrav, 0, 0.0, ctx);
            }
            if (!isTip(q2->number, tr->mxtips))
            {
                traverseCollectExact(q, q2->next->back,       mintrav2, maxtrav, 0, 0.0, ctx);
                traverseCollectExact(q, q2->next->next->back, mintrav2, maxtrav, 0, 0.0, ctx);
            }

            hookup(q->next,       q1, q1z, numBranches);
            hookup(q->next->next, q2, q2z, numBranches);
        }
    }

    return PLL_TRUE;
}

/* ── public: collection ─────────────────────────────────────────────────── */

SPRMoveRecordList * collectAllSPRMoves(pllInstance *tr, partitionList *pr,
                                       int mintrav, int maxtrav, int maxMoves)
{
    SPRMoveRecordList *list =
        (SPRMoveRecordList *) malloc(sizeof(SPRMoveRecordList));
    list->max_entries = maxMoves;
    list->entries     = 0;
    list->moves       = (SPRMoveRecord *) malloc(maxMoves * sizeof(SPRMoveRecord));

    CollectCtx ctx;
    ctx.list   = list;
    ctx.mxtips = tr->mxtips;

    int n = tr->mxtips + tr->mxtips - 2;
    for (int i = 1; i <= n; ++i)
        collectSPRExact(tr, pr, tr->nodep[i], mintrav, maxtrav, &ctx);

    return list;
}

void destroySPRMoveRecordList(SPRMoveRecordList **list)
{
    free((*list)->moves);
    free(*list);
    *list = nullptr;
}

/* ── public: feature computation ───────────────────────────────────────── */

SPRFeatures computeSPRFeatures(pllInstance *tr,
                               const SPRMoveRecord *move,
                               double tree_total_bl,
                               double tree_longest_bl)
{
    SPRFeatures f;

    f.total_branch_lengths = tree_total_bl;
    f.longest_branch       = tree_longest_bl;
    f.prune_branch_len     = move->removeNode->z[0];
    f.regraft_branch_len   = move->insertNode->z[0];
    f.topo_distance        = move->topo_depth;
    f.branch_len_distance  = move->path_bl_sum;
    f.new_branch_len       = PLL_DEFAULTZ;  /* topology-only: use default starting value */

    /* subtree a: the pruned subtree rooted at removeNode */
    f.n_leaves_a  = 0; f.total_bl_a  = 0.0; f.longest_bl_a  = 0.0;
    subtreeStats(move->removeNode, tr->mxtips,
                 &f.n_leaves_a, &f.total_bl_a, &f.longest_bl_a);

    /* subtree b: the complement side — rooted at removeNode->back */
    f.n_leaves_b  = 0; f.total_bl_b  = 0.0; f.longest_bl_b  = 0.0;
    subtreeStats(move->removeNode->back, tr->mxtips,
                 &f.n_leaves_b, &f.total_bl_b, &f.longest_bl_b);

    /* subtree b1: insertNode side */
    f.n_leaves_b1 = 0; f.total_bl_b1 = 0.0; f.longest_bl_b1 = 0.0;
    subtreeStats(move->insertNode, tr->mxtips,
                 &f.n_leaves_b1, &f.total_bl_b1, &f.longest_bl_b1);

    /* subtree b2: insertNode->back side */
    f.n_leaves_b2 = 0; f.total_bl_b2 = 0.0; f.longest_bl_b2 = 0.0;
    subtreeStats(move->insertNode->back, tr->mxtips,
                 &f.n_leaves_b2, &f.total_bl_b2, &f.longest_bl_b2);

    return f;
}

SPRFeatures * computeAllSPRFeatures(pllInstance *tr, partitionList *pr,
                                    int mintrav, int maxtrav, int maxMoves,
                                    int *count)
{
    /* Phase 1: serial collection (tree mutation is not thread-safe) */
    SPRMoveRecordList *list =
        collectAllSPRMoves(tr, pr, mintrav, maxtrav, maxMoves);

    *count = list->entries;
    SPRFeatures *result =
        (SPRFeatures *) malloc(list->entries * sizeof(SPRFeatures));

    /* Phase 2: compute global stats once (read-only, cheap) */
    double tree_total_bl, tree_longest_bl;
    treeGlobalStats(tr, &tree_total_bl, &tree_longest_bl);

    /* Phase 3: compute per-move features in parallel (all read-only) */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 32)
#endif
    for (int i = 0; i < list->entries; ++i)
        result[i] = computeSPRFeatures(tr, &list->moves[i],
                                       tree_total_bl, tree_longest_bl);

    destroySPRMoveRecordList(&list);
    return result;
}