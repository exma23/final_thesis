#pragma once
#include "pll.hpp"
#include "searchAlgo.hpp"

/* Extended move record: pllRearrangeInfo + topo_depth recorded at collection time */
struct SPRMoveRecord
{
    nodeptr  removeNode;
    nodeptr  insertNode;
    int      topo_depth;      /* feat 5: exact edge count from prune to regraft */
    double   path_bl_sum;     /* feat 6: exact sum of branch lengths along path  */
};

struct SPRMoveRecordList
{
    int           entries;
    int           max_entries;
    SPRMoveRecord *moves;
};

struct SPRFeatures
{
    double total_branch_lengths;   /* feat  1 */
    double longest_branch;         /* feat  2 */
    double prune_branch_len;       /* feat  3 */
    double regraft_branch_len;     /* feat  4 */
    int    topo_distance;          /* feat  5 */
    double branch_len_distance;    /* feat  6 */
    double new_branch_len;         /* feat  7 */
    int    n_leaves_a;             /* feat  8 */
    int    n_leaves_b;             /* feat  9 */
    int    n_leaves_b1;            /* feat 10 */
    int    n_leaves_b2;            /* feat 11 */
    double total_bl_a;             /* feat 12 */
    double total_bl_b;             /* feat 13 */
    double total_bl_b1;            /* feat 14 */
    double total_bl_b2;            /* feat 15 */
    double longest_bl_a;           /* feat 16 */
    double longest_bl_b;           /* feat 17 */
    double longest_bl_b1;          /* feat 18 */
    double longest_bl_b2;          /* feat 19 */
};

/* Collect all SPR moves with exact topo_depth and path_bl_sum recorded.
   Caller must free result->moves and result itself. */
SPRMoveRecordList * collectAllSPRMoves(pllInstance *tr, partitionList *pr,
                                       int mintrav, int maxtrav, int maxMoves);

void destroySPRMoveRecordList(SPRMoveRecordList **list);

/* Compute features for a single already-collected move.
   tree_total_bl and tree_longest_bl are passed in to avoid recomputing per move. */
SPRFeatures computeSPRFeatures(pllInstance *tr,
                               const SPRMoveRecord *move,
                               double tree_total_bl,
                               double tree_longest_bl);

/* Compute features for all moves from the current tree state.
   Uses OpenMP for the per-move feature computation.
   *count is set to number of moves. Caller must free() the returned array. */
SPRFeatures * computeAllSPRFeatures(pllInstance *tr, partitionList *pr,
                                    int mintrav, int maxtrav, int maxMoves,
                                    int *count);