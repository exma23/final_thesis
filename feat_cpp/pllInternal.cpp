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

nodeptr removeNodeBIG(pllInstance *tr, partitionList * /*pr*/, nodeptr p, int numBranches)
{
    nodeptr q = p->next->back;
    nodeptr r = p->next->next->back;

    /* estimate new branch: product of the two broken branches */
    double zqr[PLL_NUM_BRANCHES];
    for (int i = 0; i < numBranches; ++i)
        zqr[i] = q->z[i] * r->z[i];

    hookup(q, r, zqr, numBranches);

    p->next->next->back = p->next->back = nullptr;
    return q;
}

pllBoolean isTip(int number, int maxTips)
{
    return (number > 0 && number <= maxTips) ? PLL_TRUE : PLL_FALSE;
}

void insertNodeBIG(nodeptr p, nodeptr q, int numBranches)
{
    nodeptr r = q->back;

    /* estimate three new branches from the original edge q<->r */
    double zq[PLL_NUM_BRANCHES], zr[PLL_NUM_BRANCHES], zs[PLL_NUM_BRANCHES];
    for (int i = 0; i < numBranches; ++i)
    {
        double origz = q->z[i];
        /* split the original branch roughly in half (in log-space) */
        double sqz = (origz > PLL_ZMIN) ? sqrt(origz) : PLL_ZMIN;
        zq[i] = sqz;
        zr[i] = sqz;
        zs[i] = PLL_DEFAULTZ;
    }
    hookup(p->next,       q, zq, numBranches);
    hookup(p->next->next, r, zr, numBranches);
    /* p->back stay as is (the pruned subtree root) */
    /* actually p->back should already be set; we just need to fix z */
    for (int i = 0; i < numBranches; ++i)
        p->z[i] = p->back->z[i] = zs[i];
}