#include "pll.hpp"
#include "mem_alloc.hpp"
#include "hash.hpp"
#include <cassert>

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <assert.h>
#include <errno.h>

static void pllTreeInitDefaults (pllInstance * tr, int tips)
{
  nodeptr p0, p, q;
  int i, j;
  int inner;



  /* TODO: make a proper static setupTree function */

  inner = tips - 1;

  tr->mxtips = tips;

  tr->bigCutoff = PLL_FALSE;
  tr->treeStringLength = tr->mxtips * (PLL_NMLNGTH + 128) + 256 + tr->mxtips * 2;
  tr->tree_string = (char *) rax_calloc ( tr->treeStringLength, sizeof(char));
  tr->tree0 = (char*)rax_calloc((size_t)tr->treeStringLength, sizeof(char));
  tr->tree1 = (char*)rax_calloc((size_t)tr->treeStringLength, sizeof(char));
  tr->constraintVector = (int *)rax_malloc((2 * tr->mxtips) * sizeof(int));

  p0 = (nodeptr) rax_malloc ((tips + 3 * inner) * sizeof (node));
  assert (p0);

  tr->nodeBaseAddress  = p0;

  tr->nameList         = (char **)   rax_malloc ((tips + 1) * sizeof (char *));
  tr->nodep            = (nodeptr *) rax_malloc ((2 * tips) * sizeof (nodeptr));

  tr->autoProteinSelectionType = PLL_AUTO_ML;

  assert (tr->nameList && tr->nodep);

  tr->nodep[0] = NULL;


  /* TODO: FIX THIS! */
  //tr->fracchange = -1;

  for (i = 1; i <= tips; ++ i)
   {
     p = p0++;

     //p->hash      = KISS32();
     p->x         = 0;
     p->xBips     = 0;
     p->number    = i;
     p->next      = p;
     p->back      = NULL;
     p->bInf      = NULL;
     tr->nodep[i]  = p;
   }

  for (i = tips + 1; i <= tips + inner; ++i)
   {
     q = NULL;
     for (j = 1; j <= 3; ++ j)
     {
       p = p0++;
       if (j == 1)
        {
          p->xBips = 1;
          p->x = 1; //p->x     = 1;
        }
       else
        {
          p->xBips = 0;
          p->x     = 0;
        }
       p->number = i;
       p->next   = q;
       p->bInf   = NULL;
       p->back   = NULL;
       p->hash   = 0;
       q         = p;
     }
    p->next->next->next = p;
    tr->nodep[i]         = p;
   }

  tr->likelihood  = PLL_UNLIKELY;
  tr->start       = NULL;
  tr->ntips       = 0;
  tr->nextnode    = 0;

  for (i = 0; i < PLL_NUM_BRANCHES; ++ i) tr->partitionSmoothed[i] = PLL_FALSE;

  tr->bitVectors = NULL;
  tr->vLength    = 0;
  //tr->h          = NULL;

  /* TODO: Fix hash type */
  tr->nameHash   = pllHashInit (10 * tr->mxtips);

  /* TODO: do these options really fit here or should they be put elsewhere? */
  tr->td[0].count            = 0;
  tr->td[0].ti               = (traversalInfo *) rax_malloc (sizeof(traversalInfo) * (size_t)tr->mxtips);
  tr->td[0].parameterValues  = (double *) rax_malloc(sizeof(double) * (size_t)PLL_NUM_BRANCHES);
  tr->td[0].executeModel     = (pllBoolean *) rax_malloc (sizeof(pllBoolean) * (size_t)PLL_NUM_BRANCHES);
  tr->td[0].executeModel[0]  = PLL_TRUE;
  for (i = 0; i < PLL_NUM_BRANCHES; ++ i) tr->td[0].executeModel[i] = PLL_TRUE;
}

static int linkTaxa (pllInstance * pInst, pllNewickTree * nTree, int taxaExist)
{
  nodeptr
    parent,
    child;
  pllStack
    * nodeStack = NULL,
    * current;
  int
    i,
    j,
    inner = nTree->tips + 1,
    leaf  = 1;
  double z;
  pllNewickNodeInfo * nodeInfo;

  if (!taxaExist) pllTreeInitDefaults (pInst, nTree->tips);

  /* Place the ternary root node 3 times on the stack such that later on
     three nodes use it as their parent */
  current = nTree->tree;
  for (parent = pInst->nodep[inner], i  = 0; i < 3; ++ i, parent = parent->next)
    pllStackPush (&nodeStack, parent);
  ++ inner;

  /* now traverse the rest of the nodes */
  for (current = current->next; current; current = current->next)
   {
     parent   = (nodeptr) pllStackPop (&nodeStack);
     nodeInfo = (pllNewickNodeInfo *) current->item;

     /* if inner node place it twice on the stack (out-degree 2) */
     if (nodeInfo->rank)
      {
        child = pInst->nodep[inner ++];
        pllStackPush (&nodeStack, child->next);
        pllStackPush (&nodeStack, child->next->next);
      }
     else /* check if taxon already exists, i.e. we loaded another tree topology */
      {
        if (taxaExist)
         {
           assert (pllHashSearch (pInst->nameHash, nodeInfo->name, (void **) &child));
         }
        else
         {
           child = pInst->nodep[leaf];
           pInst->nameList[leaf] = strdup (nodeInfo->name);
           pllHashAdd (pInst->nameHash, pllHashString(pInst->nameList[leaf], pInst->nameHash->size), pInst->nameList[leaf], (void *) (pInst->nodep[leaf]));
           ++ leaf;
         }
      }
     assert (parent);
     /* link parent and child */
     parent->back = child;
     child->back  = parent;

     if (!taxaExist) pInst->fracchange = 1;

     /* set the branch length */
     z = exp ((-1 * atof (nodeInfo->branch)) / pInst->fracchange);
     if (z < PLL_ZMIN) z = PLL_ZMIN;
     if (z > PLL_ZMAX) z = PLL_ZMAX;
     for (j = 0; j < PLL_NUM_BRANCHES; ++ j)
       parent->z[j] = child->z[j] = z;
   }
  pllStackClear (&nodeStack);

  return PLL_TRUE;
}

static int checkTreeInclusion (pllInstance * pInst, pllNewickTree * nTree)
{
  pllStack * sList;
  pllNewickNodeInfo * sItem;
  void * dummy;

  if (!pInst->nameHash) return (PLL_FALSE);

  for (sList = nTree->tree; sList; sList = sList->next)
   {
     sItem = (pllNewickNodeInfo *) sList->item;
     if (!sItem->rank)   /* leaf */
      {
        if (!pllHashSearch (pInst->nameHash, sItem->name, &dummy)) return (PLL_FALSE);
      }
   }

  return (PLL_TRUE);
}

void resetBranches(pllInstance *tr)
{
  nodeptr  p, q;
  int  nodes, i;

  nodes = tr->mxtips  +  3 * (tr->mxtips - 2);
  p = tr->nodep[1];
  while (nodes-- > 0)
    {
      for(i = 0; i < PLL_NUM_BRANCHES; i++)
        p->z[i] = PLL_DEFAULTZ;

      q = p->next;
      while(q != p)
        {
          for(i = 0; i < PLL_NUM_BRANCHES; i++)
            q->z[i] = PLL_DEFAULTZ;
          q = q->next;
        }
      p++;
    }
}

void pllTreeInitTopologyNewick (pllInstance * tr, pllNewickTree * newick, int useDefaultz)
{
  linkTaxa (tr, newick, tr->nameHash && checkTreeInclusion (tr, newick));

  tr->start = tr->nodep[1];

  if (useDefaultz == PLL_TRUE)
    resetBranches (tr);
}

void hookup (nodeptr p, nodeptr q, double *z, int numBranches)
{
  int i;

  p->back = q;
  q->back = p;

  for(i = 0; i < numBranches; i++)
    p->z[i] = q->z[i] = z[i];
}

void hookupDefault (nodeptr p, nodeptr q)
{
  int i;

  p->back = q;
  q->back = p;

  p->z[0] = q->z[0] = PLL_DEFAULTZ;

}