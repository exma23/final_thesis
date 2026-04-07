#pragma once
#include "pll.hpp"

extern pllBoolean isTip(int number, int maxTips);
extern nodeptr removeNodeBIG(pllInstance *tr, partitionList *pr, nodeptr p,
                             int numBranches);

extern pllBoolean insertBIG(pllInstance *tr, partitionList *pr, nodeptr p,
                            nodeptr q);
extern pllBoolean insertRestoreBIG(pllInstance *tr, partitionList *pr,
                                   nodeptr p, nodeptr q);
extern pllBoolean testInsertBIG(pllInstance *tr, partitionList *pr, nodeptr p,
                                nodeptr q);

extern unsigned int **initBitVector(int mxtips, unsigned int *vectorLength);
extern void bitVectorInitravSpecial(unsigned int **bitVectors, nodeptr p,
                                    int numsp, unsigned int vectorLength,
                                    pllHashTable *h, int treeNumber,
                                    int function, branchInfo *bInf,
                                    int *countBranches, int treeVectorLength,
                                    pllBoolean traverseOnly,
                                    pllBoolean computeWRF, int processID);
extern double convergenceCriterion(pllHashTable *h, int mxtips);
