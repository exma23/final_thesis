#pragma once
#include "pll.hpp"

extern pllBoolean isTip          (int number, int maxTips);
extern nodeptr    removeNodeBIG  (pllInstance *tr, partitionList *pr,
                                  nodeptr p, int numBranches);

void insertNodeBIG(nodeptr p, nodeptr q, int numBranches);
