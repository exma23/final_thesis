#pragma once
#include "pll.hpp"

char *pllTreeToNewick(char *treestr, pllInstance *tr, nodeptr p,
                      pllBoolean printBranchLengths, pllBoolean printNames);