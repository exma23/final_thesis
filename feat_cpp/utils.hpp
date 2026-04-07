#pragma once
#include "pll.hpp"

void resetBranches(pllInstance *tr);
void pllTreeInitTopologyNewick(pllInstance *tr, pllNewickTree *newick,
                               int useDefaultz);
void hookup(nodeptr p, nodeptr q, double *z, int numBranches);
void hookupDefault(nodeptr p, nodeptr q);