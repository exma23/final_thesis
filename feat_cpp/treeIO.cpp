#include "treeIO.hpp"
#include "pll.hpp"
#include "pllInternal.hpp"
#include <cmath>
#include <cstdio>

static char *pllTreeToNewickREC(char *treestr, pllInstance *tr, nodeptr p,
                                pllBoolean printBranchLengths,
                                pllBoolean printNames) {
  if (isTip(p->number, tr->mxtips)) {
    if (printNames && tr->nameList)
      sprintf(treestr, "%s", tr->nameList[p->number]);
    else
      sprintf(treestr, "%d", p->number);
    while (*treestr)
      treestr++;
  } else {
    *treestr++ = '(';
    treestr = pllTreeToNewickREC(treestr, tr, p->next->back, printBranchLengths,
                                 printNames);
    *treestr++ = ',';
    treestr = pllTreeToNewickREC(treestr, tr, p->next->next->back,
                                 printBranchLengths, printNames);
    if (p == tr->start->back) {
      *treestr++ = ',';
      treestr = pllTreeToNewickREC(treestr, tr, p->back, printBranchLengths,
                                   printNames);
    }
    *treestr++ = ')';
  }

  if (p == tr->start->back) {
    if (printBranchLengths)
      sprintf(treestr, ":0.0;\n");
    else
      sprintf(treestr, ";\n");
  } else {
    if (printBranchLengths) {
      double z = p->z[0];
      if (z < PLL_ZMIN)
        z = PLL_ZMIN;
      double bl = -log(z);
      sprintf(treestr, ":%8.20f", bl);
    }
  }

  while (*treestr)
    treestr++;
  return treestr;
}

char *pllTreeToNewick(char *treestr, pllInstance *tr, nodeptr p,
                      pllBoolean printBranchLengths, pllBoolean printNames) {
  pllTreeToNewickREC(treestr, tr, p, printBranchLengths, printNames);
  while (*treestr)
    treestr++;
  return treestr;
}