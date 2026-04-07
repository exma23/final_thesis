#pragma once
#include "stack.hpp"

struct pllNewickTree {
  int nodes; /**< @brief Total number of nodes in the tree == 2*tips - 1 for
                rooted and 2*tips -2 for unrooted */
  int tips;  /**< @brief Number of leaves (tips) in the tree */
  pllStack *tree; /**< @brief Parsed tree represented as elements of a stack.
                     Corresponds to placing the postorder traversal of a rooted
                     tree in a pushdown store */
};

struct pllNewickNodeInfo {
  int depth;  /**< @brief Distance of node from root */
  char *name; /**< @brief Name of the taxon represented by the node (in case it
                 is a leaf) */
  char *branch; /**< @brief Length of branch that leads to its parent */
  int leaf;     /**< @brief \b PLL_TRUE if the node is a leaf, otherwise \b
                   PLL_FALSE */
  int rank;     /**< @brief Rank of the node, i.e. how many children it has */
};

extern pllNewickTree *pllNewickParseString(const char *newick);
extern void pllNewickParseDestroy(pllNewickTree **);