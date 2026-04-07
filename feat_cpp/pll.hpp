#pragma once
#include "newick.hpp"
#include "stack.hpp"

constexpr int PLL_NUM_BRANCHES = 1;

using hashNumberType = unsigned int;
using pllBoolean = int;

struct node; /* forward declaration */
using nodeptr = node *;

#define PLL_NMLNGTH 256 /* number of characters in species name */
#define PLL_TRUE 1
#define PLL_FALSE 0
#define PLL_ZMIN 1.0E-15        /* max branch prop. to -log(PLL_ZMIN) (= 34) */
#define PLL_ZMAX (1.0 - 1.0E-6) /* min branch prop. to 1.0-zmax (= 1.0E-6) */
#define PLL_DEFAULTZ 0.9        /* value of z assigned as starting point */
#define PLL_REARRANGE_SPR 0
#define PLL_REARRANGE_NNI 2
#define PLL_MASK_LENGTH 32
#define PLL_BIPARTITIONS_RF 4
#define PLL_BYTE_ALIGNMENT 1

struct branchInfo {
  unsigned int *vector;
  int support;
  struct node *oP;
  struct node *oQ;
};
struct node {
  // do dai nhanh
  double z[PLL_NUM_BRANCHES]{};

  node *next = nullptr;
  node *back = nullptr;

  // number = ID global; hash = hash bipartition cho lookup;
  hashNumberType hash = 0;
  int number = 0;

  char x = 0;
  char xBips = 0;
};

struct pllHashItem {
  void *data;
  char *str;
  pllHashItem *next = nullptr;
};

struct pllHashTable {
  unsigned int size;
  struct pllHashItem **Items = nullptr;
  unsigned int entries;
};

struct pllInstance {
  // === Topo ===
  node **nodep; /**< pointer to the list of nodes, which describe the current
                   topology */
  nodeptr nodeBaseAddress;
  node *start; /**< starting node by default for full traversals (must be a tip
                  contained in the tree we are operating on) */
  int mxtips;  /**< Number of tips in the topology */
  int ntips;

  // === Info ===
  pllHashTable *nameHash; // hash table de lookup nhanh ten species
  char **nameList;        /**< list of tips names (read from the phylip file) */
  char *tree_string;      /**< the newick representaion of the topology */
  char *tree0;            // last best
  char *tree1;            // best
  int treeStringLength;

  // === Parameters ===
  double fracchange; /**< Average substitution rate */

  // === Rearrangement ===
  nodeptr removeNode; /**< the node that has been removed. Together with \a
                         insertNode represents an SPR move */
  nodeptr insertNode; /**< the node where insertion should take place . Together
                         with \a removeNode represents an SPR move*/

  /* analdef defines */
  /* TODO: Do some initialization */
  int max_rearrange; /**< max. rearrangemenent radius */
  int stepwidth;     /**< step in rearrangement radius */
  int mode;          /**< candidate for removal */
};

struct pllRearrangeInfo {
  int rearrangeType;
  double likelihood;
  union {
    struct {
      nodeptr removeNode;
      nodeptr insertNode;
      double zqr[PLL_NUM_BRANCHES];
    } SPR;
    struct {
      nodeptr originNode;
      int swapType;
    } NNI;
  };
};

struct pllRearrangeList {
  int max_entries;
  int entries;
  pllRearrangeInfo *rearr;
};

struct partitionList {
  int numberOfPartitions;
  pllBoolean perGeneBranchLengths;
};

struct pllBipartitionEntry {
  unsigned int *bitVector;
  unsigned int *treeVector;
  unsigned int amountTips;
  int *supportVector;
  unsigned int bipNumber;
  unsigned int bipNumber2;
  unsigned int supportFromTreeset[2];
  struct pllBipartitionEntry *next;
};

pllRearrangeList *pllCreateRearrangeList(int max);
void pllDestroyRearrangeList(pllRearrangeList **bestList);