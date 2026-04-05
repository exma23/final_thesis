#pragma once
#include "stack.hpp"
#include "newick.hpp"

constexpr int PLL_NUM_BRANCHES = 1;

using hashNumberType = unsigned int;
using pllBoolean = int;

struct node;                   /* forward declaration */
using nodeptr = node *;

#define PLL_NMLNGTH                             256         /* number of characters in species name */
#define PLL_AUTO_ML   0
#define PLL_UNLIKELY                            -1.0E300    /* low likelihood for initialization */
#define PLL_TRUE                                1
#define PLL_FALSE                               0
#define PLL_ZMIN                                1.0E-15  /* max branch prop. to -log(PLL_ZMIN) (= 34) */
#define PLL_ZMAX                                (1.0 - 1.0E-6) /* min branch prop. to 1.0-zmax (= 1.0E-6) */
#define PLL_DEFAULTZ                            0.9         /* value of z assigned as starting point */
#define PLL_REARRANGE_SPR  0
#define PLL_REARRANGE_NNI  2

#ifdef __AVX
  #define PLL_BYTE_ALIGNMENT 32
#elif defined(__SSE3)
  #define PLL_BYTE_ALIGNMENT 16
#else
  #define PLL_BYTE_ALIGNMENT 1
#endif

struct branchInfo
{
  unsigned int *vector = nullptr;
  int support = 0;
  node *oP = nullptr;
  node *oQ = nullptr;
};

struct node
{
  branchInfo *bInf = nullptr;
  double z[PLL_NUM_BRANCHES]{};

  node *next = nullptr;
  node *back = nullptr;

  hashNumberType hash = 0;
  int support = 0;
  int number = 0;

  char x = 0;
  char xPars = 0;
  char xBips = 0;
};

struct pllHashItem
{
  void *data;
  char *str;
  pllHashItem *next = nullptr;
};

struct pllHashTable
{
  unsigned int size;
  struct pllHashItem **Items = nullptr;
  unsigned int entries;
};

struct recompVectors
{
  int numVectors;            /**< Number of inner vectors allocated in RAM*/
  int *iVector = nullptr;    /**< size: numVectors, stores node id || PLL_SLOT_UNUSED  */
  int *iNode = nullptr;      /**< size: inner nodes, stores slot id || PLL_NODE_UNPINNED */
  int *stlen = nullptr;      /**< Number of tips behind the current orientation of the indexed inner node (subtree size/cost) */
  int *unpinnable = nullptr; /**< size:numVectors , TRUE if we dont need the vector */
  int maxVectorsUsed;
  pllBoolean allSlotsBusy; /**< on if all slots contain an ancesctral node (the usual case after first full traversal) */
};

struct traversalInfo
{
  int tipCase; /**< Type of entry, must be PLL_TIP_TIP PLL_TIP_INNER or PLL_INNER_INNER */
  int pNumber; /**< should exist in some nodeptr p->number */
  int qNumber; /**< should exist in some nodeptr q->number */
  int rNumber; /**< should exist in some nodeptr r->number */
  double qz[PLL_NUM_BRANCHES];
  double rz[PLL_NUM_BRANCHES];
  /* recom */
  int slot_p; /**< In recomputation mode, the RAM slot index for likelihood vector of node p, otherwise unused */
  int slot_q; /**< In recomputation mode, the RAM slot index for likelihood vector of node q, otherwise unused */
  int slot_r; /**< In recomputation mode, the RAM slot index for likelihood vector of node r, otherwise unused */
  /* E recom */
};

struct traversalData
{
  traversalInfo *ti = nullptr; /**< list of traversal steps */
  int count;                   /**< number of traversal steps */
  int functionType;
  pllBoolean traversalHasChanged;
  pllBoolean *executeModel;
  double *parameterValues;
};

struct checkPointState
{

  int state;

  /*unsigned int vLength;*/
  double accumulatedTime;
  int rearrangementsMax;
  int rearrangementsMin;
  int thoroughIterations;
  int fastIterations;
  int mintrav;
  int maxtrav;
  int bestTrav;
  double startLH;
  double lh;
  double previousLh;
  double difference;
  double epsilon;
  pllBoolean impr;
  pllBoolean cutoff;

  double tr_startLH;
  double tr_endLH;
  double tr_likelihood;
  double tr_bestOfNode;
  double tr_lhCutoff;
  double tr_lhAVG;
  double tr_lhDEC;
  int tr_NumberOfCategories;
  int tr_itCount;
  int tr_doCutoff;
  int tr_thoroughInsertion;
  int tr_optimizeRateCategoryInvocations;

  /* prevent users from doing stupid things */

  int searchConvergenceCriterion;
  int rateHetModel;
  int maxCategories;
  int NumberOfModels;
  int numBranches;
  int originalCrunchedLength;
  int mxtips;
  char seq_file[1024];
};

struct pllInstance
{

  int *ti;

  /* recomp */
  recompVectors *rvec;       /**< this data structure tracks which vectors store which nodes */
  float maxMegabytesMemory;  /**< User says how many MB in main memory should be used */
  float vectorRecomFraction; /**< vectorRecomFraction ~= 0.8 * maxMegabytesMemory  */
  pllBoolean useRecom;       /**< ON if we apply recomputation of ancestral vectors*/

  pllBoolean fastScaling;
  pllBoolean saveMemory;
  int startingTree;
  long randomNumberSeed;

  double *lhs;    /**< Array to store per-site log likelihoods of \a originalCrunchedLength (compressed) sites */
  double *patrat; /**< rates per pattern */
  double *patratStored;
  int *rateCategory;
  int *aliaswgt; /**< weight by pattern */
  pllBoolean manyPartitions;

  int threadID;
  volatile int numberOfThreads;

  // #if (defined(_USE_PTHREADS) || defined(_FINE_GRAIN_MPI))

  unsigned char *y_ptr;

  double lower_spacing;
  double upper_spacing;

  double *ancestralVector;

  // #endif
  pllHashTable  *nameHash;
  char **tipNames;

  char *secondaryStructureInput;

  traversalData td[1];

  int maxCategories;
  int categories;

  double coreLZ[PLL_NUM_BRANCHES];

  branchInfo *bInf;

  int multiStateModel;

  pllBoolean curvatOK[PLL_NUM_BRANCHES];

  /* the stuff below is shared among DNA and AA, span does
     not change depending on datatype */

  /* model stuff end */
  unsigned char **yVector; /**< list of raw sequences (parsed from the alignment)*/

  int secondaryStructureModel;
  int originalCrunchedLength; /**< Length of alignment after removing duplicate sites in each partition */

  int *secondaryStructurePairs;

  double fracchange; /**< Average substitution rate */
  double lhCutoff;
  double lhAVG;
  unsigned long lhDEC;
  unsigned long itCount;
  int numberOfInvariableColumns;
  int weightOfInvariableColumns;
  int rateHetModel;

  double startLH;
  double endLH;
  double likelihood; /**< last likelihood value evaluated for the current topology */

  node **nodep; /**< pointer to the list of nodes, which describe the current topology */
  nodeptr nodeBaseAddress;
  node *start; /**< starting node by default for full traversals (must be a tip contained in the tree we are operating on) */
  int mxtips;  /**< Number of tips in the topology */

  int *constraintVector; /**< @todo What is this? */
  int numberOfSecondaryColumns;
  pllBoolean searchConvergenceCriterion;
  int ntips;
  int nextnode;

  pllBoolean bigCutoff;
  pllBoolean partitionSmoothed[PLL_NUM_BRANCHES];
  pllBoolean partitionConverged[PLL_NUM_BRANCHES];
  pllBoolean rooted;
  pllBoolean doCutoff;

  double gapyness;

  char **nameList;   /**< list of tips names (read from the phylip file) */
  char *tree_string; /**< the newick representaion of the topology */
  char *tree0;
  char *tree1;
  int treeStringLength;

  unsigned int bestParsimony;
  unsigned int *parsimonyScore;

  double bestOfNode;
  nodeptr removeNode; /**< the node that has been removed. Together with \a insertNode represents an SPR move */
  nodeptr insertNode; /**< the node where insertion should take place . Together with \a removeNode represents an SPR move*/

  double zqr[PLL_NUM_BRANCHES];
  double currentZQR[PLL_NUM_BRANCHES];

  double currentLZR[PLL_NUM_BRANCHES];
  double currentLZQ[PLL_NUM_BRANCHES];
  double currentLZS[PLL_NUM_BRANCHES];
  double currentLZI[PLL_NUM_BRANCHES];
  double lzs[PLL_NUM_BRANCHES];
  double lzq[PLL_NUM_BRANCHES];
  double lzr[PLL_NUM_BRANCHES];
  double lzi[PLL_NUM_BRANCHES];

  unsigned int **bitVectors;

  unsigned int vLength;

  pllHashTable *h; /**< hashtable for ML convergence criterion */
  // hashtable *h;

  int optimizeRateCategoryInvocations;

  checkPointState ckp;
  pllBoolean thoroughInsertion; /**< true if the neighbor branches should be optimized when a subtree is inserted (slower)*/
  pllBoolean useMedian;

  int autoProteinSelectionType;

  pllStack *rearrangeHistory;

  /* analdef defines */
  /* TODO: Do some initialization */
  int bestTrav;          /**< best rearrangement radius */
  int max_rearrange;     /**< max. rearrangemenent radius */
  int stepwidth;         /**< step in rearrangement radius */
  int initial;           /**< user defined rearrangement radius which also sets bestTrav if initialSet is set */
  pllBoolean initialSet; /**< set bestTrav according to initial */
  int mode;              /**< candidate for removal */
  pllBoolean perGeneBranchLengths;
  pllBoolean permuteTreeoptimize; /**< randomly select subtrees for SPR moves */
  pllBoolean compressPatterns;
  double likelihoodEpsilon;
  pllBoolean useCheckpoint;
};

struct pllRearrangeInfo
{
  int rearrangeType;
  double likelihood;
  union
  {
    struct
    {
      nodeptr removeNode;
      nodeptr insertNode;
      double zqr[PLL_NUM_BRANCHES];
    } SPR;
    struct
    {
      nodeptr originNode;
      int swapType;
    } NNI;
  };
};

struct pllRearrangeList
{
  int max_entries;
  int entries;
  pllRearrangeInfo *rearr;
};

struct pInfo;
struct linkageList;

struct partitionList
{
  pInfo        **partitionData;
  int            numberOfPartitions;
  pllBoolean     perGeneBranchLengths;
  pllBoolean     dirty;
  linkageList   *alphaList;
  linkageList   *rateList;
  linkageList   *freqList;
};

pllRearrangeList * pllCreateRearrangeList  (int max);
void               pllDestroyRearrangeList (pllRearrangeList **bestList);