#pragma once
#include "phylotree.h"
#include "features.cc"
#include <vector>
#include <array>

struct SPRUndoRecord {
    // Prune side
    Node*  prune_dad;
    Node*  prune_node;
    double prune_length;
    bool   prune_was_left;   // was prune_node the left child of prune_dad?

    // Regraft side
    Node*  regraft_dad;
    Node*  regraft_node;
    double regraft_length;
    bool   regraft_was_left; // was regraft_node the left child of regraft_dad?

    // New node inserted during apply
    Node*     new_node;
    Neighbor* new_to_parent;    // new_node -> regraft_dad
    Neighbor* parent_to_new;    // regraft_dad -> new_node
    Neighbor* new_to_regraft;   // new_node -> regraft_node  (left child)
    Neighbor* regraft_to_new;   // regraft_node -> new_node  (parent)
    Neighbor* new_to_prune;     // new_node -> prune_node    (right child)
    Neighbor* prune_to_new;     // prune_node -> new_node    (parent)
    bool created;
};

class SPRMoveEvaluator {
public:
    explicit SPRMoveEvaluator(PhyloTree* tree);

    // Apply move in-place, fill undo record so it can be reversed
    void apply_move(const SPRMove& move, SPRUndoRecord& undo);

    // Undo a previously applied move
    void undo_move(const SPRUndoRecord& undo);

    // For each move: apply -> features -> reward -> undo
    void batch_evaluate(const std::vector<SPRMove>& moves,
                        std::vector<FeatureVector>&  features,
                        std::vector<double>&         rewards,
                        const PhyloTree&             gt_tree);

private:
    PhyloTree* tree_;
};

// -----------------------------------------------------------------------
// Single entry-point: given a tree in Newick format and a ground-truth
// Newick, compute all possible SPR moves, their feature vectors, and
// their RF-distance rewards.
//
//   out_moves    — all valid SPR moves from the current tree state
//   out_features — feature vector for each move  (same order as moves)
//   out_rewards  — RF distance after applying each move (same order)
// -----------------------------------------------------------------------
void evaluate_all_spr_moves(const std::string&         newick_str,
                             const std::string&         gt_newick_str,
                             std::vector<SPRMove>&      out_moves,
                             std::vector<FeatureVector>& out_features,
                             std::vector<double>&        out_rewards);