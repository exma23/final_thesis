#include "spr_move_eval.h"
#include "phylotree.h"
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// compute_rf_distance: counts symmetric difference of bipartitions.
// Simple O(n^2) implementation sufficient for moderate tree sizes.
static void collect_bipartitions(Node* node, std::vector<std::vector<int>>& bips,
                                  std::vector<int>& buf) {
    if (!node->left && !node->right) {
        buf.push_back(node->index);
        return;
    }
    size_t before = buf.size();
    if (node->left)  collect_bipartitions(node->left->node,  bips, buf);
    if (node->right) collect_bipartitions(node->right->node, bips, buf);
    // record the leaf set below this node as a bipartition
    std::vector<int> bip(buf.begin() + before, buf.end());
    std::sort(bip.begin(), bip.end());
    bips.push_back(bip);
}

static double compute_rf_distance(const PhyloTree& a, const PhyloTree& b) {
    std::vector<std::vector<int>> bips_a, bips_b;
    std::vector<int> buf;
    collect_bipartitions(a.root, bips_a, buf);
    buf.clear();
    collect_bipartitions(b.root, bips_b, buf);

    int shared = 0;
    for (auto& ba : bips_a)
        for (auto& bb : bips_b)
            if (ba == bb) { ++shared; break; }

    return static_cast<double>((int(bips_a.size()) - shared) +
                               (int(bips_b.size()) - shared));
}

// ---------------------------------------------------------------------------
// SPRMoveEvaluator
// ---------------------------------------------------------------------------

SPRMoveEvaluator::SPRMoveEvaluator(PhyloTree* tree) : tree_(tree) {}

void SPRMoveEvaluator::apply_move(const SPRMove& move, SPRUndoRecord& undo) {
    undo.created = false;

    undo.prune_dad    = move.prune_dad;
    undo.prune_node   = move.prune_node;
    undo.prune_length = move.prune_node->parent ? move.prune_node->parent->length : 0.0;
    undo.prune_was_left = (move.prune_dad->left && move.prune_dad->left->node == move.prune_node);

    undo.regraft_dad    = move.regraft_dad;
    undo.regraft_node   = move.regraft_node;
    undo.regraft_length = move.regraft_node->parent ? move.regraft_node->parent->length : 0.0;
    undo.regraft_was_left = (move.regraft_dad->left && move.regraft_dad->left->node == move.regraft_node);

    double prune_len   = undo.prune_length;
    double regraft_len = undo.regraft_length;

    Node* prune_dad    = move.prune_dad;
    Node* prune_node   = move.prune_node;
    Node* regraft_dad  = move.regraft_dad;
    Node* regraft_node = move.regraft_node;

    // Detach prune_node from prune_dad
    if (prune_dad->left  && prune_dad->left->node  == prune_node) prune_dad->left  = nullptr;
    if (prune_dad->right && prune_dad->right->node == prune_node) prune_dad->right = nullptr;
    prune_node->parent = nullptr;

    // Insert new internal node between regraft_dad and regraft_node
    Node* new_node = new Node(tree_->nodes.size());
    tree_->nodes.push_back(new_node);

    Neighbor* new_to_parent  = new Neighbor(regraft_dad,  regraft_len / 2.0);
    Neighbor* parent_to_new  = new Neighbor(new_node,     regraft_len / 2.0);
    new_node->parent = new_to_parent;
    if (regraft_dad->left  && regraft_dad->left->node  == regraft_node) regraft_dad->left  = parent_to_new;
    if (regraft_dad->right && regraft_dad->right->node == regraft_node) regraft_dad->right = parent_to_new;
    tree_->neighbors.push_back(new_to_parent);
    tree_->neighbors.push_back(parent_to_new);

    Neighbor* new_to_regraft  = new Neighbor(regraft_node, regraft_len / 2.0);
    Neighbor* regraft_to_new  = new Neighbor(new_node,     regraft_len / 2.0);
    new_node->left       = new_to_regraft;
    regraft_node->parent = regraft_to_new;
    tree_->neighbors.push_back(new_to_regraft);
    tree_->neighbors.push_back(regraft_to_new);

    Neighbor* new_to_prune = new Neighbor(prune_node, prune_len);
    Neighbor* prune_to_new = new Neighbor(new_node,   prune_len);
    new_node->right    = new_to_prune;
    prune_node->parent = prune_to_new;
    tree_->neighbors.push_back(new_to_prune);
    tree_->neighbors.push_back(prune_to_new);

    // Save new objects for undo
    undo.new_node       = new_node;
    undo.new_to_parent  = new_to_parent;
    undo.parent_to_new  = parent_to_new;
    undo.new_to_regraft = new_to_regraft;
    undo.regraft_to_new = regraft_to_new;
    undo.new_to_prune   = new_to_prune;
    undo.prune_to_new   = prune_to_new;
    undo.created        = true;
}

void SPRMoveEvaluator::undo_move(const SPRUndoRecord& undo) {
    if (!undo.created) return;

    Node* prune_dad    = undo.prune_dad;
    Node* prune_node   = undo.prune_node;
    Node* regraft_dad  = undo.regraft_dad;
    Node* regraft_node = undo.regraft_node;
    Node* new_node     = undo.new_node;

    // Detach prune_node from new_node
    prune_node->parent = nullptr;
    if (new_node->left  && new_node->left->node  == prune_node) new_node->left  = nullptr;
    if (new_node->right && new_node->right->node == prune_node) new_node->right = nullptr;

    // Detach new_node from regraft_dad and regraft_node
    if (regraft_dad->left  && regraft_dad->left->node  == new_node) regraft_dad->left  = nullptr;
    if (regraft_dad->right && regraft_dad->right->node == new_node) regraft_dad->right = nullptr;
    new_node->parent  = nullptr;
    new_node->left    = nullptr;
    new_node->right   = nullptr;
    regraft_node->parent = nullptr;

    // Restore regraft_node as child of regraft_dad
    Neighbor* restored_regraft = new Neighbor(regraft_node, undo.regraft_length);
    Neighbor* restored_regraft_parent = new Neighbor(regraft_dad, undo.regraft_length);
    if (undo.regraft_was_left) regraft_dad->left  = restored_regraft;
    else                       regraft_dad->right = restored_regraft;
    regraft_node->parent = restored_regraft_parent;
    tree_->neighbors.push_back(restored_regraft);
    tree_->neighbors.push_back(restored_regraft_parent);

    // Restore prune_node as child of prune_dad
    Neighbor* restored_prune = new Neighbor(prune_node, undo.prune_length);
    Neighbor* restored_prune_parent = new Neighbor(prune_dad, undo.prune_length);
    if (undo.prune_was_left) prune_dad->left  = restored_prune;
    else                     prune_dad->right = restored_prune;
    prune_node->parent = restored_prune_parent;
    tree_->neighbors.push_back(restored_prune);
    tree_->neighbors.push_back(restored_prune_parent);

    // Remove new_node from nodes list and remove its 6 Neighbor objects
    tree_->nodes.erase(std::remove(tree_->nodes.begin(), tree_->nodes.end(), new_node),
                       tree_->nodes.end());
    for (Neighbor* nei : {undo.new_to_parent, undo.parent_to_new,
                          undo.new_to_regraft, undo.regraft_to_new,
                          undo.new_to_prune, undo.prune_to_new}) {
        tree_->neighbors.erase(std::remove(tree_->neighbors.begin(), tree_->neighbors.end(), nei),
                               tree_->neighbors.end());
        delete nei;
    }
    delete new_node;
}

void SPRMoveEvaluator::batch_evaluate(const std::vector<SPRMove>& moves,
                                       std::vector<FeatureVector>&  features,
                                       std::vector<double>&         rewards,
                                       const PhyloTree&             gt_tree) {
    features.resize(moves.size());
    rewards.resize(moves.size());
    for (size_t i = 0; i < moves.size(); ++i) {
        SPRUndoRecord undo;
        apply_move(moves[i], undo);
        features[i] = compute_spr_features(*tree_, moves[i]);
        rewards[i]  = compute_rf_distance(*tree_, gt_tree);
        undo_move(undo);
    }
}

// ---------------------------------------------------------------------------
// Single public interface
// ---------------------------------------------------------------------------

void evaluate_all_spr_moves(const std::string&          newick_str,
                             const std::string&          gt_newick_str,
                             std::vector<SPRMove>&       out_moves,
                             std::vector<FeatureVector>& out_features,
                             std::vector<double>&        out_rewards) {
    PhyloTree tree;
    load_newick(newick_str, tree);

    PhyloTree gt_tree;
    load_newick(gt_newick_str, gt_tree);

    out_moves = tree.get_possible_SPR();

    SPRMoveEvaluator evaluator(&tree);
    evaluator.batch_evaluate(out_moves, out_features, out_rewards, gt_tree);
}