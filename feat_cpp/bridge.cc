#include "spr_move_eval.h"   // brings in phylotree.h, features.cc, SPRMoveEvaluator
#include <sstream>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Internal helpers (not exported)
// ---------------------------------------------------------------------------

// Load a PhyloTree directly from a Newick-format string (not a filename)
static void load_from_str(const std::string& newick, PhyloTree& tree) {
    size_t i = 0;
    tree.nodes.clear();
    tree.neighbors.clear();
    double root_len = 0.0;
    tree.root = parse_newick_rec(newick, i, tree.nodes, tree.neighbors, root_len);
}

// Serialise a PhyloTree back to a Newick string
static std::string to_newick_str(const PhyloTree& tree) {
    std::ostringstream oss;
    write_newick_rec(tree.root, oss);
    oss << ";";
    return oss.str();
}

// Lookup a node by its integer index
static Node* find_node(PhyloTree& tree, int idx) {
    for (Node* n : tree.nodes)
        if (n->index == idx) return n;
    return nullptr;
}

// ---------------------------------------------------------------------------
// Exported C interface
// ---------------------------------------------------------------------------

/*
 * get_state_action_c
 *
 * Given the current tree (newick_str) and a chosen SPR action (4 node indices),
 * applies the action, then enumerates all possible next SPR moves from the
 * resulting state and computes their feature vectors + RF rewards.
 *
 * Parameters
 * ----------
 * newick_str      : current tree in Newick format
 * action          : int[4] = [prune_dad_idx, prune_child_idx,
 *                              regraft_dad_idx, regraft_child_idx]
 *                   Pass [-1, -1, -1, -1] for the initial step (no action applied).
 * gt_newick_str   : ground-truth tree in Newick format (used for RF reward)
 * out_newick      : output buffer for the new tree Newick string
 * out_newick_cap  : capacity of out_newick (bytes)
 * out_actions     : output int[max_actions × 4]  — node indices of each move
 * out_feats       : output double[max_actions × 20] — feature vectors
 * out_rewards     : output double[max_actions]      — RF reward per move
 * out_n_actions   : output: number of valid moves written
 */
extern "C" void get_state_action_c(
    const char* newick_str,
    const int*  action,
    const char* gt_newick_str,
    char*       out_newick,
    int         out_newick_cap,
    int*        out_actions,
    double*     out_feats,
    double*     out_rewards,
    int*        out_n_actions
) {
    // 1. Load current tree from Newick string
    PhyloTree tree;
    load_from_str(std::string(newick_str), tree);

    // 2. Apply chosen action (skip if initial dummy action [-1,-1,-1,-1])
    if (action[0] >= 0) {
        Node* pd = find_node(tree, action[0]);
        Node* pn = find_node(tree, action[1]);
        Node* rd = find_node(tree, action[2]);
        Node* rn = find_node(tree, action[3]);
        if (pd && pn && rd && rn)
            tree.update_state(SPRMove(pd, pn, rd, rn));
    }

    // 3. Serialise resulting tree state back to out_newick
    std::string new_nwk = to_newick_str(tree);
    int copy_len = std::min((int)new_nwk.size(), out_newick_cap - 1);
    std::memcpy(out_newick, new_nwk.c_str(), copy_len);
    out_newick[copy_len] = '\0';

    // 4. Load ground-truth tree
    PhyloTree gt_tree;
    load_from_str(std::string(gt_newick_str), gt_tree);

    // 5. Enumerate moves + compute features + rewards
    std::vector<SPRMove>      moves   = tree.get_possible_SPR();
    std::vector<FeatureVector> feats;
    std::vector<double>        rewards;

    SPRMoveEvaluator evaluator(&tree);
    evaluator.batch_evaluate(moves, feats, rewards, gt_tree);

    *out_n_actions = (int)moves.size();

    // 6. Write outputs
    for (int i = 0; i < (int)moves.size(); ++i) {
        out_actions[i * 4 + 0] = moves[i].prune_dad->index;
        out_actions[i * 4 + 1] = moves[i].prune_node->index;
        out_actions[i * 4 + 2] = moves[i].regraft_dad->index;
        out_actions[i * 4 + 3] = moves[i].regraft_node->index;

        for (int j = 0; j < 20; ++j)
            out_feats[i * 20 + j] = feats[i].features[j];

        out_rewards[i] = rewards[i];
    }
}