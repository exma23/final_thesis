#include "features.h"
#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>


static int count_leaves(Node* node) {
    if (!node->left && !node->right) return 1;
    int cnt = 0;
    if (node->left)  cnt += count_leaves(node->left->node);
    if (node->right) cnt += count_leaves(node->right->node);
    return cnt;
}

static double sum_branch_lengths(Node* node) {
    double sum = 0.0;
    if (node->left)  sum += node->left->length  + sum_branch_lengths(node->left->node);
    if (node->right) sum += node->right->length + sum_branch_lengths(node->right->node);
    return sum;
}

static double max_branch_length(Node* node) {
    double maxlen = 0.0;
    if (node->left) {
        maxlen = std::max(maxlen, node->left->length);
        maxlen = std::max(maxlen, max_branch_length(node->left->node));
    }
    if (node->right) {
        maxlen = std::max(maxlen, node->right->length);
        maxlen = std::max(maxlen, max_branch_length(node->right->node));
    }
    return maxlen;
}

static double total_branch_length(const PhyloTree& tree) {
    double sum = 0.0;
    for (const auto* n : tree.nodes)
        if (n->parent) sum += n->parent->length;
    return sum;
}

static double longest_branch(const PhyloTree& tree) {
    double maxlen = 0.0;
    for (const auto* n : tree.nodes)
        if (n->parent) maxlen = std::max(maxlen, n->parent->length);
    return maxlen;
}

static std::pair<int, double> path_info(Node* from, Node* to) {
    std::vector<Node*> ancestors;
    Node* cur = from;
    while (cur) {
        ancestors.push_back(cur);
        if (!cur->parent) break;
        cur = cur->parent->node;
    }
    Node* cur2 = to;
    std::vector<Node*> path2;
    while (cur2) {
        path2.push_back(cur2);
        if (!cur2->parent) break;
        cur2 = cur2->parent->node;
    }
    int i = ancestors.size() - 1, j = path2.size() - 1;
    while (i >= 0 && j >= 0 && ancestors[i] == path2[j]) { i--; j--; }
    int n_branch = (i+1) + (j+1);
    double branch_sum = 0.0;
    for (int k = 0; k <= i; ++k)
        if (ancestors[k]->parent) branch_sum += ancestors[k]->parent->length;
    for (int k = 0; k <= j; ++k)
        if (path2[k]->parent) branch_sum += path2[k]->parent->length;
    return {n_branch, branch_sum};
}

FeatureVector compute_spr_features(const PhyloTree& tree, const SPRMove& move) {
    FeatureVector fv = {};

    fv.features[0] = total_branch_length(tree);
    fv.features[1] = longest_branch(tree);

    double prune_len   = move.prune_node->parent   ? move.prune_node->parent->length   : 0.0;
    double regraft_len = move.regraft_node->parent ? move.regraft_node->parent->length : 0.0;

    fv.features[2] = prune_len;
    fv.features[3] = regraft_len;

    std::pair<int, double> path = path_info(move.prune_node, move.regraft_node);
    fv.features[4] = path.first;
    fv.features[5] = path.second;

    fv.features[6] = regraft_len / 2.0;

    Node* a  = move.prune_node;
    Node* b  = move.regraft_node;
    Node* b1 = b && b->left  ? b->left->node  : nullptr;
    Node* b2 = b && b->right ? b->right->node : nullptr;

    fv.features[7]  = a  ? count_leaves(a)  : 0;
    fv.features[8]  = b  ? count_leaves(b)  : 0;
    fv.features[9]  = b1 ? count_leaves(b1) : 0;
    fv.features[10] = b2 ? count_leaves(b2) : 0;

    fv.features[11] = a  ? sum_branch_lengths(a)  : 0.0;
    fv.features[12] = b  ? sum_branch_lengths(b)  : 0.0;
    fv.features[13] = b1 ? sum_branch_lengths(b1) : 0.0;
    fv.features[14] = b2 ? sum_branch_lengths(b2) : 0.0;

    fv.features[15] = a  ? max_branch_length(a)  : 0.0;
    fv.features[16] = b  ? max_branch_length(b)  : 0.0;
    fv.features[17] = b1 ? max_branch_length(b1) : 0.0;
    fv.features[18] = b2 ? max_branch_length(b2) : 0.0;

    fv.features[19] = 0.0;

    return fv;
}