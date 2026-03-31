#include "phylotree.h"
#include <algorithm>

void PhyloTree::collect_nodes(Node* node, std::vector<Node*>& out) {
    if (!node) return;
    out.push_back(node);
    if (node->left)  collect_nodes(node->left->node,  out);
    if (node->right) collect_nodes(node->right->node, out);
}

bool PhyloTree::is_descendant(Node* ancestor, Node* node) {
    while (node) {
        if (node == ancestor) return true;
        if (!node->parent) break;
        node = node->parent->node;
    }
    return false;
}

std::vector<SPRMove> PhyloTree::get_possible_SPR() {
    std::vector<SPRMove> moves;
    for (Node* pn : nodes) {
        if (!pn->parent) continue;         // skip root
        Node* pd = pn->parent->node;
        for (Node* rn : nodes) {
            if (!rn->parent) continue;     // skip root
            Node* rd = rn->parent->node;
            if (rn == pn) continue;
            if (is_descendant(pn, rn)) continue;
            moves.emplace_back(pd, pn, rd, rn);
        }
    }
    return moves;
}

void PhyloTree::update_state(const SPRMove& move) {
    Node* prune_dad    = move.prune_dad;
    Node* prune_node   = move.prune_node;
    Node* regraft_dad  = move.regraft_dad;
    Node* regraft_node = move.regraft_node;

    double prune_len   = prune_node->parent   ? prune_node->parent->length   : 0.0;
    double regraft_len = regraft_node->parent ? regraft_node->parent->length : 0.0;

    // Detach prune_node from prune_dad
    if (prune_dad->left  && prune_dad->left->node  == prune_node) prune_dad->left  = nullptr;
    if (prune_dad->right && prune_dad->right->node == prune_node) prune_dad->right = nullptr;
    prune_node->parent = nullptr;

    // Insert new internal node between regraft_dad and regraft_node
    Node* new_node = new Node(nodes.size());
    nodes.push_back(new_node);

    // new_node ↔ regraft_dad
    Neighbor* new_to_parent   = new Neighbor(regraft_dad, regraft_len / 2.0);
    Neighbor* parent_to_new   = new Neighbor(new_node,    regraft_len / 2.0);
    new_node->parent = new_to_parent;
    if (regraft_dad->left  && regraft_dad->left->node  == regraft_node) regraft_dad->left  = parent_to_new;
    if (regraft_dad->right && regraft_dad->right->node == regraft_node) regraft_dad->right = parent_to_new;
    neighbors.push_back(new_to_parent);
    neighbors.push_back(parent_to_new);

    // new_node ↔ regraft_node (left child)
    Neighbor* new_to_regraft  = new Neighbor(regraft_node, regraft_len / 2.0);
    Neighbor* regraft_to_new  = new Neighbor(new_node,     regraft_len / 2.0);
    new_node->left        = new_to_regraft;
    regraft_node->parent  = regraft_to_new;
    neighbors.push_back(new_to_regraft);
    neighbors.push_back(regraft_to_new);

    // new_node ↔ prune_node (right child)
    Neighbor* new_to_prune    = new Neighbor(prune_node, prune_len);
    Neighbor* prune_to_new    = new Neighbor(new_node,   prune_len);
    new_node->right      = new_to_prune;
    prune_node->parent   = prune_to_new;
    neighbors.push_back(new_to_prune);
    neighbors.push_back(prune_to_new);
}