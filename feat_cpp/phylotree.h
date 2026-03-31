#pragma once
#include <vector>
#include <string>
#include <ostream>

struct Node;

struct Neighbor {
    Node*  node;
    double length;
    Neighbor(Node* n, double len) : node(n), length(len) {}
};

struct Node {
    int         index;
    std::string name;       // leaf taxon label
    Neighbor*   parent;     // nullptr for root
    Neighbor*   left;       // nullptr for leaf
    Neighbor*   right;      // nullptr for leaf
    int         bottomsize;
    int         topsize;
    Node(int idx) : index(idx), parent(nullptr), left(nullptr), right(nullptr), bottomsize(0), topsize(0) {}
};

struct SPRMove {
    Node*  prune_dad;
    Node*  prune_node;
    Node*  regraft_dad;
    Node*  regraft_node;
    double score;
    SPRMove(Node* pd, Node* pn, Node* rd, Node* rn, double s = 0.0)
        : prune_dad(pd), prune_node(pn), regraft_dad(rd), regraft_node(rn), score(s) {}
};

struct SPR_compare {
    bool operator()(const SPRMove& s1, const SPRMove& s2) const {
        return s1.score > s2.score;
    }
};

class PhyloTree {
public:
    Node* root;
    std::vector<Node*>     nodes;
    std::vector<Neighbor*> neighbors;  // owns all Neighbor objects for memory management

    PhyloTree(Node* root_ = nullptr) : root(root_) {}

    void collect_nodes(Node* node, std::vector<Node*>& out);
    bool is_descendant(Node* ancestor, Node* node);
    std::vector<SPRMove> get_possible_SPR();
    void update_state(const SPRMove& move);
};

std::string read_file(const std::string& filename);
void skip_ws(const std::string& s, size_t& i);
std::string parse_token(const std::string& s, size_t& i);
double parse_length(const std::string& s, size_t& i);
Node* parse_newick_rec(const std::string& s, size_t& i,
                       std::vector<Node*>& nodes, std::vector<Neighbor*>& neighbors,
                       double& out_length);
void write_newick_rec(Node* node, std::ostream& out);
void load_newick(const std::string& filename, PhyloTree& tree);
void save_newick(const std::string& filename, const PhyloTree& tree);