#include "phylotree.h"
#include <fstream>
#include <sstream>
#include <cctype>
#include <iomanip>
#include <stdexcept>

std::string read_file(const std::string& filename) {
    std::ifstream in(filename);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(s[i])) ++i;
}

std::string parse_token(const std::string& s, size_t& i) {
    skip_ws(s, i);
    size_t start = i;
    while (i < s.size() && (std::isalnum(s[i]) || s[i] == '_' || s[i] == '.' || s[i] == '-')) ++i;
    return s.substr(start, i - start);
}

double parse_length(const std::string& s, size_t& i) {
    skip_ws(s, i);
    size_t start = i;
    while (i < s.size() && (std::isdigit(s[i]) || s[i] == '.' || s[i] == 'e' ||
                             s[i] == 'E' || s[i] == '-' || s[i] == '+')) ++i;
    return std::stod(s.substr(start, i - start));
}

Node* parse_newick_rec(const std::string& s, size_t& i,
                       std::vector<Node*>& nodes, std::vector<Neighbor*>& neighbors,
                       double& out_length) {
    skip_ws(s, i);
    Node* node = new Node(nodes.size());
    nodes.push_back(node);

    if (s[i] == '(') {
        ++i;

        double left_len = 0.0;
        Node* left = parse_newick_rec(s, i, nodes, neighbors, left_len);

        skip_ws(s, i);
        if (s[i] != ',') throw std::runtime_error("Expected ',' in Newick");
        ++i;

        double right_len = 0.0;
        Node* right = parse_newick_rec(s, i, nodes, neighbors, right_len);

        skip_ws(s, i);
        if (s[i] != ')') throw std::runtime_error("Expected ')' in Newick");
        ++i;

        // node → left child (and back)
        Neighbor* to_left       = new Neighbor(left,  left_len);
        Neighbor* left_to_node  = new Neighbor(node,  left_len);
        node->left   = to_left;
        left->parent = left_to_node;
        neighbors.push_back(to_left);
        neighbors.push_back(left_to_node);

        // node → right child (and back)
        Neighbor* to_right      = new Neighbor(right, right_len);
        Neighbor* right_to_node = new Neighbor(node,  right_len);
        node->right   = to_right;
        right->parent = right_to_node;
        neighbors.push_back(to_right);
        neighbors.push_back(right_to_node);

    } else {
        node->name = parse_token(s, i);
    }

    // Parse branch length of the edge from this node to its parent
    skip_ws(s, i);
    out_length = 0.0;
    if (i < s.size() && s[i] == ':') {
        ++i;
        out_length = parse_length(s, i);
    }

    return node;
}

void load_newick(const std::string& filename, PhyloTree& tree) {
    std::string s = read_file(filename);
    size_t i = 0;
    tree.nodes.clear();
    tree.neighbors.clear();
    double root_len = 0.0;
    tree.root = parse_newick_rec(s, i, tree.nodes, tree.neighbors, root_len);
}

void write_newick_rec(Node* node, std::ostream& out) {
    if (node->left && node->right) {
        out << "(";
        write_newick_rec(node->left->node, out);
        out << ",";
        write_newick_rec(node->right->node, out);
        out << ")";
    } else {
        out << node->name;
    }
    if (node->parent) {
        out << ":" << std::fixed << std::setprecision(6) << node->parent->length;
    }
}

void save_newick(const std::string& filename, const PhyloTree& tree) {
    std::ofstream out(filename);
    write_newick_rec(tree.root, out);
    out << ";" << std::endl;
}