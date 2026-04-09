#include "utilsDraw.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

nodeDraw::~nodeDraw() {
  for (auto child : children) {
    delete child;
  }
  children.clear();
}

treeDraw::treeDraw() : root(nullptr) {}
treeDraw::~treeDraw() {
  if (this->root)
    delete root;
}

void treeDraw::readNewick(const std::string &newick) {
  std::string s = newick;
  s.erase(s.find_last_of(';'), 1); // remove ';'
  int pos = 0;
  root = parseNewick(s, pos);
}

void treeDraw::drawTree(nodeDraw *node, const std::string &prefix, bool isLast) {
  if (!node)
    node = root;
  if (!node)
    return;

  std::cout << prefix << (isLast ? "└── " : "├── ");
  std::cout << node->name;
  if (node->len > 0)
    std::cout << " :" << node->len;
  std::cout << std::endl;

  for (size_t i = 0; i < node->children.size(); i++) {
    std::string newPrefix = prefix + (isLast ? "    " : "│   ");
    drawTree(node->children[i], newPrefix, i == node->children.size() - 1);
  }
}

std::string treeDraw::toNewick(nodeDraw *node) {
  if (!node)
    node = root;
  if (!node)
    return "";

  if (node->children.empty()) {
    return node->name + ":" + std::to_string(node->len);
  }

  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < node->children.size(); i++) {
    if (i > 0)
      ss << ",";
    ss << toNewick(node->children[i]);
  }
  ss << ")" << node->name << ":" << node->len;
  return ss.str();
}
nodeDraw *treeDraw::parseNewick(const std::string &s, int &pos) {
  nodeDraw *node = new nodeDraw();
  node->len = 0;

  if (s[pos] == '(') {
    pos++; // skip '('
    while (s[pos] != ')') {
      if (s[pos] == ',') {
        pos++;
        continue;
      }
      node->children.push_back(parseNewick(s, pos));
    }
    pos++; // skip ')'
  }

  // Read name
  while (pos < s.length() && s[pos] != ':' && s[pos] != ',' && s[pos] != ')') {
    node->name += s[pos];
    pos++;
  }

  // Read branch length
  if (pos < s.length() && s[pos] == ':') {
    pos++; // skip ':'
    std::string len_str;
    while (pos < s.length() && s[pos] != ',' && s[pos] != ')') {
      len_str += s[pos];
      pos++;
    }
    node->len = stod(len_str);
  }

  return node;
}
