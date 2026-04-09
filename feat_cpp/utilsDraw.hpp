#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct nodeDraw {
  std::string name;
  double len;
  std::vector<nodeDraw *> children;

  ~nodeDraw();
};

class treeDraw {
public:
  nodeDraw *root;

  treeDraw();
  ~treeDraw();

  // Parse Newick string: (A:0.1,B:0.2):0.5;
  void readNewick(const std::string &newick);
  void drawTree(nodeDraw *node = nullptr, const std::string &prefix = "", bool isLast = true);

  std::string toNewick(nodeDraw *node);

private:
  nodeDraw *parseNewick(const std::string &s, int &pos);
};