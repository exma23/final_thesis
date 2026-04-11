#include "hash.hpp"
#include "mem_alloc.hpp"
#include "newick.hpp"
#include "pll.hpp"
#include "treeIO.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static std::string slurp(const std::string &p) {
  std::ifstream f(p);
  std::string s((std::istreambuf_iterator<char>(f)), {});
  while (!s.empty() && s.back() <= ' ')
    s.pop_back();
  return s;
}

static void spit(const std::string &p, const std::string &s) {
  std::ofstream(p) << s;
}

static void freeTree(pllInstance *tr) {
  if (!tr)
    return;
  if (tr->nameList) {
    for (int i = 1; i <= tr->mxtips; i++)
      rax_free(tr->nameList[i]);
    rax_free(tr->nameList);
  }
  if (tr->nameHash)
    pllHashDestroy(&tr->nameHash, 0);
  rax_free(tr->nodeBaseAddress);
  rax_free(tr->nodep);
  rax_free(tr->tree_string);
  rax_free(tr->tree0);
  rax_free(tr->tree1);
  rax_free(tr);
}

static std::string unroot(const std::string &nwk) {
  pllNewickTree *nw = pllNewickParseString(nwk.c_str());
  if (!nw)
    return "";
  if (!pllValidateNewick(nw))
    pllNewickUnroot(nw);
  auto *tr = (pllInstance *)rax_calloc(1, sizeof(pllInstance));
  pllTreeInitTopologyNewick(tr, nw, PLL_FALSE);
  pllNewickParseDestroy(&nw);
  char buf[65536] = {};
  pllTreeToNewick(buf, tr, tr->start->back, PLL_TRUE, PLL_TRUE);
  freeTree(tr);
  std::string out(buf);
  while (!out.empty() && out.back() <= ' ')
    out.pop_back();
  return out;
}

static std::string rand_tree(int n, std::mt19937 &rng) {
  std::exponential_distribution<> bl(3.0);
  std::vector<std::string> v;
  for (int i = 1; i <= n; i++)
    v.push_back("t" + std::to_string(i));
  while (v.size() > 1) {
    int a = rng() % v.size(), b;
    do
      b = rng() % v.size();
    while (b == a);
    if (a > b)
      std::swap(a, b);
    std::ostringstream os;
    os << "(" << v[a] << ":" << bl(rng) << "," << v[b] << ":" << bl(rng)
       << ")";
    v.erase(v.begin() + b);
    v[a] = os.str();
  }
  return v[0] + ";";
}

int main(int argc, char **argv) {
  int taxa = 30, len = 500, num = 20;
  unsigned seed = time(nullptr);
  std::string outdir = "data", iqtree = "iqtree3", model = "GTR+I+G",
              raxmlng = "raxml-ng";

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--taxa" && i + 1 < argc)
      taxa = atoi(argv[++i]);
    else if (a == "--len" && i + 1 < argc)
      len = atoi(argv[++i]);
    else if (a == "--num" && i + 1 < argc)
      num = atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc)
      seed = atoi(argv[++i]);
    else if (a == "--outdir" && i + 1 < argc)
      outdir = argv[++i];
    else if (a == "--iqtree" && i + 1 < argc)
      iqtree = argv[++i];
    else if (a == "--model" && i + 1 < argc)
      model = argv[++i];
    else if (a == "--raxmlng" && i + 1 < argc)
      raxmlng = argv[++i];
  }

  system(("mkdir -p " + outdir).c_str());
  std::mt19937 rng(seed);

  for (int i = 0; i < num; i++) {
    // data/<i>/  — one folder per tree
    std::string d = outdir + "/" + std::to_string(i);
    system(("mkdir -p " + d).c_str());
    std::cout << "[" << i + 1 << "/" << num << "] " << d << "\n";

    // 1. GT tree: random rooted → unroot
    std::string gt = unroot(rand_tree(taxa, rng));
    if (gt.empty()) {
      std::cerr << "  unroot failed\n";
      continue;
    }
    spit(d + "/gt.newick", gt);

    // 2. MSA via iqtree3 --alisim
    //    alisim writes <prefix>.phy, so prefix = d + "/data"
    std::string prefix = d + "/data";
    std::string cmd = iqtree + " --alisim " + prefix + " -t " + d +
                      "/gt.newick -m " + model + " --length " +
                      std::to_string(len) + " --seed " + std::to_string(i) +
                      " 2>&1";
    if (system(cmd.c_str())) {
      std::cerr << "  alisim failed\n";
      continue;
    }

    // 3. Start tree: different random topo → optimize BL with raxml-ng
    std::string tmp_tree = d + "/tmp.tre";
    std::string opt_prefix = d + "/opt";
    spit(tmp_tree, unroot(rand_tree(taxa, rng)));
    cmd = raxmlng + " --evaluate --msa " + prefix + ".phy --tree " +
          tmp_tree + " --model " + model + " --prefix " + opt_prefix +
          " --threads 1 --force perf_threads 2>&1";
    if (system(cmd.c_str())) {
      std::cerr << "  BL opt failed\n";
      continue;
    }
    spit(d + "/start.newick", slurp(opt_prefix + ".raxml.bestTree"));

    // 4. Cleanup temp files
    for (auto &e :
         {"/tmp.tre", "/opt.raxml.bestTree", "/opt.raxml.bestModel",
          "/opt.raxml.log", "/opt.raxml.startTree", "/data.log"})
      remove((d + e).c_str());
  }
  std::cout << "done\n";
}