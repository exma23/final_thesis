// feat_cpp/generate.cpp
// Minimal: generate random trees → iqtree alisim → infer starting tree → unroot
// all.
#include "hash.hpp"
#include "mem_alloc.hpp"
#include "newick.hpp"
#include "pll.hpp"
#include "treeIO.hpp"
#include "utils.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

static std::string read_file(const std::string &path) {
  std::ifstream f(path);
  std::string s((std::istreambuf_iterator<char>(f)), {});
  while (!s.empty() && s.back() <= ' ')
    s.pop_back();
  return s;
}

static void write_file(const std::string &path, const std::string &s) {
  std::ofstream(path) << s;
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

// ── unroot: rooted binary → unrooted ternary ─────────────

static std::string unroot(const std::string &newick) {
  pllNewickTree *nw = pllNewickParseString(newick.c_str());
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
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
    out.pop_back();
  return out;
}

// ── random tree (Yule process) ──────────────────────────

static double rand_bl(std::mt19937 &rng) {
  double lam =
      std::exp(std::uniform_real_distribution<>(std::log(2), std::log(5))(rng));
  return std::exponential_distribution<>(lam)(rng);
}

static std::string rand_tree(int n, std::mt19937 &rng) {
  std::vector<std::string> v;
  for (int i = 0; i < n; i++)
    v.push_back("t" + std::to_string(i + 1));

  while (v.size() > 1) {
    int a = std::uniform_int_distribution<>(0, v.size() - 1)(rng);
    int b;
    do {
      b = std::uniform_int_distribution<>(0, v.size() - 1)(rng);
    } while (b == a);
    if (a > b)
      std::swap(a, b);

    std::ostringstream os;
    os << "(" << v[a] << ":" << rand_bl(rng) << "," << v[b] << ":"
       << rand_bl(rng) << ")";
    v.erase(v.begin() + b);
    v[a] = os.str();
  }
  return v[0] + ";";
}

// ── main ─────────────────────────────────────────────────

int main(int argc, char **argv) {
  int taxa = 30, len = 500, num = 20;
  unsigned seed = time(nullptr);
  std::string outdir = "data",
              iqtree = "/home/ha/miniconda3/envs/thesis/bin/iqtree3",
              model = "GTR+I+G";
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
    else {
      std::cerr << "Usage: " << argv[0]
                << " [--taxa N] [--len L] [--num N] [--seed S] [--outdir D] "
                   "[--iqtree P] [--model M]\n";
      return 1;
    }
  }

  system(("mkdir -p " + outdir).c_str());
  std::mt19937 rng(seed);

  for (int i = 0; i < num; i++) {
    std::string base = outdir + "/" + std::to_string(i);
    std::cout << "[" << i + 1 << "/" << num << "] " << base << "\n";

    // 1. random rooted tree → temp file for iqtree
    std::string rooted = rand_tree(taxa, rng);
    write_file(base + "_raw.tre", rooted);

    // 2. unroot → ground truth
    std::string gt = unroot(rooted);
    if (gt.empty()) {
      std::cerr << "  unroot failed\n";
      continue;
    }
    write_file(base + "_gt.newick", gt);

    // 3. iqtree --alisim (simulate alignment)
    std::string cmd = iqtree + " --alisim " + base + " -m " + model + " -t " +
                      base + "_raw.tre --length " + std::to_string(len) +
                      " --no-unaligned --seed " + std::to_string(i) + " 2>&1";
    if (system(cmd.c_str())) {
      std::cerr << "  alisim failed\n";
      continue;
    }

    // 4. iqtree infer starting tree
    cmd = iqtree + " -s " + base + ".phy -m " + model + " --prefix " + base +
          "_infer -fast --seed " + std::to_string(i) + " 2>&1";
    if (system(cmd.c_str())) {
      std::cerr << "  infer failed\n";
      continue;
    }

    // 5. unroot inferred tree → starting tree
    std::string inferred = read_file(base + "_infer.treefile");
    if (!inferred.empty()) {
      std::string start = unroot(inferred);
      if (!start.empty())
        write_file(base + "_start.newick", start);
    }

    // 6. cleanup: remove unnecessary files
    remove((base + "_raw.tre").c_str());
    remove((base + "_raw.tre.log").c_str());
    remove((base + "_infer.bionj").c_str());
    remove((base + "_infer.mldist").c_str());
    remove((base + "_infer.ckp.gz").c_str());
  }
  std::cout << "done\n";
}