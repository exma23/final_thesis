#include "features.hpp"
#include "pllInternal.hpp"
#include "utils.hpp"
#include "newick.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

int main() {
    // Đọc file Newick
    const char* filename = "../data/1_ground_truth.tre";
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return 1;
    }
    std::string newick_str((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    fin.close();

    // Parse Newick và khởi tạo cây
    pllNewickTree* newick = pllNewickParseString(newick_str.c_str());
    if (!newick) {
        std::cerr << "Failed to parse Newick tree." << std::endl;
        return 1;
    }
    pllInstance tr = {};
    partitionList pr = {};
    tr.mxtips = newick->tips;
    tr.ntips = newick->tips;
    pllTreeInitTopologyNewick(&tr, newick, 0);
    int sprDist = 2*tr.mxtips;
    std::cout << "MXTIPS " << sprDist << "\n";
    auto features = computeAllSPRFeatures(&tr, &pr, sprDist);

    for (size_t i = 0; i < std::min<size_t>(features.size(), 9); ++i) {
        const auto& f = features[i];
        std::cout << "SPR " << i << ": "
                  << "totalBL=" << f.total_branch_lengths
                  << ", pruneBL=" << f.prune_branch_len
                  << ", regraftBL=" << f.regraft_branch_len
                  << ", topoDist=" << f.topo_distance
                  << ", pathBL=" << f.branch_len_distance
                  << ", n_leaves_a=" << f.n_leaves_a
                  << ", n_leaves_b=" << f.n_leaves_b
                  << std::endl;
    }
    return 0;
}