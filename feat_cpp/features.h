#pragma once
#include "phylotree.h"
#include <array>

struct FeatureVector {
    std::array<double, 20> features;
};

FeatureVector compute_spr_features(const PhyloTree& tree, const SPRMove& move);