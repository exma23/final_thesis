#include "features.hpp"
#include "mem_alloc.hpp"
#include "pllInternal.hpp"
#include "utils.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>
#include <utility>


using actions = std::pair<int, int>;
using actions_X_y = std::tuple<std::vector<actions>, std::vector<SPRFeatures>, std::vector<double>>;



actions_X_y get_state_action(pllInstance *tr, partitionList *pr, int sprDist) {


}