#pragma once
#include "pll.hpp"
#include "utils.hpp"
#include "pllInternal.hpp"

pllRearrangeList * pllGetAllSPRMoves(pllInstance *tr, partitionList *pr,
                                     int mintrav, int maxtrav, int maxMoves);