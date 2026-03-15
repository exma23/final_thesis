//
//  Copyright 2002-2007 Rick Desper, Olivier Gascuel
//  Copyright 2007-2014 Olivier Gascuel, Stephane Guindon, Vincent Lefort
//
//  This file is part of FastME.
//
//  FastME is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  FastME is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with FastME.  If not, see <http://www.gnu.org/licenses/>
//


#ifndef FEATURES_H_
#define FEATURES_H_

#include "graph.h"
#include "bNNI.h"
#include "subtrees_spr.h"
// #include "distance.h"
#include "subtrees_f.h"
#include "initialiser.h"
#include <vector>

double feat_run(double **d, int **sparse_A, int n_taxa, int m, short n_moves, short* features_map, double* features);
void print_swap_mat(double*** sm, int size);

#endif /*FEATURES_H_*/

