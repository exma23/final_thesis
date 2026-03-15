#include <iostream>
#include "features_f.h"
#include <vector>


void print_swap_mat(double*** sm, int size){
    for(int k = 0; k<2; k++){
        for(int i =0; i<size; i++) {
            for(int j =0; j<size; j++) std::cout<<sm[k][i][j]<<" ";
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}


double feat_run(double **d, int **sparse_A, int n_taxa, int m, short n_moves, short* features_map, double* features)
{


	tree *T = nullptr;
	int i, k = 0;

	T = adj_to_tree(sparse_A, n_taxa, m);
	// for (i = 0; i < m; i++)
	// 	delete[] sparse_A[i];
	// delete[] sparse_A;

	double **A;
	A = nullptr;


	A = initDoubleMatrix(m);
	
    
	makeBMEAveragesTable (T, d, A);


	assignBMEWeights (T, A);
	weighTree (T);

	int n_subtrees = (n_taxa * 2) - 3;

	double ** subtrees = new double* [n_subtrees*2];
	for(i=0; i<n_subtrees*2; i++) {subtrees[i] = new double[3]; for(k=0; k<3; k++) subtrees[i][k] = 0;}

	subtrees_features(T, d, A, subtrees, n_taxa, n_subtrees);
	double longest_branch = get_longest_branch(subtrees, n_subtrees);
	// print_matrix(subtrees, n_subtrees*2, 3);

    double ***swapWeights;
	swapWeights = new double** [2];
	bool **validSwaps = initBoolMatrix(T->size);

    for (i=0; i<2; i++) swapWeights[i] = initDoubleMatrix (T->size);

    edge* e;
	int counter = 0;

    for (e = depthFirstTraverse (T, nullptr); nullptr != e; e = depthFirstTraverse (T, e)) {
		assign_features (e->head, A, swapWeights, validSwaps, &counter, features_map, features, subtrees, T->weight, longest_branch, n_subtrees);
	}
	// std::cout<<"moves "<<counter<<" "<< n_moves<<std::endl;
	// print_swap_mat(swapWeights, T->size);
	// print_features(features, n_moves, 20);
	// explainedVariance (D, T, numSpecies, options->precision, options->input_type, options->fpO_stat_file);

	deleteMatrix(A, m);
	deleteMatrix(subtrees, n_subtrees*2);

	for (i=0; i<2; i++)
		deleteMatrix (swapWeights[i], T->size);
	delete[] swapWeights;
	deleteBoolMatrix (validSwaps, T->size);

	double tree_length = T ->weight;
	deleteTree(T);
	return tree_length;

}
