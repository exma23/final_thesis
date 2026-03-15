#include "fastme.h"
#include "features.h"
#include <iostream>
#include <omp.h>
#include <sched.h>
#include "results.h"
#include "features_f.h"
#include <vector>

extern "C" {

    int* test (double* d, int* init_adj, int n_taxa, int m) {

        double ** D = new double*[n_taxa];
        int i,j;

        for(i = 0; i< n_taxa; i++) D[i] = &d[i*n_taxa];
 
        int ** A = new int*[m];
        for(i = 0; i< m; i++) A[i] = &init_adj[i*m];
                
        int* solution_mat = new int[m*m];
        double obj_val;
        int nni_count;
        int spr_count;
        run(D, A, solution_mat, n_taxa, m, obj_val, nni_count, spr_count);

        if (D!= nullptr) delete[] D;
        if (A!= nullptr) delete[] A;

        return solution_mat;

        }




    results* test_parallel (double* d, int* init_adj, int n_taxa, int m, int population_size, int num_procs) {

        
        double ** D = new double*[n_taxa];
        int i,j,t, mat_size;
        mat_size = m*m;

        for(i = 0; i< n_taxa; i++) D[i] = &d[i*n_taxa];

        int *** A = new int**[population_size];
        int* solution_mat = new int[population_size*mat_size];


        double* obj_vals = new double[population_size];
        int * nni_counts = new int[population_size];
        int * spr_counts = new int[population_size];



        omp_set_num_threads(num_procs);
        #pragma omp parallel for schedule(dynamic) shared(A, d)
        for(t = 0; t < population_size; t++) {
            A[t]= new int*[m];
            for(i = 0; i< m; i++) A[t][i] = &init_adj[t*mat_size + i*m];
                
            run(D, A[t], &solution_mat[t*mat_size], n_taxa, m, obj_vals[t], nni_counts[t], spr_counts[t]);
        }

        results* res = new results[1];
        res -> nni_counts = 0;
        res -> spr_counts = 0;
        res -> solution_adjs = solution_mat;
        res -> objs = obj_vals;

        for(t = 0; t < population_size; t++){
            delete[] A[t];
            res -> nni_counts += nni_counts[t];
            res -> spr_counts += spr_counts[t];
//            std::cout<<obj_vals[t]<<" ";
        }
        // std::cout<<std::endl;
        delete[] D;
        delete[] A; 
        delete[] nni_counts;
        delete[] spr_counts;

        return res;
        }


    results* run_BSPR_(double* d, int* init_adj, int n_taxa, int m, int max_steps) {

        
        double ** D = new double*[n_taxa];
        int i,j,t, mat_size;
        mat_size = m*m;

        for(i = 0; i< n_taxa; i++) D[i] = &d[i*n_taxa];

        int ** A = new int*[m];
        int* solution_mat = new int[mat_size];
        
        for(i = 0; i< m; i++) A[i] = &init_adj[i*m];
        

        double obj_val;
        std::vector<double>  spr_vals= std::vector<double> ();

        //run_BSPR(double **d, int **init_adj, int* solution_mat, int n_taxa, int m, double & obj_val, int max_steps, std::vector<double> &)
        run_BSPR(D, A, solution_mat, n_taxa, m, obj_val, max_steps, spr_vals);


        results* res = new results[1];
        res -> nni_counts = 0;
        res -> spr_counts = spr_vals.size();
        res -> solution_adjs = solution_mat;
        res -> objs = new double[spr_vals.size()];

        for(i = 0; i<spr_vals.size(); i++) res -> objs[i] = spr_vals[i];


        // std::cout<<std::endl;
        delete[] D;
        delete[] A; 

        return res;
        }


    BSPRresults* run_BSPR_batch(double* d, int* init_adj, int* n_taxa, int* m,int batch_size, int max_steps) {

        double* obj_val = new double[batch_size];
        std::vector<std::vector<double>>  spr_vals= std::vector<std::vector<double>> (batch_size, std::vector<double> ());

        std::vector<int> d_start_idx = std::vector<int> (batch_size, 0);
        std::vector<int> adj_start_idx = std::vector<int> (batch_size, 0);
        for(int b=1; b < batch_size; b ++) {
            d_start_idx[b] = d_start_idx[b - 1] + n_taxa[b]*n_taxa[b];
            adj_start_idx[b] = adj_start_idx[b - 1] + m[b]*m[b];
        }
        int** solution_mat = new int*[batch_size];

        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(static) 
        for(int b=0; b < batch_size; b++) {
            double ** D = new double*[n_taxa[b]];
            int i,j,t, mat_size;
            mat_size = m[b]*m[b];

            for(i = 0; i< n_taxa[b]; i++) D[i] = &d[d_start_idx[b] + i*n_taxa[b]];

            int ** A = new int*[m[b]];
            solution_mat[b] = new int[mat_size];
            
            for(i = 0; i< m[b]; i++) A[i] = &init_adj[adj_start_idx[b] + i*m[b]];
            



            //run_BSPR(double **d, int **init_adj, int* solution_mat, int n_taxa, int m, double & obj_val, int max_steps, std::vector<double> &)
            run_BSPR(D, A, solution_mat[b], n_taxa[b], m[b], obj_val[b], max_steps, spr_vals[b]);

            delete[] D;
            delete[] A; 
        }

        

        int total_spr_counts = 0;
        int* spr_counts = new int[batch_size];
        int solution_size = 0;
        for(int b = 0; b < batch_size; b++) {
            total_spr_counts += spr_vals[b].size();
            spr_counts[b] = spr_vals[b].size();
            solution_size += m[b]*m[b];
            }

        BSPRresults* res = new BSPRresults[1];
        res -> total_spr = total_spr_counts;
        res -> spr_counts = spr_counts;
        res -> solution_adjs = new int[solution_size];
        res -> objs = new double[total_spr_counts];

        int k = 0;
        int kk = 0;

        for(int b = 0; b< batch_size; b++) {
            for(int i = 0; i<spr_vals[b].size(); i++) {res -> objs[k] = spr_vals[b][i]; k++;}
            for(int i = 0; i<m[b]*m[b]; i++) {res -> solution_adjs[kk] = solution_mat[b][i]; kk++;}
        }
        

        // delete [] solution_mat;

        return res;
        }

    Features * get_features_(double* d, int* edges, int n_taxa, int m){
        
        std::vector<std::tuple<int, int>> edges_ (2*n_taxa - 3, Edge (0, 0) );

        for (int i =0; i  < 2*n_taxa - 3 ; i++) {
            get_head(edges_[i]) = edges[i*2];
            get_tail(edges_[i]) = edges[i*2 + 1];
        }
        double * feat = new double[(2 * n_taxa - 6) * (2 * n_taxa - 7) * 20];
        short * feat_map = new short[(2 * n_taxa - 6) * (2 * n_taxa - 7) * 4];
        Ftree tree(d, n_taxa, m, edges_, feat, feat_map);


        Features* features = new Features[1];
        features -> n_moves = tree.level;
        features -> features = feat;
        features -> moves = feat_map;
        

        return features;

    }

    BatchFeatures * get_features_batch_(double* d, int* edges, int* n_taxa, int* m, int batch_size, bool nni_repetitions){

        // for(int i = 0; i < (2*n_taxa[0] - 3)*2; i+=2) std::cout<<edges[i]<< " "<<edges[i + 1]<<std::endl; 
        
        std::vector<int> edge_start_idx = std::vector<int> (batch_size, 0);
        std::vector<int> d_start_idx = std::vector<int> (batch_size, 0);
        std::vector<int> feat_start_idx = std::vector<int> (batch_size, 0);
        int feat_size;
        if (not nni_repetitions) feat_size = (2 * n_taxa[0] - 6) * (2 * n_taxa[0] - 7);
        else feat_size = (2 * n_taxa[0] - 6) * (2 * n_taxa[0] - 7) + 2 * (n_taxa[0] - 3) * 3;
        int feat_size_tree;

        for(int b=1; b < batch_size; b ++) {
            edge_start_idx[b] = edge_start_idx[b - 1] + (2*n_taxa[b] - 3)*2;
            d_start_idx[b] = d_start_idx[b - 1] + n_taxa[b] * n_taxa[b];
            feat_size_tree = (2 * n_taxa[b] - 6) * (2 * n_taxa[b] - 7);
            if(nni_repetitions) feat_size_tree += 2 * (n_taxa[b] - 3) * 3;
            feat_start_idx[b] = feat_start_idx[b - 1] + feat_size_tree;
            feat_size += feat_size_tree;
        }
        

        int* n_moves = new int[batch_size];
        double* tree_length = new double[batch_size];
        double * feat = new double[feat_size * 20];
        short * feat_map = new short[feat_size * 4];

        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(static) shared(nni_repetitions)
        for(int b=0; b < batch_size; b ++) {
            std::vector<std::tuple<int, int>> edges_ (2*n_taxa[b] - 3, Edge (0, 0) );

            for (int i =0; i  < 2*n_taxa[b] - 3 ; i++) {
                get_head(edges_[i]) = edges[edge_start_idx[b] + i*2];
                get_tail(edges_[i]) = edges[edge_start_idx[b] + i*2 + 1];
            }
            Ftree tree(&d[d_start_idx[b]], n_taxa[b], m[b], edges_, 
                                &feat[feat_start_idx[b]*20], &feat_map[feat_start_idx[b]*4], nni_repetitions);
            n_moves[b] = tree.level;
            tree_length[b] = tree.tree_length;
        }

        

        BatchFeatures* features_results = new BatchFeatures[1];
        features_results -> tree_length = tree_length;
        features_results -> n_rows = feat_size;
        features_results -> n_moves = n_moves;        
        features_results -> features = feat;
        features_results -> moves = feat_map;

        return features_results;

    }




BatchFeatures * get_features_batch_new_(double* d, int* edges, int* n_taxa, int* m, int batch_size){
        


        int tot_moves = 2 * (n_taxa[0] - 3) * (2 * n_taxa[0] - 7); //init with only the first
        int* n_moves = new int[batch_size];
        int* feat_start= new int[batch_size]; 
        int* feat_map_start= new int[batch_size];
        int* edge_start= new int[batch_size];
        int* taxa_start= new int[batch_size];

        n_moves[0] = tot_moves;
        feat_map_start[0] = 0;
        feat_start[0] = 0;
        edge_start[0] = 0;
        taxa_start[0] = 0;

        for(int b=1; b<batch_size; b++) {
            n_moves[b] = 2 * (n_taxa[b] - 3) * (2 * n_taxa[b] - 7);
            tot_moves += n_moves[b];
            feat_map_start[b] = feat_map_start[b-1] + n_moves[b-1] * 4;
            feat_start[b] = feat_start[b-1] + n_moves[b-1] * 20;
            edge_start[b] = edge_start[b-1] + (2*n_taxa[b - 1] - 3)*2;
            taxa_start[b] = taxa_start[b-1] +  n_taxa[b - 1] *  n_taxa[b - 1];
        }


        short* features_map = new short[tot_moves*4];
        double* features = new double[tot_moves*20];
        double* tree_length = new double[batch_size];
        // std::cout<<batch_size<< "   call" <<std::endl;


        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(static)
        for(int b=0; b < batch_size; b ++) {
            int** sparse_A = get_sparse_A_from_edge(&edges[edge_start[b]], 2*n_taxa[b] - 3, m[b]);
            
            double** D = new double*[n_taxa[b]];
            for(int i=0; i<n_taxa[b]; i++) D[i] = &d[taxa_start[b] + n_taxa[b] * i];
            // print_matrix(D, n_taxa[b], n_taxa[b]);
            tree_length[b] = feat_run(D, sparse_A, n_taxa[b], m[b], n_moves[b], &features_map[feat_map_start[b]], &features[feat_start[b]]);
            // print_features(&features[feat_start[b]], n_moves[b], 20);
            // print_features_map(&features_map[feat_map_start[b]], n_moves[b]);
            // print_matrix(sparse_A, m[b], 3);
            deleteIntMatrix(sparse_A, m[b]); delete[] D;

        }



        BatchFeatures* features_results = new BatchFeatures[1];
        features_results -> tree_length = tree_length;
        features_results -> n_rows = tot_moves;
        features_results -> n_moves = n_moves;        
        features_results -> features = features;
        features_results -> moves = features_map;

        delete[] feat_start; 
        delete[] feat_map_start;
        delete[] edge_start;
        delete[] taxa_start;

        return features_results;

    }



    void free_result(results* res){
        delete[] res -> solution_adjs;
        delete[] res -> objs;
        res -> solution_adjs = nullptr;
        res -> objs = nullptr;
        res  = nullptr;
    }

    void free_features(Features* feat){
        delete[] feat -> moves;
        delete[] feat -> features;
        feat -> moves = nullptr;
        feat -> features = nullptr;
        feat  = nullptr;
    }


    void free_features_batch(BatchFeatures* feat){
        delete[] feat -> tree_length;
        delete[] feat ->n_moves;
        delete[] feat -> moves;
        delete[] feat -> features;
        feat -> tree_length = nullptr;
        feat -> n_moves = nullptr;
        feat -> moves = nullptr;
        feat -> features = nullptr;
        feat  = nullptr;
    }

    void free_spr_batch_results(BSPRresults* res){
        delete[] res -> solution_adjs;
        delete[] res -> objs;
        delete[] res -> spr_counts;
        res -> solution_adjs = nullptr;
        res -> objs = nullptr;
        res -> spr_counts = nullptr;
        res  = nullptr;
    }


    results* test_obj(){

        int * adj = new int[10];
        double* obj_vals = new double[10];

        for(int i=0; i< 10; i ++ ) {
            adj[i] = 2*i;
            obj_vals[i] = 0.2*i;
        }
        results* obj = new results[1];
        obj ->objs = obj_vals; obj->nni_counts=4; obj->spr_counts=3; obj->solution_adjs=adj;
        //std::cout<<"fjfjfjfjf"<< obj->spr_counts<<std::endl;

        return obj;
    }

}



