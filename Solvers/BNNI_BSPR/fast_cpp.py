import ctypes
import os

import numpy as np
import torch
from numpy.ctypeslib import ndpointer

class Result(ctypes.Structure):
    _fields_ = [
        ('nni_counts', ctypes.c_int),
        ('spr_counts', ctypes.c_int),
        ('solution_adjs', ctypes.POINTER(ctypes.c_int)),
        ('objs', ctypes.POINTER(ctypes.c_double)),
    ]


class Features(ctypes.Structure):
    _fields_ = [
        ('n_moves', ctypes.c_int),
        ('moves', ctypes.POINTER(ctypes.c_short)),
        ('features', ctypes.POINTER(ctypes.c_double)),
    ]


class FeaturesBatch(ctypes.Structure):
    _fields_ = [
        ('n_rows', ctypes.c_int),
        ('n_moves', ctypes.POINTER(ctypes.c_int)),
        ('moves', ctypes.POINTER(ctypes.c_short)),
        ('features', ctypes.POINTER(ctypes.c_double)),
        ('tree_length', ctypes.POINTER(ctypes.c_double)),
    ]


class BSPRresults(ctypes.Structure):
    _fields_ = [
        ('total_spr', ctypes.c_int),
        ('spr_counts', ctypes.POINTER(ctypes.c_int)),
        ('solution_adjs', ctypes.POINTER(ctypes.c_int)),
        ('objs', ctypes.POINTER(ctypes.c_double)),
    ]


class FastCpp:

    def __init__(self):
        self.run_results = None
        self.lib = ctypes.CDLL('Solvers/BNNI_BSPR/bridge.so')
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.numProcs = os.cpu_count()
        # self.lib.test.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
        #                           ctypes.c_int, ctypes.c_int]
        self.lib.test.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                  ctypes.c_int, ctypes.c_int]

        self.lib.test_parallel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

        self.lib.run_BSPR_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]

        self.lib.run_BSPR_batch.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                            ctypes.c_int, ctypes.c_int]

        self.lib.get_features_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int, ctypes.c_int]

        self.lib.get_features_batch_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                                 ctypes.c_int, ctypes.c_bool]

        self.lib.get_features_batch_new_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                                 ctypes.c_int]

        self.lib.free_features_batch.argtypes = [ctypes.POINTER(FeaturesBatch)]
        self.lib.free_spr_batch_results.argtypes = [ctypes.POINTER(BSPRresults)]

        self.lib.test_obj.argtypes = []
        self.lib.free_result.argtypes = [ctypes.POINTER(Result)]

        self.lib.test_obj.restype = ctypes.POINTER(Result)
        self.lib.test_parallel.restype = ctypes.POINTER(Result)
        self.lib.run_BSPR_.restype = ctypes.POINTER(Result)
        self.lib.run_BSPR_batch.restype = ctypes.POINTER(BSPRresults)
        self.lib.get_features_.restype = ctypes.POINTER(Features)
        self.lib.get_features_batch_.restype = ctypes.POINTER(FeaturesBatch)
        self.lib.get_features_batch_new_.restype = ctypes.POINTER(FeaturesBatch)

    def free_result_memory(self):
        self.lib.free_result(self.run_results)

    def test(self, d, adj_mat, n_taxa):
        np.savetxt("Solvers/Fast_BNNI_BSPR/mat", d, fmt='%.19f', delimiter=' ')
        np.savetxt("Solvers/Fast_BNNI_BSPR/init_mat", adj_mat, fmt='%i', delimiter=' ')
        n = np.array([n_taxa], dtype=np.int32)
        np.savetxt("Solvers/Fast_BNNI_BSPR/n_taxa", n, fmt='%i', delimiter=' ')
        os.system("Solvers/Fast_BNNI_BSPR/fast_me")
        return np.loadtxt("Solvers/Fast_BNNI_BSPR/result_adj_mat.txt", dtype=int)

    def run(self, d, adj_mat, n_taxa, m):
        self.lib.test.restype = ndpointer(dtype=ctypes.c_int32, shape=(m, m))
        adj = self.lib.test(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            adj_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                            ctypes.c_int(n_taxa), ctypes.c_int(m))
        return adj

    def run_parallel(self, d, adj_mats, n_taxa, m, population_size):
        run_results = self.lib.test_parallel(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             adj_mats.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                             ctypes.c_int(n_taxa), ctypes.c_int(m), ctypes.c_int(population_size),
                                             ctypes.c_int(self.numProcs))

        adjs = np.ctypeslib.as_array(run_results.contents.solution_adjs, shape=adj_mats.shape)
        objs = np.array(run_results.contents.objs[:population_size])
        nni_counts = run_results.contents.nni_counts
        spr_counts = run_results.contents.spr_counts
        self.run_results = run_results

        return adjs, objs, nni_counts, spr_counts

    def run_BSPR(self, d, adj_mat, n_taxa, m, max_steps):
        run_results = self.lib.run_BSPR_(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         adj_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                         ctypes.c_int(n_taxa), ctypes.c_int(m), ctypes.c_int(max_steps))

        spr_counts = run_results.contents.spr_counts

        adjs = np.ctypeslib.as_array(run_results.contents.solution_adjs, shape=adj_mat.shape)
        objs = np.array(run_results.contents.objs[:spr_counts])

        self.run_results = run_results

        return adjs, objs, spr_counts

    def run_BSPR_batch(self, tree_list, max_steps=10 ** 4):
        d = np.array([t.d.flatten() for t in tree_list])
        adj_mat = np.array([t.adj.flatten() for t in tree_list], dtype=np.int32).flatten()
        n_taxa = np.array([t.n_taxa for t in tree_list], dtype=np.int32)
        m = np.array([t.m for t in tree_list], dtype=np.int32)
        run_results = self.lib.run_BSPR_batch(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                              adj_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                              n_taxa.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                              m.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                              ctypes.c_int(len(tree_list)), ctypes.c_int(max_steps))

        spr_total = run_results.contents.total_spr
        spr_counts = np.copy(np.ctypeslib.as_array(run_results.contents.spr_counts, shape=(len(tree_list),)))

        count_idx = np.zeros(len(spr_counts) + 1, dtype=np.int64)
        count_idx[1:] = np.cumsum(spr_counts)

        objs = np.copy(np.ctypeslib.as_array(run_results.contents.objs, shape=(spr_total,)))
        objs = [objs[count_idx[i]: count_idx[i + 1]] for i in range(len(count_idx) - 1)]
        final_objs = np.array([objs[i][-1] for i in range(len(tree_list))])

        self.lib.free_spr_batch_results(run_results)

        return final_objs, objs, spr_counts

    def get_features_cpp(self, d, edges, n_taxa, m):
        edges_ = np.array(edges, dtype=np.int32)
        res = self.lib.get_features_(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     edges_.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                     ctypes.c_int(n_taxa), ctypes.c_int(m))
        n_moves = res.contents.n_moves
        moves = torch.tensor(np.ctypeslib.as_array(res.contents.moves, shape=(n_moves, 4)), device=self.device)
        features = torch.tensor(np.ctypeslib.as_array(res.contents.features, shape=(n_moves, 20)), device=self.device, dtype=torch.float32)
        self.lib.free_features(res)

        return moves, features

    def get_features_batch_cpp(self, tree_list, nni_repetition: bool = False):
        d = np.array([t.d.flatten() for t in tree_list])
        edges = np.array([t.edges for t in tree_list], dtype=np.int32)
        n_taxa = np.array([t.n_taxa for t in tree_list], dtype=np.int32)
        m = np.array([t.m for t in tree_list], dtype=np.int32)
        res = self.lib.get_features_batch_(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                           edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                           n_taxa.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                           m.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                           ctypes.c_int(len(tree_list)), ctypes.c_bool(nni_repetition))
        #res = self.lib.get_features_batch_new_(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        #                                   edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        #                                   n_taxa.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        #                                   m.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        #                                   ctypes.c_int(len(tree_list)))

        n_rows = res.contents.n_rows
        n_moves = np.copy(np.ctypeslib.as_array(res.contents.n_moves, shape=(len(tree_list),)))
        tree_length = np.copy(np.ctypeslib.as_array(res.contents.tree_length, shape=(len(tree_list),)))
        moves = torch.tensor(np.ctypeslib.as_array(res.contents.moves, shape=(n_rows, 4))).reshape((len(tree_list), -1, 4))
        features = torch.tensor(np.ctypeslib.as_array(res.contents.features, shape=(n_rows, 20)),
                                device=self.device, dtype=torch.float32).reshape((len(tree_list), -1, 20))

        self.lib.free_features_batch(res)

        return n_moves, moves, features, tree_length
