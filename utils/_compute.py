import torch
from typing import Set, Tuple

def feat_normalization_fn(x, n_taxa):
    div_t_len = torch.ones(20)
    div_t_len[[0, 4, 8, 11, 14, 17]] = 0

    div_taxa = torch.zeros(20)
    div_taxa[[4, 8, 11, 14, 17]] = 1

    x[..., div_t_len == 1] = x[..., div_t_len == 1] / x[..., 0].view(*x.shape[:-1], 1)
    x[..., div_taxa == 1] = x[..., div_taxa == 1] / n_taxa

    return x

#-------------------------------------
def get_splits(adj, n_taxa, root=None):
    if root is None:
        root = n_taxa

    splits = set()
    all_taxa = set(range(n_taxa))

    def dfs(node, parent):
        taxa = set()

        if node < n_taxa:
            taxa.add(node)

        for child in adj[node]:
            if child != parent:
                taxa |= dfs(child, node)

        if parent is not None and 0 < len(taxa) < n_taxa:
            if len(taxa) > n_taxa // 2:
                taxa = all_taxa - taxa
            splits.add(frozenset(taxa))

        return taxa

    dfs(root, None)
    return splits

def rf_distance(adj1, adj2, n_taxa):
    s1 = get_splits(adj1, n_taxa)
    s2 = get_splits(adj2, n_taxa)
    return len(s1 ^ s2)   # symmetric difference

def normalized_rf_distance(adj_mat1, adj_mat2, n_taxa: int) -> float:
    rf = rf_distance(adj_mat1, adj_mat2, n_taxa)
    max_rf = 2 * (n_taxa - 3)
    return rf / max_rf if max_rf > 0 else 0.0