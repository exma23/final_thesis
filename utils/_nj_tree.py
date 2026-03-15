import torch

def nj_torch(d):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = torch.tensor(d, device=device, dtype=torch.double)
    n_taxa = d.shape[0]

    max_val = d.max() * d.shape[0]
    adj_idxs = [i for i in range(n_taxa)]
    int_node = n_taxa + 1

    edges = []

    while d.shape[0] > 3:

        s = d.sum(dim=-1).unsqueeze(0)

        q = (d.shape[0] - 2) * d - s - s.T
        q.fill_diagonal_(max_val)
        idxs = torch.argmin(q).item()
        minI, minJ = idxs // q.shape[0], idxs % q.shape[0]

        edges += [(adj_idxs[minI], int_node), (adj_idxs[minJ], int_node)]

        adj_idxs.pop(max(minI, minJ))
        adj_idxs[min(minI, minJ)] = int_node
        int_node += 1

        new_dist = torch.zeros(d.shape[0] - 1, device=device, dtype=torch.double)
        new_dist[: max(minI, minJ)] = (d[minI, : max(minI, minJ)] + d[minJ, : max(minI, minJ)] - d[minI, minJ]) / 2
        new_dist[max(minI, minJ):] = (d[minI, max(minI, minJ) + 1:] + d[minJ, max(minI, minJ) + 1:] - d[
            minI, minJ]) / 2

        d = torch.cat([d[:, : max(minI, minJ)], d[:, max(minI, minJ) + 1:]], dim=1)
        d = torch.cat([d[: max(minI, minJ), :], d[max(minI, minJ) + 1:, :]], dim=0)
        d[min(minI, minJ), :] = new_dist
        d[:, min(minI, minJ)] = new_dist

    for i in range(3):
        edges.append((adj_idxs[i], n_taxa))

    return edges


