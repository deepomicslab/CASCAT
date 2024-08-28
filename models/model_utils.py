import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances
import os
from scipy import sparse
from scipy.sparse import csr_matrix
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from InformationMeasure.MatrixMeasures import *
from utils.Plot import *


class CMIPlot:
    def __init__(self, adata, pesudo_key, predict_key, connect_key, adj_key, emb_key, save_path=None,
                 start_cell_idx=None, root=None, group_frac=None):
        self.load_config_data(adata, pesudo_key, predict_key, connect_key,
                              adj_key, emb_key, save_path, start_cell_idx,
                              root, group_frac)

    def load_config_data(self, adata, pesudo_key, predict_key, connect_key, adj_key, emb_key,
                         save_path=None, start_cell_idx=None, root=None, group_frac=None):
        self.adata = adata
        if pesudo_key in adata.obs_keys():
            self.pesudotime = adata.obs[pesudo_key]
        else:
            raise Exception(f"Pesudotime key {pesudo_key} not found in adata.obs")
        if 'cluster' not in adata.obs.keys():
            raise Exception("No cluster label found in adata.obs")
        else:
            self.labels = adata.obs['cluster']
        if predict_key in adata.obs.keys():
            self.predict_labels = adata.obs[predict_key]
        else:
            raise Exception(f"Predict key {predict_key} not found in adata.obs")
        if connect_key in adata.uns.keys():
            self.connectivities = adata.uns[connect_key]
        else:
            raise Exception(f"Connect key {connect_key} not found in adata.uns")
        if adj_key in adata.obsp.keys():
            self.adj = adata.obsp[adj_key]
        else:
            raise Exception(f"Adj key {adj_key} not found in adata.obsp")
        if emb_key in adata.obsm.keys():
            self.emd = adata.obsm[emb_key]
        else:
            raise Exception(f"Embedding key {emb_key} not found in adata.obsm")
        self.group_frac = group_frac
        self.start_cell_idx = start_cell_idx
        self.root = root
        self.img_path = save_path
        self.graph = nx.from_pandas_adjacency(self.connectivities, create_using=nx.DiGraph)
        self.labels_map = {i: self.group_frac.loc[:, i].idxmax() for i in list(self.graph.nodes)}

    def run_group_param(self):
        n_groups = len(set(self.predict_labels))
        group_pop = np.zeros([n_groups, 1])
        group_pt = np.zeros([n_groups, 1])
        set_labels = sorted(list(set(self.predict_labels)))
        for group_i in set_labels:
            group_idx = int(group_i)
            loc_i = np.where(self.predict_labels == group_i)[0]
            group_pop[group_idx] = len(loc_i)
            group_pt[group_idx] = np.mean(self.pesudotime.iloc[loc_i])
        return group_pop, group_pt

    def plot_trajectory_tree(self, show=True):
        plot_pydot_graph(self.graph, self.labels_map, self.img_path, show=show)

    def plot_embedding(self, show=True, colors='Paired'):
        plot_embedding(self.emd, self.group_frac, self.connectivities, self.predict_labels, colors=colors,
                       save_path=self.img_path, label_map=self.labels_map, show=show)

    def plot_st_embedding(self, show_trajectory=False, colors='tab10'):
        save_path = self.img_path + 'st_emb.png' if self.img_path is not None else None
        plot_st_embedding(self.adata, self.predict_labels, group_frac=self.group_frac, colors=colors,
                          save_path=save_path, show_trajectory=show_trajectory, connectivities=self.connectivities)

    def plot_pseudotime(self, show=True):
        plot_pesudotime(self.emd, self.group_frac, self.connectivities, self.predict_labels, self.pesudotime,
                        self.labels_map, show, save_path=self.img_path)

    def plot_st_pseudotime(self):
        plot_st_pesudotime(self.adata, self.pesudotime, save_path=self.img_path)


def get_feat_mask(features, rate):
    feat_size = features.shape[1]
    mask = torch.ones(features.shape, device=features.device)
    samples = np.random.choice(feat_size, size=int(feat_size * rate), replace=False)

    mask[:, samples] = 0
    return mask


def dense2sparse(adj):
    (row, col), val = dense_to_sparse(adj)
    num_nodes = adj.size(0)
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes))


def df2tensor(df, adj):
    num_nodes = adj.shape[0]
    matrix = np.zeros((num_nodes, num_nodes))
    for row in df.values:
        node1, node2 = int(row[0]), int(row[1])
        matrix[node1, node2] = 1
    matrix = csr_matrix(matrix)
    return matrix


def df2sparse(df, num_nodes):
    rows, cols, values = [], [], []
    for row in df.values:
        node1, node2 = row[0], row[1]
        rows.append(node1)
        cols.append(node2)
        values.append(row[2])
    rows = torch.tensor(rows, dtype=torch.int64)
    cols = torch.tensor(cols, dtype=torch.int64)
    edge_index = torch.stack([rows, cols], dim=0)
    values = torch.tensor(values, dtype=torch.float64)
    sparse_tensor = SparseTensor(row=edge_index[0], col=edge_index[1], value=values,
                                 sparse_sizes=(num_nodes, num_nodes))
    return sparse_tensor


def normalize_adj_symm(adj):
    assert adj.size(0) == adj.size(1)
    if not isinstance(adj, SparseTensor):
        adj = dense2sparse(adj)
    if not adj.has_value():
        adj = adj.fill_value(1., dtype=torch.float32)
    deg = torch_sparse.sum(adj, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj = torch_sparse.mul(adj, deg_inv_sqrt.view(-1, 1))
    adj = torch_sparse.mul(adj, deg_inv_sqrt.view(1, -1))
    return adj


def df2Mtx(df, labels, col1: str = 'Cell1', col2: str = 'Cell2'):
    labels.index = [int(i) for i in range(len(labels))]
    df[col1] = df[col1].map(labels)
    df[col2] = df[col2].map(labels)
    grouped = df.groupby(col1)
    matrix = pd.DataFrame(np.zeros((len(set(labels)), len(set(labels)))))
    matrix.columns = list(set(labels))
    matrix.index = list(set(labels))
    for name, group in grouped:
        sub_group = group.groupby(col2)
        for sub_name, sub_group_ in sub_group:
            matrix.loc[name, sub_name] += sub_group_.shape[0]
    return matrix


def adj2Mtx(adj, labels, col1: str = 'Cell1', col2: str = 'Cell2'):
    if isinstance(adj, sparse.csr.csr_matrix):
        adj = adj.toarray()
    if isinstance(labels.values[0], str):
        labels = {int(i): labels[i] for i in range(len(labels))}
    edges = np.nonzero(adj)
    df = pd.DataFrame({col1: edges[0], col2: edges[1], 'Value': adj[edges]})
    df[col1] = df[col1].map(labels)
    df[col2] = df[col2].map(labels)
    grouped = df.groupby(col1, observed=False)
    matrix = pd.DataFrame(np.zeros((len(set(labels.values())), len(set(labels.values())))))
    matrix.columns = list(set(labels.values()))
    matrix.index = list(set(labels.values()))
    for name, group in grouped:
        sub_group = group.groupby(col2, observed=False)
        for sub_name, sub_group_ in sub_group:
            matrix.loc[name, sub_name] += sub_group_.shape[0]
    return matrix


def return_condition_mi_pairs(adj_mask):
    if isinstance(adj_mask, torch.Tensor):
        adj_mask = np.array(adj_mask.cpu().detach())
    tri_pairs, pairs = [], []
    for row_idx in range(adj_mask.shape[0]):
        for col_idx in np.where(adj_mask[row_idx])[0]:
            for rest_idx in np.where(adj_mask[row_idx])[0]:
                if rest_idx != col_idx:
                    tri_pairs.append([row_idx, col_idx, rest_idx])
    print('The number of triplets is {}'.format(len(tri_pairs)))
    for row_idx in range(adj_mask.shape[0]):
        for col_idx in np.where(adj_mask[row_idx])[0]:
            pairs.append([row_idx, col_idx])
    print('The number of pairs is {}'.format(len(pairs)))
    return tri_pairs, pairs


def get_CMI_connectivities(adata, dir, percent=0.2):
    adj = adata.obsp["connectivities"]
    if not os.path.exists(dir + "CMI.csv"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        print('Start to generate ', dir, 'CMI ...')
        if not isinstance(adata.X, np.ndarray):
            adata.X = adata.X.toarray()
        df = generate_CMI_from_adj(adj.toarray(), adata.X, dir)
    else:
        df = pd.read_csv(dir + "CMI.csv")
    adj = CMIdf2Adj(df, adj, percent)
    return adj


def generate_CMI_from_adj(adj_mask, gene_exprs, save_path):
    all_tri_pairs, all_pairs = return_condition_mi_pairs(adj_mask)
    if not os.path.exists(save_path + 'entropy_matrix.csv'):
        entropy_matrix = get_entropy_matrix([i for i in range(len(gene_exprs))], gene_exprs)
        entropy_matrix.to_csv(save_path + 'entropy_matrix.csv')
    else:
        entropy_matrix = pd.read_csv(save_path + 'entropy_matrix.csv', index_col=0).values
    if not os.path.exists(save_path + 'dual_entropy_matrix.csv'):
        pairs = [sorted(i) for i in all_pairs]
        pairs = [list(i) for i in list(set([tuple(i) for i in pairs]))]
        dual_entropy_matrix = get_dual_joint_entropy_matrix(pairs, gene_exprs)
        dual_entropy_matrix.to_csv(save_path + 'dual_entropy_matrix.csv')
    else:
        dual_entropy_matrix = pd.read_csv(save_path + 'dual_entropy_matrix.csv', index_col=0).values
    if not os.path.exists(save_path + 'triple_entropy_matrix.csv'):
        tri_pairs = [sorted(i) for i in all_tri_pairs]
        tri_pairs = [list(i) for i in list(set([tuple(i) for i in tri_pairs]))]
        triple_entropy_matrix = get_triple_joint_entropy_matrix(tri_pairs, gene_exprs)
        triple_entropy_matrix.to_csv(save_path + 'triple_entropy_matrix.csv')
    else:
        triple_entropy_matrix = pd.read_csv(save_path + 'triple_entropy_matrix.csv', index_col=0).values
    condition_mi = get_conditional_mutual_info_matrix(all_tri_pairs, entropy_matrix, dual_entropy_matrix,
                                                      triple_entropy_matrix)
    print('Saving conditional_mutual_info values: ', condition_mi.shape[0])
    condition_mi.to_csv(save_path + 'CMI.csv')
    return condition_mi


def CMIdf2Adj(df, adj, percent=0.5, col1: str = 'Cell1', col2: str = 'Cell2', value: str = 'CMI'):
    group_CMI = df.groupby([col1, col2])[value].mean()
    group_CMI = group_CMI.loc[group_CMI > -np.inf]
    theshold = group_CMI.quantile(percent)
    group_CMI = group_CMI.loc[group_CMI > theshold]
    group_CMI.fillna(np.mean(group_CMI.values), inplace=True)
    if group_CMI.min() < 0:
        group_CMI = group_CMI - group_CMI.min()
    adj = df2tensor(group_CMI.reset_index(), adj)
    return adj


import numpy as np
from scipy.stats import mode


def refine_labels(raw_labels, dist_sort_idx, n_neigh):
    n_cell = len(raw_labels)
    raw_labels = np.tile(raw_labels, (n_cell, 1))
    idx = dist_sort_idx[:, 1:n_neigh + 1]
    new_labels = raw_labels[np.arange(n_cell)[:, None], idx]
    new_labels = mode(new_labels, axis=1, keepdims=True).mode
    return new_labels


def mclust_R(embedding, n_clusters, random_state, modelNames='EEE'):
    np.random.seed(random_state)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_state)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(embedding), n_clusters, modelNames)
    if not isinstance(res, rpy2.rinterface_lib.sexp.NULLType):
        clusters = np.array(res[-2])
    else:
        clusters = np.ones(len(embedding))

    return clusters


def connect_graph(adj, data, start_cell_id):
    # TODO: Update the heuristic here which involves using the
    # cell with the max distance to establish a connection with
    # the disconnected parts of the clusters.

    index = adj.index
    dists = pd.Series(dijkstra(adj, indices=start_cell_id), index=index)
    unreachable_nodes = index[dists == np.inf]
    if len(unreachable_nodes) == 0:
        return adj

    # Connect unreachable nodes
    while len(unreachable_nodes) > 0:
        farthest_reachable_id = dists.loc[index[dists != np.inf]].idxmax()
        # Compute distances to unreachable nodes
        unreachable_dists = pairwise_distances(
            data.loc[farthest_reachable_id, :].values.reshape(1, -1),
            data.loc[unreachable_nodes, :])
        unreachable_dists = pd.Series(
            np.ravel(unreachable_dists), index=unreachable_nodes
        )
        # Add edge between farthest reacheable and its nearest unreachable
        adj.loc[farthest_reachable_id, unreachable_dists.idxmin()] = unreachable_dists.min()
        # Recompute distances to early cell
        dists = pd.Series(dijkstra(adj, indices=start_cell_id), index=index)
        # Idenfity unreachable nodes
        unreachable_nodes = index[dists == np.inf]
    return adj


def run_group_frac(predict_int_labels, labels):
    predict_str_labels = np.asarray([str(i) for i in predict_int_labels])
    n_groups = len(set(predict_str_labels))
    n_truegroups = len(set(labels))
    group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=list(set(labels)),
                              index=list(set(predict_str_labels)))
    set_labels = sorted(list(set(predict_str_labels)))
    labels = np.asarray(labels)
    for group_i in set_labels:
        loc_i = np.where(predict_str_labels == group_i)[0]
        true_label_in_group_i = list(labels[loc_i])
        ll_temp = list(set(true_label_in_group_i))
        for ii in ll_temp:
            group_frac.loc[group_i, ii] = true_label_in_group_i.count(ii)
    group_frac = group_frac.T
    print(f'group_frac: {group_frac}')
    return group_frac


def assign_label_map(group_frac):
    label_map = {}
    group_frac = group_frac.div(group_frac.sum(axis=1), axis=0)
    for group_i in group_frac.columns:
        label_map[group_i] = group_frac[group_i].idxmax()
    return label_map
