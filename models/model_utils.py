import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
from scipy.sparse.csgraph import dijkstra
from scipy.stats import mode
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigs
import scanpy as sc
import copy
from joblib import Parallel, delayed
from copy import deepcopy
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from time import time
import torch
import numpy as np
import pandas as pd
import cupy as cp
import os
from scipy import sparse
from scipy.sparse import csr_matrix, find
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from utils.Plot import plot_st_embedding, plot_subtype, plot_embedding, plot_pydot_graph, plot_pesudotime, \
    plot_st_pesudotime
from utils.Metrics import ClusteringMetrics
from InformationMeasure.CudaMeasures import get_entropy_matrix, get_dual_joint_entropy_matrix, \
    get_triple_joint_entropy_matrix, get_conditional_mutual_info_matrix


class CMIPlot:
    def __init__(self, adata, pesudo_key, predict_key, connect_key, emb_key, save_path=None,
                 start_cell_idx=None, root=None, group_frac=None):
        self.load_config_data(adata, pesudo_key, predict_key, connect_key, emb_key, save_path, start_cell_idx,
                              root, group_frac)

    def load_config_data(self, adata, pesudo_key, predict_key, connect_key, emb_key,
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
        plot_pydot_graph(self.graph, self.labels_map, self.img_path + 'trajectory_tree.png', show=show)

    def plot_embedding(self, show=True, colors='Paired'):
        plot_embedding(self.emd, self.group_frac, self.connectivities, self.predict_labels, colors=colors,
                       save_path=self.img_path + 'emb.png', label_map=self.labels_map, show=show)

    def plot_st_embedding(self, show_trajectory=False, colors='tab10'):
        save_path = self.img_path + 'st_emb.png' if self.img_path is not None else None
        plot_st_embedding(self.adata, self.predict_labels, group_frac=self.group_frac, colors=colors,
                          save_path=save_path, show_trajectory=show_trajectory, connectivities=self.connectivities)

    def plot_pseudotime(self, show=True):
        plot_pesudotime(self.emd, self.group_frac, self.connectivities, self.predict_labels, self.pesudotime,
                        self.labels_map, show, save_path=self.img_path + 'cascat_pseudotime.png')

    def plot_st_pseudotime(self):
        plot_st_pesudotime(self.adata, self.pesudotime, save_path=self.img_path + 'st_pseudotime.png')

    def plot_subtype(self, show=True):
        plot_subtype(self.predict_labels, self.labels, self.img_path, show)

    def plot_marker_heatmap(self, sorted_genes, order_layer, show=True):
        if 'gene_symbol' not in self.adata.var.keys():
            df = pd.DataFrame(self.adata.X, index=self.adata.obs_names, columns=self.adata.var.index)
        else:
            df = pd.DataFrame(self.adata.X, columns=self.adata.var['gene_symbol'], index=self.adata.obs_names)
        df = (df - df.mean()) / df.std()
        df = df.loc[:, df.columns.isin(sorted_genes)]
        df['cluster'] = self.predict_labels
        df = df.groupby('cluster').mean()
        df = df[sorted_genes]
        df = df.loc[order_layer]
        sns.set(style="white")
        g = sns.clustermap(df, center=0, cmap="vlag",
                           dendrogram_ratio=(.1, .2),
                           cbar_pos=(.02, .32, .03, .2),
                           row_cluster=False,
                           linewidths=.75, figsize=(12, 5))
        g.ax_row_dendrogram.remove()
        plt.xlabel('Gene')
        if self.img_path is not None:
            plt.savefig(self.img_path + 'marker_heatmap.png', dpi=300)
        plt.show()

    def plot_marker_gene(self, markers, order_layer, show=True):
        if 'gene_symbol' not in self.adata.var.keys():
            df = pd.DataFrame(self.adata.X.T, columns=self.adata.obs_names, index=self.adata.var.index)
        else:
            df = pd.DataFrame(self.adata.X.T, index=self.adata.var['gene_symbol'], columns=self.adata.obs_names)
        df = df.loc[markers].T
        df['Layer'] = ["Layer_" + str(l) for l in self.predict_labels]
        sns.set(style="white")
        fig, axes = plt.subplots(nrows=1, ncols=len(markers), figsize=(12, 6), sharey=True)
        genes = [i for i in df.columns.unique() if i != 'Layer']
        for i, gene in enumerate(genes):
            summary_stats = df[[gene, 'Layer']].reset_index().groupby('Layer')[gene].mean().reindex(order_layer)
            axes[i].plot(summary_stats, order_layer, marker='o', color='black')
            axes[i].set_title(gene, fontsize=25, fontweight='bold')
        plt.tight_layout()
        if self.img_path is not None:
            plt.savefig(self.img_path + 'marker_gene.png', dpi=300)
        if show:
            plt.show()
        else:
            return fig, axes


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
        adj_mask = cp.array(adj_mask.cpu().detach())
    tri_pairs = []
    for row_idx in range(adj_mask.shape[0]):
        col_indices = cp.where(adj_mask[row_idx])[0]
        if len(col_indices) >= 2:
            col_pairs = cp.array(cp.meshgrid(col_indices, col_indices)).T.reshape(-1, 2)
            col_pairs = col_pairs[col_pairs[:, 0] != col_pairs[:, 1]]
            tri_pairs.extend([[row_idx, col_pair[0], col_pair[1]] for col_pair in cp.asnumpy(col_pairs)])
    print('The number of triplets is {}'.format(len(tri_pairs)))
    rows, cols = cp.where(adj_mask)
    pairs = np.vstack((rows, cols)).T.tolist()
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
        all_pairs = cp.array(all_pairs)
        sorted_pairs_cp = cp.sort(all_pairs, axis=1)
        sorted_pairs = sorted_pairs_cp.get()
        dual_entropy_matrix = get_dual_joint_entropy_matrix(sorted_pairs, gene_exprs)
        dual_entropy_matrix.to_csv(save_path + 'dual_entropy_matrix.csv')
    else:
        dual_entropy_matrix = pd.read_csv(save_path + 'dual_entropy_matrix.csv', index_col=0).values
    if not os.path.exists(save_path + 'triple_entropy_matrix.csv'):
        sorted_tri_pairs = cp.array(all_tri_pairs)
        sorted_tri_pairs_cp = cp.sort(sorted_tri_pairs, axis=1)
        sorted_tri_pairs = sorted_tri_pairs_cp.get()
        triple_entropy_matrix = get_triple_joint_entropy_matrix(sorted_tri_pairs, gene_exprs)
        triple_entropy_matrix.to_csv(save_path + 'triple_entropy_matrix.csv')
    else:
        triple_entropy_matrix = pd.read_csv(save_path + 'triple_entropy_matrix.csv', index_col=0).values
    condition_mi = get_conditional_mutual_info_matrix(all_tri_pairs, entropy_matrix, dual_entropy_matrix,
                                                      triple_entropy_matrix)
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


def connect_graph(adj, data, start_cell):
    # TODO: Update the heuristic here which involves using the
    # cell with the max distance to establish a connection with
    # the disconnected parts of the clusters.

    index = adj.index
    dists = pd.Series(dijkstra(adj, indices=np.where(data.index == start_cell)[0][0]), index=index)
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
        unreachable_dists = pd.Series(np.ravel(unreachable_dists), index=unreachable_nodes)
        # Add edge between farthest reacheable and its nearest unreachable
        adj.loc[farthest_reachable_id, unreachable_dists.idxmin()] = unreachable_dists.min()
        # Recompute distances to early cell
        dists = pd.Series(dijkstra(adj, indices=np.where(data.index == start_cell)[0][0]), index=index)
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


def run_pseudotime(
        data, early_cell, terminal_states=None,
        knn: int = 30, num_waypoints: int = 1200, n_jobs: int = -1,
        scale_components: bool = True, use_early_cell_as_start: bool = False,
        max_iterations: int = 25, eigvec_key: str = "DM_EigenVectors_multiscaled", seed: int = 20, ):
    """
    Executes the Palantir algorithm to derive pseudotemporal ordering of cells, their fate probabilities, and
    state entropy based on the multiscale diffusion map results.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Either a DataFrame of multiscale space diffusion components or a Scanpy AnnData object.
    early_cell : str
        Start cell for pseudotime construction.
    terminal_states : List/Series/Dict, optional
        User-defined terminal states structure in the format {terminal_name:cell_name}. Default is None.
    knn : int, optional
        Number of nearest neighbors for graph construction. Default is 30.
    num_waypoints : int, optional
        Number of waypoints to sample. Default is 1200.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is -1.
    scale_components : bool, optional
        If True, components are scaled. Default is True.
    use_early_cell_as_start : bool, optional
        If True, the early cell is used as start. Default is False.
    max_iterations : int, optional
        Maximum number of iterations for pseudotime convergence. Default is 25.
    eigvec_key : str, optional
        Key to access multiscale space diffusion components from obsm of the AnnData object. Default is 'DM_EigenVectors_multiscaled'.
    pseudo_time_key : str, optional
        Key to store the pseudotime in obs of the AnnData object. Default is 'palantir_pseudotime'.
    entropy_key : str, optional
        Key to store the entropy in obs of the AnnData object. Default is 'palantir_entropy'.
    fate_prob_key : str, optional
        Key to store the fate probabilities in obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
        If save_as_df is True, the fate probabilities are stored as pandas DataFrame with terminal state names as columns.
        If False, the fate probabilities are stored as numpy array and the terminal state names are stored in uns[fate_prob_key + "_columns"].
    save_as_df : bool, optional
        If True, the fate probabilities are saved as pandas DataFrame. If False, the data is saved as numpy array.
        The option to save as DataFrame is there due to some versions of AnnData not being able to
        write h5ad files with DataFrames in ad.obsm. Default is palantir.SAVE_AS_DF = True.
    waypoints_key : str, optional
        Key to store the waypoints in uns of the AnnData object. Default is 'palantir_waypoints'.
    seed : int, optional
        The seed for the random number generator used in waypoint sampling. Default is 20.

    Returns
    -------
    Optional[PResults]
        PResults object with pseudotime, entropy, branch probabilities, and waypoints.
        If an AnnData object is passed as data, the result is written to its obs, obsm, and uns attributes
        using the provided keys and None is returned.
    """

    if isinstance(terminal_states, dict):
        terminal_states = pd.Series(terminal_states)
    if isinstance(terminal_states, pd.Series):
        terminal_cells = terminal_states.index.values
    else:
        terminal_cells = terminal_states
    if isinstance(data, sc.AnnData):
        ms_data = pd.DataFrame(data.obsm[eigvec_key], index=data.obs_names)
    else:
        ms_data = data

    if scale_components:
        data_df = pd.DataFrame(
            preprocessing.minmax_scale(ms_data),
            index=ms_data.index, columns=ms_data.columns)
    else:
        data_df = copy.copy(ms_data)
    # ################################################
    # Determine the boundary cell closest to user defined early cell
    dm_boundaries = pd.Index(set(data_df.idxmax()).union(data_df.idxmin()))
    dists = pairwise_distances(
        data_df.loc[dm_boundaries, :], data_df.loc[early_cell, :].values.reshape(1, -1))
    start_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()
    if use_early_cell_as_start:
        start_cell = early_cell
    # Sample waypoints
    print("Sampling and flocking waypoints...")
    start = time()

    # Append start cell
    if isinstance(num_waypoints, int):
        waypoints = _max_min_sampling(data_df, num_waypoints, seed)
    else:
        waypoints = num_waypoints
    waypoints = waypoints.union(dm_boundaries)
    if terminal_cells is not None:
        waypoints = waypoints.union(terminal_cells)
    waypoints = pd.Index(waypoints.difference([start_cell]).unique())

    # Append start cell
    waypoints = pd.Index([start_cell]).append(waypoints)
    end = time()
    print("Time for determining waypoints: {} minutes".format((end - start) / 60))

    # pseudotime and weighting matrix
    print("Determining pseudotime...")
    pseudotime, W = _compute_pseudotime(
        data_df, start_cell, knn, waypoints, n_jobs, max_iterations)
    return pseudotime


def _max_min_sampling(data, num_waypoints, seed=None):
    """Function for max min sampling of waypoints

    :param data: Data matrix along which to sample the waypoints,
                 usually diffusion components
    :param num_waypoints: Number of waypoints to sample
    :param seed: Random number generator seed to find initial guess.
    :return: pandas Series reprenting the sampled waypoints
    """

    waypoint_set = list()
    no_iterations = int((num_waypoints) / data.shape[1])
    if seed is not None:
        np.random.seed(seed)

    # Sample along each component
    N = data.shape[0]
    for ind in data.columns:
        # Data vector
        vec = np.ravel(data[ind])

        # Random initialzlation
        iter_set = [
            np.random.randint(N),
        ]

        # Distances along the component
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis=1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)

            # Update distances
            dists[:, k] = abs(vec - data[ind].values[new_wp])

        # Update global set
        waypoint_set = waypoint_set + iter_set

    # Unique waypoints
    waypoints = data.index[waypoint_set].unique()

    return waypoints


def _compute_pseudotime(data, start_cell, knn, waypoints, n_jobs, max_iterations=25):
    """Function for compute the pseudotime

    :param data: Multiscale space diffusion components
    :param start_cell: Start cell for pseudotime construction
    :param knn: Number of nearest neighbors for graph construction
    :param waypoints: List of waypoints
    :param n_jobs: Number of jobs for parallel processing
    :param max_iterations: Maximum number of iterations for pseudotime convergence
    :return: pseudotime and weight matrix
    """

    # ################################################
    # Shortest path distances to determine trajectories
    print("Shortest path distances using {}-nearest neighbor graph...".format(knn))
    start = time()
    nbrs = NearestNeighbors(n_neighbors=knn, metric="euclidean", n_jobs=n_jobs).fit(data)
    adj = nbrs.kneighbors_graph(data, mode="distance")
    # Connect graph if it is disconnected
    adj = _connect_graph(adj, data, np.where(data.index == start_cell)[0][0])
    # Distances
    dists = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_shortest_path_helper)(np.where(data.index == cell)[0][0], adj)
        for cell in waypoints)
    # Convert to distance matrix
    D = pd.DataFrame(0.0, index=waypoints, columns=data.index)
    for i, cell in enumerate(waypoints):
        D.loc[cell, :] = pd.Series(
            np.ravel(dists[i]), index=data.index[dists[i].index])[data.index]
    end = time()
    print("Time for shortest paths: {} minutes".format((end - start) / 60))

    # ###############################################
    # Determine the perspective matrix

    print("Iteratively refining the pseudotime...")
    # Waypoint weights
    sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    # Stochastize the matrix
    W = W / W.sum()

    # Initalize pseudotime to start cell distances
    pseudotime = D.loc[start_cell, :]
    converged = False

    # Iteratively update perspective and determine pseudotime
    iteration = 1
    while not converged and iteration < max_iterations:
        # Perspective matrix by alinging to start distances
        P = deepcopy(D)
        for wp in waypoints[1:]:
            # Position of waypoints relative to start
            idx_val = pseudotime[wp]

            # Convert all cells before starting point to the negative
            before_indices = pseudotime.index[pseudotime < idx_val]
            P.loc[wp, before_indices] = -D.loc[wp, before_indices]

            # Align to start
            P.loc[wp, :] = P.loc[wp, :] + idx_val

        # Weighted pseudotime
        new_traj = P.multiply(W).sum()

        # Check for convergence
        corr = pearsonr(pseudotime, new_traj)[0]
        print("Correlation at iteration %d: %.4f" % (iteration, corr))
        if corr > 0.9999:
            converged = True

        # If not converged, continue iteration
        pseudotime = new_traj
        iteration += 1

    pseudotime -= np.min(pseudotime)
    pseudotime /= np.max(pseudotime)

    return pseudotime, W


def _shortest_path_helper(cell, adj):
    return pd.Series(dijkstra(adj, False, cell))


def _connect_graph(adj, data, start_cell):
    # Create graph and compute distances
    graph = nx.Graph(adj)
    dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
    dists = pd.Series(dists.values, index=data.index[dists.index])

    # Idenfity unreachable nodes
    unreachable_nodes = data.index.difference(dists.index)

    # Connect unreachable nodes
    while len(unreachable_nodes) > 0:
        farthest_reachable = np.where(data.index == dists.idxmax())[0][0]

        # Compute distances to unreachable nodes
        unreachable_dists = pairwise_distances(
            data.iloc[farthest_reachable, :].values.reshape(1, -1),
            data.loc[unreachable_nodes, :],
        )
        unreachable_dists = pd.Series(
            np.ravel(unreachable_dists), index=unreachable_nodes
        )

        # Add edge between farthest reacheable and its nearest unreachable
        add_edge = np.where(data.index == unreachable_dists.idxmin())[0][0]
        adj[farthest_reachable, add_edge] = unreachable_dists.min()

        # Recompute distances to early cell
        graph = nx.Graph(adj)
        dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
        dists = pd.Series(dists.values, index=data.index[dists.index])

        # Idenfity unreachable nodes
        unreachable_nodes = data.index.difference(dists.index)

    return adj


def determine_multiscale_space(
        dm_res,
        n_eigs=None,
        eigval_key: str = "DM_EigenValues",
        eigvec_key: str = "DM_EigenVectors",
        out_key: str = "DM_EigenVectors_multiscaled",
):
    """
    Determine the multi-scale space of the data.

    Parameters
    ----------
    dm_res : Union[dict, sc.AnnData]
        Diffusion map results from run_diffusion_maps.
        If sc.AnnData is passed, its uns[eigval_key] and obsm[eigvec_key] are used.
    n_eigs : Union[int, None], optional
        Number of eigen vectors to use. If None is specified, the number
        of eigen vectors will be determined using the eigen gap. Default is None.
    eigval_key : str, optional
        Key to retrieve EigenValues from dm_res if it is a sc.AnnData object. Default is 'DM_EigenValues'.
    eigvec_key : str, optional
        Key to retrieve EigenVectors from dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors'.
    out_key : str, optional
        Key to store the result in obsm of dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors_multiscaled'.

    Returns
    -------
    Union[pd.DataFrame, None]
        Multi-scale data matrix. If sc.AnnData is passed as dm_res, the result
        is written to its obsm[out_key] and None is returned.
    """
    if isinstance(dm_res, sc.AnnData):
        eigenvectors = dm_res.obsm[eigvec_key]
        if not isinstance(eigenvectors, pd.DataFrame):
            eigenvectors = pd.DataFrame(eigenvectors, index=dm_res.obs_names)
        dm_res_dict = {
            "EigenValues": dm_res.uns[eigval_key],
            "EigenVectors": eigenvectors,
        }
    else:
        dm_res_dict = dm_res

    if not isinstance(dm_res_dict, dict):
        raise ValueError("'dm_res' should be a dict or a sc.AnnData instance")
    if n_eigs is None:
        vals = np.ravel(dm_res_dict["EigenValues"])
        n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
        if n_eigs < 3:
            n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 1
        if n_eigs < 3:
            # Fix for #39
            n_eigs = 3

    # Scale the data
    use_eigs = list(range(1, n_eigs))
    eig_vals = np.ravel(dm_res_dict["EigenValues"][use_eigs])
    data = dm_res_dict["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
    data = pd.DataFrame(data, index=dm_res_dict["EigenVectors"].index)

    if isinstance(dm_res, sc.AnnData):
        dm_res.obsm[out_key] = data.values

    return data


def run_diffusion_maps(
        data,
        n_components: int = 10,
        knn: int = 30,
        alpha: float = 0,
        seed=0,
        pca_key: str = "X_pca",
        kernel_key: str = "DM_Kernel",
        sim_key: str = "DM_Similarity",
        eigval_key: str = "DM_EigenValues",
        eigvec_key: str = "DM_EigenVectors"):
    """
    Run Diffusion maps using the adaptive anisotropic kernel.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        PCA projections of the data or adjacency matrix.
        If sc.AnnData is passed, its obsm[pca_key] is used and the result is written to
        its obsp[kernel_key], obsm[eigvec_key], and uns[eigval_key].
    n_components : int, optional
        Number of diffusion components. Default is 10.
    knn : int, optional
        Number of nearest neighbors for graph construction. Default is 30.
    alpha : float, optional
        Normalization parameter for the diffusion operator. Default is 0.
    seed : Union[int, None], optional
        Numpy random seed, randomized if None, set to an arbitrary integer for reproducibility.
        Default is 0.
    pca_key : str, optional
        Key to retrieve PCA projections from data if it is a sc.AnnData object. Default is 'X_pca'.
    kernel_key : str, optional
        Key to store the kernel in obsp of data if it is a sc.AnnData object. Default is 'DM_Kernel'.
    sim_key : str, optional
        Key to store the similarity in obsp of data if it is a sc.AnnData object. Default is 'DM_Similarity'.
    eigval_key : str, optional
        Key to store the EigenValues in uns of data if it is a sc.AnnData object. Default is 'DM_EigenValues'.
    eigvec_key : str, optional
        Key to store the EigenVectors in obsm of data if it is a sc.AnnData object. Default is 'DM_EigenVectors'.

    Returns
    -------
    dict
        Diffusion components, corresponding eigen values and the diffusion operator.
        If sc.AnnData is passed as data, these results are also written to the input object
        and returned.
    """
    from scipy.sparse import issparse

    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(data.obsm[pca_key], index=data.obs_names)
    else:
        data_df = data

    if not isinstance(data_df, pd.DataFrame) and not issparse(data_df):
        raise ValueError("'data_df' should be a pd.DataFrame or sc.AnnData")

    if not issparse(data_df):
        kernel = compute_kernel(data_df, knn, alpha)
    else:
        kernel = data_df

    res = diffusion_maps_from_kernel(kernel, n_components, seed)

    res["kernel"] = kernel
    if not issparse(data_df):
        res["EigenVectors"].index = data_df.index

    if isinstance(data, sc.AnnData):
        data.obsp[kernel_key] = res["kernel"]
        data.obsp[sim_key] = res["T"]
        data.obsm[eigvec_key] = res["EigenVectors"].values
        data.uns[eigval_key] = res["EigenValues"].values

    return res


def diffusion_maps_from_kernel(kernel: csr_matrix, n_components: int = 10, seed=0):
    """
    Compute the diffusion map given a kernel matrix.

    Parameters
    ----------
    kernel : csr_matrix
        Precomputed kernel matrix.
    n_components : int
        Number of diffusion components to compute. Default is 10.
    seed : Union[int, None]
        Seed for random initialization. Default is 0.

    Returns
    -------
    dict
        T-matrix (T), Diffusion components (EigenVectors) and corresponding eigenvalues (EigenValues).
    """
    N = kernel.shape[0]
    D = np.ravel(kernel.sum(axis=1))
    D[D != 0] = 1 / D[D != 0]
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)

    np.random.seed(seed)
    v0 = np.random.rand(min(T.shape))
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000, v0=v0)

    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]

    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

    return {"T": T, "EigenVectors": pd.DataFrame(V), "EigenValues": pd.Series(D)}


def compute_kernel(
        data,
        knn: int = 30,
        alpha: float = 0,
        pca_key: str = "X_pca",
        kernel_key: str = "DM_Kernel") -> csr_matrix:
    """
    Compute the adaptive anisotropic diffusion kernel.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Data points (rows) in a feature space (columns) for pd.DataFrame.
        For sc.AnnData, it uses the .X attribute.
    knn : int
        Number of nearest neighbors for adaptive kernel calculation. Default is 30.
    alpha : float
        Normalization parameter for the diffusion operator. Default is 0.
    pca_key : str, optional
        Key to retrieve PCA projections from data if it is a sc.AnnData object. Default is 'X_pca'.
    kernel_key : str, optional
        Key to store the kernel in obsp of data if it is a sc.AnnData object. Default is 'DM_Kernel'.

    Returns
    -------
    csr_matrix
        Computed kernel matrix.
    """

    # If the input is sc.AnnData, convert it to a DataFrame
    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(data.obsm[pca_key], index=data.obs_names)
    else:
        data_df = data

    N = data_df.shape[0]
    temp = sc.AnnData(data_df.values)
    sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn)
    kNN = temp.obsp["distances"]

    adaptive_k = int(np.floor(knn / 3))
    adaptive_std = np.zeros(N)
    for i in np.arange(N):
        adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i]: kNN.indptr[i + 1]])[
            adaptive_k - 1
            ]

    x, y, dists = find(kNN)
    dists /= adaptive_std[x]
    W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

    kernel = W + W.T

    if alpha > 0:
        D = np.ravel(kernel.sum(axis=1))
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)

    if isinstance(data, sc.AnnData):
        data.obsp[kernel_key] = kernel

    return kernel


def evaluation(embedding, nclasses, labels, clu_model='kmeans'):
    from sklearn.cluster import KMeans

    if clu_model == 'kmeans':
        ari_ls, ami_ls = [], []
        for clu_trial in range(5):
            kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial, n_init="auto").fit(embedding)
            predict_labels = kmeans.predict(embedding)
            cm_all = ClusteringMetrics(labels.cpu().numpy(), predict_labels)
            ari, ami = cm_all.evaluationClusterModelFromLabel()
            ari_ls.append(ari)
            ami_ls.append(ami)
        ari, ami = np.mean(ari_ls), np.mean(ami_ls)
    else:
        raise Exception(f'Unknown cluster model {clu_model}')
    return ari, ami
