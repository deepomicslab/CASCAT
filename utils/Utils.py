import scanpy as sc
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from scipy.sparse.csgraph import dijkstra


def preprocess(adata):
    print('Preprocessing...')
    adata.X = adata.X.toarray()
    # sc.pp.filter_cells(adata, min_counts=3)
    # sc.pp.normalize_total(adata)
    # log_transform(adata)
    # sc.pp.scale(adata)
    # dygen
    sc.pp.normalize_total(adata, target_sum=1e3)
    sc.pp.log1p(adata)
    return adata


def denoise(adata, n_comps=10, gt_key="cluster"):
    # Apply PCA on the raw data for initial denoising
    gt_clusters = pd.Series(index=adata.obs_names)
    for obs_name in adata.obs_names:
        res = (adata.uns['milestone_percentages']['cell_id'] == obs_name)
        milestones = adata.uns['milestone_percentages'].loc[res, 'milestone_id']
        percentages = adata.uns['milestone_percentages'].loc[res, 'percentage']
        cluster_id = milestones.loc[percentages.idxmax()]
        # cluster_id = adata.obs[gt_key].loc[obs_name]
        gt_clusters.loc[obs_name] = cluster_id
    adata.obs['gt_clusters'] = gt_clusters
    # print('Computing PCA...')
    # sc.tl.pca(adata, svd_solver='arpack', n_comps=n_comps)
    # print(f'Components computed: {n_comps}')
    # print('Computing UMAP...')
    # sc.pp.neighbors(adata, use_rep='X_pca')
    # sc.tl.umap(adata)
    # print('UMAP computed.')
    return adata


def log_transform(data, pseudo_count=1):
    """Perform log-transformation of scRNA data

    Args:
        data ([sc.AnnData, np.ndarray, pd.DataFrame]): Input data
        pseudo_count (int, optional): [description]. Defaults to 1.
    """
    if type(data) is sc.AnnData:
        data.X = np.log2(data.X + pseudo_count) - np.log2(pseudo_count)
    else:
        return np.log2(data + pseudo_count) - np.log2(pseudo_count)


def cluster_analysis(adata):
    from sklearn import metrics
    print(f'ARI: {metrics.adjusted_rand_score(adata.obs["gt_clusters"], adata.obs["metric_clusters"])}')
    gt_clusters = pd.DataFrame(adata.obs['gt_clusters'].values, index=adata.obs_names)
    metric_clusters = pd.DataFrame(adata.obs['metric_clusters'].values, index=adata.obs_names)
    cluster_df = pd.concat([gt_clusters, metric_clusters], axis=1)
    cluster_df.columns = ['gt_clusters', 'metric_clusters']
    cluster_df = cluster_df.groupby('metric_clusters')['gt_clusters'].value_counts(normalize=True)
    for i in range(len(set(adata.obs['metric_clusters']))):
        print(i, cluster_df[i])


def determine_cell_clusters(data, obsm_key="X_pca", backend="leiden", cluster_key="metric_clusters", resolution=1.0):
    """Run clustering of cells"""
    if not isinstance(data, sc.AnnData):
        raise Exception(f"Expected data to be of type sc.AnnData found : {type(data)}")
    try:
        X = data.obsm[obsm_key]
    except KeyError:
        raise Exception(f"Either `X_pca` or `{obsm_key}` must be set in the data")
    if backend == "kmeans":
        kmeans = KMeans()
        data.obs[cluster_key] = kmeans.fit_predict(X).astype(np.int64)
    elif backend == "louvain":
        sc.pp.neighbors(data, use_rep=obsm_key)
        sc.tl.louvain(data, key_added=cluster_key, resolution=resolution)
        data.obs[cluster_key] = data.obs[cluster_key].to_numpy().astype(np.int64)
    elif backend == "leiden":
        sc.pp.neighbors(data, use_rep=obsm_key)
        sc.tl.leiden(data, key_added=cluster_key, resolution=resolution)
        data.obs[cluster_key] = data.obs[cluster_key].to_numpy().astype(np.int64)
    else:
        raise NotImplementedError(f"The backend {backend} is not supported yet!")
    return data


def run_leiden(adata, n_cluster, range_min=0, range_max=3, max_steps=30, cluster_key='metric_clusters', random_state=0):
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    this_resolution = 1
    while this_step < max_steps:
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.leiden(adata, resolution=this_resolution, key_added=cluster_key, random_state=random_state)
        adata.obs[cluster_key] = adata.obs[cluster_key].to_numpy().astype(np.int64)
        this_clusters = adata.obs[cluster_key].nunique()
        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f" % (n_cluster, this_resolution))
            return this_resolution, adata
        this_step += 1
    return this_resolution, adata


def run_louvain(adata, n_cluster, range_min=0, range_max=3, max_steps=30, cluster_key='metric_clusters',
                random_state=0):
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    this_resolution = 1
    while this_step < max_steps:
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.louvain(adata, resolution=this_resolution, key_added=cluster_key, random_state=random_state)
        adata.obs[cluster_key] = adata.obs[cluster_key].to_numpy().astype(np.int64)
        this_clusters = adata.obs[cluster_key].nunique()
        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f" % (n_cluster, this_resolution))
            return this_resolution, adata
        this_step += 1
    return this_resolution, adata


def compute_pseudotime(
        ad, start_cell_ids, adj_dist, adj_cluster, comm_key="metric_clusters",
        data_key="metric_embedding", pseudotime_key="metric_pseudotime"
):
    communities = ad.obs[comm_key]
    cluster_ids = np.unique(communities)
    data = pd.DataFrame(ad.obsm[data_key], index=ad.obs_names)
    # Create cluster index
    clusters = {}
    for idx in cluster_ids:
        cluster_idx = communities == idx
        clusters[idx] = cluster_idx

    # Prune the initial adjacency matrix
    adj_dist = pd.DataFrame(adj_dist.todense(), index=ad.obs_names, columns=ad.obs_names)
    adj_dist_pruned = adj_dist.copy()
    # adj_dist_pruned = prune_network_edges(communities, adj_dist, adj_cluster)
    # Pseudotime computation on the pruned graph
    start_indices = [np.where(ad.obs_names == s)[0][0] for s in start_cell_ids]
    p = dijkstra(adj_dist_pruned.to_numpy(), indices=start_indices, min_only=True, directed=False)
    pseudotime = pd.Series(p, index=ad.obs_names)
    # update dis in clusters
    for _, cluster in clusters.items():
        p_cluster = pseudotime.loc[cluster]
        cluster_start_cell = p_cluster.idxmin()
        adj_sc = adj_dist_pruned.loc[cluster, cluster]
        adj_sc = connect_graph(
            adj_sc,
            data.loc[cluster, :],
            np.where(adj_sc.index == cluster_start_cell)[0][0])
        adj_dist_pruned.loc[cluster, cluster] = adj_sc
    # Recompute the pseudotime with the updated graph
    p = dijkstra(adj_dist_pruned, indices=start_indices, min_only=True)
    pseudotime = pd.Series(p, index=ad.obs_names)
    # Set the pseudotime for unreachable cells to 0
    pseudotime[pseudotime == np.inf] = 0
    # Add pseudotime to annotated data object
    ad.obs[pseudotime_key] = pseudotime
    return ad


def prune_network_edges(communities, adj_sc, adj_cluster):
    cluster_ids = np.unique(communities)
    # Create cluster index
    clusters = {}
    for idx in cluster_ids:
        cluster_idx = communities == idx
        clusters[idx] = cluster_idx

    col_ids = adj_cluster.columns
    for c_idx in adj_cluster.index:
        cluster_i = clusters[c_idx]
        non_connected_clusters = col_ids[adj_cluster.loc[c_idx, :] == 0]
        for nc_idx in non_connected_clusters:
            if nc_idx == c_idx:
                continue
            cluster_nc = clusters[nc_idx]
            # Prune (remove the edges between two non-connected clusters)
            adj_sc.loc[cluster_i, cluster_nc] = 0
    return adj_sc


def connect_graph(adj, data, start_cell_id):
    from sklearn.metrics import pairwise_distances
    # TODO: Update the heuristic here which involves using the
    # cell with the max distance to establish a connection with
    # the disconnected parts of the clusters.
    index = adj.index
    dists = pd.Series(dijkstra(adj, indices=start_cell_id), index=index)
    unreachable_nodes = index[dists == np.inf]
    print(f'Unreachable nodes: {len(unreachable_nodes)}')
    if len(unreachable_nodes) == 0:
        return adj
    while len(unreachable_nodes) > 0:  # Connect unreachable nodes
        farthest_reachable_id = dists.loc[index[dists != np.inf]].idxmax()
        # Compute distances to unreachable nodes
        unreachable_dists = pairwise_distances(
            data.loc[farthest_reachable_id, :].values.reshape(1, -1),
            data.loc[unreachable_nodes, :],
        )
        unreachable_dists = pd.Series(
            np.ravel(unreachable_dists), index=unreachable_nodes
        )
        # Add edge between farthest reacheable and its nearest unreachable
        adj.loc[
            farthest_reachable_id, unreachable_dists.idxmin()
        ] = unreachable_dists.min()
        # Recompute distances to early cell
        dists = pd.Series(dijkstra(adj, indices=start_cell_id), index=index)
        # Idenfity unreachable nodes
        unreachable_nodes = index[dists == np.inf]
    return adj
