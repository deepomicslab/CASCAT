import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_sparse import SparseTensor
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


def get_sketched_cci(adata, num_neighbors=5, is_undirected=True):
    # Get cell-cell communication graph from spatial coordinates
    adjacency = np.zeros(shape=(adata.n_obs, adata.n_obs), dtype=int)
    if 'spatial' in adata.obsm.keys():
        coords = adata.obsm["spatial"]
        dis = distance_matrix(coords, coords)
    else:
        if 'X_pca' not in adata.obsm.keys():
            sc.pp.pca(adata)
        coords = adata.obsm["X_pca"]
        dis = distance_matrix(coords, coords)
    neighbors_idx = np.argsort(dis, axis=1)

    for i, n_idx in enumerate(neighbors_idx):
        n_idx = n_idx[n_idx != i][:num_neighbors]
        adjacency[i, n_idx] = 1
        assert adjacency[i, i] != 1

    if is_undirected:
        print('Use undirected cell-cell communication graph')
        adjacency = ((adjacency + adjacency.T) > 0).astype(int)

    adata.obsp['knn_adj'] = csr_matrix(adjacency)
    return adata, dis


def load_data_from_raw(args):
    adata = sc.read_h5ad(args.adata_file)
    print('Raw adata:', adata, sep='\n')

    if args.hvg:
        sc.pp.filter_genes(adata, min_cells=args.filter_cell)
        print('After flitering: ', adata.shape)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, layer='raw')

    args.norm_target = float(args.norm_target) if args.norm_target is not None else None
    if 'j_delta_x_perturbation' not in adata.layers.keys():
        sc.pp.normalize_total(adata, target_sum=args.norm_target)
        sc.pp.log1p(adata)

    if 'highly_variable' in adata.var.keys():
        adata = adata[:, adata.var['highly_variable']].copy()
    print('Normalization target:', args.norm_target)
    if args.log_transform:
        sc.pp.normalize_per_cell(adata)
        log_transform(adata)
    if scipy.sparse.issparse(adata.X):
        counts = adata.X.toarray()
    else:
        counts = adata.X
    gene_exp = torch.tensor(counts, dtype=torch.float32)
    if 'cluster' not in adata.obs.keys():
        gt_clusters = pd.Series(index=adata.obs_names)
        for obs_name in adata.obs_names:
            res = (adata.uns['milestone_percentages']['cell_id'] == obs_name)
            milestones = adata.uns['milestone_percentages'].loc[res, 'milestone_id']
            percentages = adata.uns['milestone_percentages'].loc[res, 'percentage']
            cluster_id = milestones.loc[percentages.idxmax()]
            gt_clusters.loc[obs_name] = cluster_id
        adata.obs['cluster'] = gt_clusters
    cat = adata.obs['cluster'].astype('category').values
    labels = torch.tensor(cat.codes, dtype=torch.long)
    nclasses = len(cat.categories)

    adata, dis = get_sketched_cci(adata, num_neighbors=args.a_k)
    (row, col), val = from_scipy_sparse_matrix(adata.obsp['knn_adj'])
    num_nodes = adata.obsp['knn_adj'].shape[0]
    adj_knn = SparseTensor(row=row, col=col, value=val.to(torch.float32), sparse_sizes=(num_nodes, num_nodes))
    return gene_exp, labels, nclasses, adj_knn, dis, adata


def log_transform(data, pseudo_count=0.1):
    import anndata
    from scipy.sparse import issparse
    """Log transform the matrix

    :param data: Counts matrix: Cells x Genes or Anndata object
    :return: Log transformed matrix
    """
    if isinstance(data, anndata.AnnData):
        if issparse(data.X):
            data.X.data = np.log2(data.X.data + pseudo_count) - np.log2(pseudo_count)
        else:
            data.X = np.log2(data.X + pseudo_count) - np.log2(pseudo_count)
    else:
        return np.log2(data + pseudo_count)


def load_adata_from_raw(args):
    adata = sc.read_h5ad(args.adata_file)
    print('Raw adata:', adata, sep='\n')
    if (args.hvg if 'hvg' in args else False):
        args.filter_cell = int(args.filter_cell) if 'filter_cell' in args else 0
        sc.pp.filter_genes(adata, min_cells=args.filter_cell)
        print('After flitering: ', adata.shape)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)

    args.norm_target = float(args.norm_target) if ('norm_target' in args) and args.norm_target is not None else 1e4
    print('Normalization target:', args.norm_target)
    sc.pp.normalize_total(adata, target_sum=args.norm_target)
    sc.pp.log1p(adata)

    if 'highly_variable' in adata.var.keys():
        adata = adata[:, adata.var['highly_variable']].copy()

    print('Processed adata:', adata, sep='\n')

    if scipy.sparse.issparse(adata.X):
        counts = adata.X.toarray()
    else:
        counts = adata.X
    adata.X = counts

    if 'cluster' not in adata.obs.keys():
        gt_clusters = pd.Series(index=adata.obs_names)
        for obs_name in adata.obs_names:
            res = (adata.uns['milestone_percentages']['cell_id'] == obs_name)
            milestones = adata.uns['milestone_percentages'].loc[res, 'milestone_id']
            percentages = adata.uns['milestone_percentages'].loc[res, 'percentage']
            cluster_id = milestones.loc[percentages.idxmax()]
            gt_clusters.loc[obs_name] = cluster_id
        adata.obs['cluster'] = gt_clusters
    labels = adata.obs['cluster'].astype('category').values
    return adata, labels, len(labels.categories)


def load_labels_from_emb(args, nclasses, adata, emb_key="metric_embedding",
                         label_key='metric_clusters'):
    emb = np.load(args.emb_path)
    adata.obsm[emb_key] = emb
    predict_labels = KMeans(n_clusters=nclasses, random_state=0, n_init="auto").fit_predict(emb)
    adata.obs[label_key] = predict_labels
    return adata
