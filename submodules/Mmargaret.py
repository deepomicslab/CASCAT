from sklearn.neighbors import NearestNeighbors
import argparse
import sys
sys.path.append('/home/a/yingyingyu/CASCAT')
from submodules.Margret_funs import *
from utils.Utils import run_leiden
from utils.Metrics import caculate_metric, ClusteringMetrics


class Experiment:
    def __init__(self, args):
        self.args = args
        self.neighbor = args.neighbor
        self.predict_key = args.predict_key
        self.adata = self.prepare_data(args.adata_file)
        self.labels = self.adata.obs['gt_cluster']
        self.n_classes = len(set(self.labels))
        self.start_cell_ids_idx = np.where(self.adata.obs_names.values == self.adata.uns['start_id'])[0][0]
        print(f'Start cell index: {self.start_cell_ids_idx}')
        self.img_path = args.img_path
        self.resolution = self.get_resolution()

    def prepare_data(self, data_path):
        data = sc.read_h5ad(data_path)
        data = preprocess_recipe(
            data, min_expr_level=3, min_cells=None,
            use_hvg=False, n_top_genes=1000, scale=True)
        print('Computing PCA...')
        x_pca, _, _ = run_pca(data, random_state=self.args.seed, use_hvg=False, n_components=self.args.n_comp)
        data.obsm['X_pca'] = x_pca
        print(f'Components computed: {str(self.args.n_comp)}')
        if 'gt_cluster' not in data.obs.keys():
            gt_clusters = pd.Series(index=data.obs_names)
            for obs_name in data.obs_names:
                res = (data.uns['milestone_percentages']['cell_id'] == obs_name)
                milestones = data.uns['milestone_percentages'].loc[res, 'milestone_id']
                percentages = data.uns['milestone_percentages'].loc[res, 'percentage']
                cluster_id = milestones.loc[percentages.idxmax()]
                gt_clusters.loc[obs_name] = cluster_id
            data.obs['gt_cluster'] = gt_clusters
        return data

    def get_resolution(self):
        sc.pp.neighbors(self.adata, n_neighbors=self.neighbor, use_rep='X_pca', random_state=self.args.seed)
        resolution, adata = run_leiden(self.adata, n_cluster=self.n_classes, cluster_key=self.predict_key,
                                       random_state=self.args.seed)
        return resolution

    def run_train_emb(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_metric_learner(self.adata, n_episodes=10, n_metric_epochs=10, obsm_data_key='X_pca',
                                 code_size=10, backend='leiden', device='cuda',
                                 cluster_kwargs={'random_state': self.args.seed, 'resolution': self.resolution},
                                 nn_kwargs={'random_state': self.args.seed, 'n_neighbors': self.neighbor},
                                 save_path=self.args.out_path + 'metric_learner.pt',
                                 trainer_kwargs={'optimizer': 'SGD', 'lr': 0.01, 'batch_size': 256,
                                                 'train_loader_kwargs': {'num_workers': 2, 'pin_memory': True,
                                                                         'drop_last': True}},
                                 loss_kwargs={'margin': 1.0, 'p': 2})
        X_embedded = generate_plot_embeddings(self.adata.obsm['metric_embedding'], method='umap',
                                              random_state=self.args.seed)
        self.adata.obsm['X_met_embedding'] = X_embedded

    def get_group_frac(self):
        self.predict_labels = self.adata.obs[self.predict_key].to_numpy().astype(np.int64)
        n_groups = len(set(self.predict_labels))
        n_truegroups = len(set(self.labels))
        sorted_col_ = list(set(self.labels))
        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=sorted_col_)

        cluster_population_dict = {}
        set_labels = list(set(self.predict_labels))
        set_labels.sort()
        for group_i in set_labels:
            loc_i = np.where(self.predict_labels == group_i)[0]
            cluster_population_dict[group_i] = len(loc_i)
            true_label_in_group_i = list(np.asarray(self.labels)[loc_i])
            ll_temp = list(set(true_label_in_group_i))
            for ii in ll_temp:
                group_frac[ii][group_i] = true_label_in_group_i.count(ii)
        group_labels = group_frac.idxmax(axis=1)
        group_labels = {i: group_labels[i] for i in group_labels.keys()}
        return group_labels

    def run_margaret(self):
        self.run_train_emb()
        communities = self.adata.obs[self.predict_key].to_numpy().astype(np.int64)
        X = self.adata.obsm['metric_embedding']
        nbrs = NearestNeighbors(n_neighbors=self.neighbor, metric="euclidean").fit(X)
        adj_dist = nbrs.kneighbors_graph(X, mode="distance", n_neighbors=self.neighbor)
        adj_conn = nbrs.kneighbors_graph(X, n_neighbors=self.neighbor, mode="connectivity")
        un_connectivity, _ = compute_undirected_cluster_connectivity(communities, adj_conn)
        self.adata.uns['metric_connectivities'] = un_connectivity
        G_undirected, node_positions = compute_connectivity_graph(self.adata.obsm['X_met_embedding'],
                                                                  self.adata.obs[self.predict_key],
                                                                  un_connectivity)
        adj_cluster = nx.to_pandas_adjacency(G_undirected)
        self.adata = compute_pseudotime(self.adata, self.start_cell_ids_idx, adj_dist, adj_cluster)
        ax = plot_connectivity_graph(self.adata.obsm['X_met_embedding'], communities,
                                     self.adata.obs['metric_pseudotime'], un_connectivity,
                                     mode='undirected', start_cell_ids=[self.start_cell_ids_idx],
                                     offset=0.2, labels=self.get_group_frac(), cmap='plasma', node_size=800, alpha=0.9,
                                     font_size=20, linewidths=20)
        plot_pseudotime(
            self.adata, embedding_key="X_met_embedding", pseudotime_key="metric_pseudotime", ax=ax,
            s=10, cmap='plasma', figsize=(10, 10), cb_axes_pos=[0.92, 0.55, 0.02, 0.3],
            save_path=os.path.join(self.img_path, f'{self.args.dataname}_margret.png'), save_kwargs={
                'dpi': 300, 'bbox_inches': 'tight', 'transparent': True}, show_colorbar=False)

    def caculate_metrics(self):
        cm_all = ClusteringMetrics(self.labels, self.adata.obs[self.predict_key].to_numpy().astype(np.int64))
        ari, nmi = cm_all.evaluationClusterModelFromLabel()
        print('ARI: ', ari, 'NMI: ', nmi)
        IM, OT, KT, SR = caculate_metric(self.adata)
        print(f'IM: {IM},OT: {OT}, KT: {KT}, SR: {SR}')
        return IM, OT, KT, SR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='real1')
    parser.add_argument('--adata_file', type=str, default='../dataset/scdata/')
    parser.add_argument('--img_path', type=str, default='../img/scimg/')
    parser.add_argument('--out_path', type=str, default='../result/')
    parser.add_argument('--predict_key', type=str, default="metric_clusters")
    parser.add_argument('--neighbor', type=int, default=20)
    parser.add_argument('--n_comp', type=int, default=50, help='10(sim) or 50(real)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    from time import time

    start = time()
    args = parse_arguments()
    args.adata_file = args.adata_file + args.dataname + '/data.h5ad'
    IM_ls, OT_ls, KT_ls, SR_ls = [], [], [], []
    for seed in range(5):
        args.seed = seed
        exp = Experiment(args)
        exp.run_margaret()
        print(f'Running time: {time() - start}')
        IM, OT, KT, SR = exp.caculate_metrics()
        IM_ls.append(IM)
        OT_ls.append(OT)
        KT_ls.append(KT)
        SR_ls.append(SR)
    print(f'IM: {np.mean(IM_ls)},OT: {np.mean(OT_ls)}, KT: {np.mean(KT_ls)}, SR: {np.mean(SR_ls)}')
    metric_path = os.path.join(args.out_path,
                               '{}_margaret_meanIM{:.5f}_std{:.5f}_meanOT{:.5f}_std{:.5f}_meanKT{:.5f}_std{:.5f}_meanSR{:.5f}_std{:.5f}'.format(
                                   args.dataname, np.mean(IM_ls), np.std(IM_ls), np.mean(OT_ls), np.std(OT_ls),
                                   np.mean(KT_ls), np.std(KT_ls), np.mean(SR_ls), np.std(SR_ls)))
    open(metric_path, 'a').close()
