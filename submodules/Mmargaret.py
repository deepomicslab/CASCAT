from sklearn.neighbors import NearestNeighbors
import argparse
import sys

# sys.path.append(r'C:\Users\yingyinyu3\PycharmProjects\STCMI')
# sys.path.append('/mnt/c/Users/yingyinyu3/PycharmProjects/STCMI')
sys.path.append('/home/a/yingyingyu/STCMI')
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
            use_hvg=False, n_top_genes=1500, scale=True)
        print('Computing PCA...')
        x_pca, _, _ = run_pca(data, random_state=self.args.seed, use_hvg=False)
        data.obsm['X_pca'] = x_pca
        print(f'Components computed: {50}')
        sc.pp.neighbors(data, n_neighbors=self.neighbor, use_rep='X_pca', n_pcs=20, random_state=self.args.seed)
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
        resolution, adata = run_leiden(self.adata, n_cluster=self.n_classes, cluster_key=self.predict_key,
                                       random_state=self.args.seed)
        return resolution

    def run_train_emb(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_metric_learner(self.adata, n_episodes=10, n_metric_epochs=5, obsm_data_key='X_pca',
                                 code_size=10, backend='leiden', device='cuda',
                                 cluster_kwargs={'random_state': self.args.seed, 'resolution': 1},
                                 nn_kwargs={'random_state': self.args.seed, 'n_neighbors': self.neighbor},
                                 save_path=self.args.out_path + 'metric_learner.pt',
                                 trainer_kwargs={'optimizer': 'SGD', 'lr': 0.01, 'batch_size': 256,
                                                 'train_loader_kwargs': {'num_workers': 16, 'pin_memory': True,
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
        adj_dist = nbrs.kneighbors_graph(X, mode="distance")
        adj_conn = nbrs.kneighbors_graph(X)
        un_connectivity, _ = compute_undirected_cluster_connectivity(communities, adj_conn)
        self.adata.uns['metric_connectivities'] = un_connectivity
        # plot topology
        ax = plot_connectivity_graph(self.adata.obsm['X_met_embedding'], communities, un_connectivity,
                                     mode='undirected', start_cell_ids=[self.start_cell_ids_idx],
                                     offset=0.2, labels=self.get_group_frac(), cmap='summer')
        G_undirected, node_positions = compute_connectivity_graph(self.adata.obsm['X_met_embedding'],
                                                                  self.adata.obs[self.predict_key],
                                                                  un_connectivity)
        adj_cluster = nx.to_pandas_adjacency(G_undirected)
        self.adata = compute_pseudotime(self.adata, self.start_cell_ids_idx, adj_dist, adj_cluster)
        plot_pseudotime(
            self.adata, ax=ax, embedding_key="X_met_embedding", pseudotime_key="metric_pseudotime",
            s=2, cmap='plasma', figsize=(8, 8), cb_axes_pos=[0.92, 0.55, 0.02, 0.3],
            save_path=self.img_path + 'margret.png', save_kwargs={
                'dpi': 300,
                'bbox_inches': 'tight',
                'transparent': True
            }
        )

    def caculate_metrics(self):
        cm_all = ClusteringMetrics(self.labels, self.adata.obs[self.predict_key].to_numpy().astype(np.int64))
        ari, nmi = cm_all.evaluationClusterModelFromLabel()
        print('ARI: ', ari, 'NMI: ', nmi)
        IM, KT, SR = caculate_metric(self.adata)
        print(f'IM: {IM}, KT: {KT}, SR: {SR}')
        return IM, KT, SR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='binary3')
    parser.add_argument('--adata_file', type=str, default='../dataset/scdata/')
    parser.add_argument('--img_path', type=str, default='../img/')
    parser.add_argument('--out_path', type=str, default='../result/')
    parser.add_argument('--predict_key', type=str, default="metric_clusters")
    parser.add_argument('--neighbor', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    args.adata_file = args.adata_file + args.dataname + '/data.h5ad'
    args.img_path = args.img_path + args.dataname + "/"
    args.out_path = args.out_path + args.dataname + "/"
    IM_ls, KT_ls, SR_ls = [], [], []
    for seed in range(1):
        args.seed = seed
        exp = Experiment(args)
        exp.run_margaret()
        IM, KT, SR = exp.caculate_metrics()
        IM_ls.append(IM)
        KT_ls.append(KT)
        SR_ls.append(SR)
    print(f'IM: {np.mean(IM_ls)}, KT: {np.mean(KT_ls)}, SR: {np.mean(SR_ls)}')
    # metric_path = os.path.join(args.out_path,
    #                            'margaret_meanIM{:.5f}_stdIM{:.5f}_meanKT{:.5f}_stdKT{:.5f}_meanSR{:.5f}_stdSR{:.5f}'.format(
    #                                np.mean(IM_ls), np.std(IM_ls), np.mean(KT_ls), np.std(KT_ls), np.mean(SR_ls),
    #                                np.std(SR_ls)))
    # open(metric_path, 'a').close()
