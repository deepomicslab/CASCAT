import networkx as nx
import palantir
import os
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import matplotlib.pyplot as plt
import random
import sys
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sys.path.append('/home/a/yingyingyu/CASCAT')
from utils.Utils import run_louvain
from utils.Metrics import caculate_metric, ClusteringMetrics


class Experiment:
    def __init__(self, args):
        self.args = args
        self.n_comp = args.n_comp
        self.k = args.k
        self.predict_key = 'louvain'
        self.adata = self.load_data(args.adata_file)
        self.nclasses = len(self.adata.obs['cluster'].unique())
        self.labels = self.adata.obs['cluster'].values
        self.start_cell = self.adata.uns['start_id']

    def seed_everything(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def load_data(self, adata_file):
        adata = sc.read_h5ad(adata_file)
        adata.X = adata.X.astype('float64')  # this is not required and results will be comparable without it
        if 'cluster' not in adata.obs.keys():
            gt_clusters = pd.Series(index=adata.obs_names)
            for obs_name in adata.obs_names:
                res = (adata.uns['milestone_percentages']['cell_id'] == obs_name)
                milestones = adata.uns['milestone_percentages'].loc[res, 'milestone_id']
                percentages = adata.uns['milestone_percentages'].loc[res, 'percentage']
                cluster_id = milestones.loc[percentages.idxmax()]
                gt_clusters.loc[obs_name] = cluster_id
            adata.obs['cluster'] = gt_clusters
        return adata

    def plot_trajectory(self, pr_res, connectivities, labels):
        def get_centriods(embed, pesudotime, labels, cluster):
            centroids = {}
            centroids_time = {}
            centroids_label = {}
            for label in np.unique(labels):
                centroids[label] = list(np.mean(embed[labels == label], axis=0))
                centroids_time[label] = np.mean(pesudotime[labels == label])
                centroids_label[label] = cluster[labels == label].value_counts().idxmax()
            return centroids, centroids_time, centroids_label

        pesudotime = pr_res.pseudotime

        data_min, data_max = min(pesudotime), max(pesudotime)
        norm = Normalize(vmin=data_min, vmax=data_max)
        sm = ScalarMappable(cmap=plt.get_cmap('plasma'), norm=norm)
        fig, ax = plt.subplots(figsize=(10, 10))
        embed_data = pd.DataFrame(self.adata.obsm["X_umap"], index=self.adata.obs_names, columns=['UMAP1', 'UMAP2'])
        ax.scatter(embed_data.iloc[:, 0], embed_data.iloc[:, 1],
                   c=[sm.to_rgba(i) for i in pesudotime[embed_data.index]])
        # cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        # cbar.set_label('Pseudotime')
        ax.set_axis_off()
        center, center_time, center_label = get_centriods(embed_data.values, pesudotime, labels,
                                                          self.adata.obs["cluster"].values)
        connectivities = pd.DataFrame(np.tril(connectivities['connectivities'].toarray(), k=-1))
        graph = nx.from_pandas_adjacency(connectivities, create_using=nx.Graph)
        nx.draw_networkx_nodes(graph, pos=center, ax=ax, node_color=[sm.to_rgba(center_time[i]) for i in center.keys()])
        nx.draw_networkx_edges(graph, pos=center, ax=ax, width=2)
        x_min, x_max = min(center.values(), key=lambda x: x[0])[0], max(center.values(), key=lambda x: x[0])[0]
        for idx, pos in center.items():
            ax.text(pos[0] - (x_max - x_min) * 0.02, pos[1], str(center_label[idx]), fontsize=22, color='black')
        return fig

    def run_palantir(self):
        sc.pp.normalize_per_cell(self.adata)
        palantir.preprocess.log_transform(self.adata)
        sc.pp.highly_variable_genes(self.adata, n_top_genes=1500, flavor="cell_ranger")
        sc.pp.pca(self.adata, random_state=self.args.seed)
        dm_res = palantir.utils.run_diffusion_maps(self.adata, n_components=self.n_comp,
                                                   seed=self.args.seed)  # dm_res
        palantir.utils.determine_multiscale_space(self.adata)  # ms data
        self.adata.obsm['X_diffusion'] = dm_res['EigenVectors'].to_numpy()
        sc.pp.neighbors(self.adata, use_rep='X_diffusion', n_neighbors=self.k, random_state=self.args.seed)
        _, adata = run_louvain(self.adata, self.nclasses, cluster_key=self.predict_key,
                               random_state=self.args.seed)
        self.adata.obs[self.predict_key] = self.adata.obs[self.predict_key].astype('category')
        sc.tl.paga(self.adata, groups=self.predict_key)
        pr_res = palantir.core.run_palantir(
            self.adata, self.start_cell, num_waypoints=min(self.adata.shape[0], 1200), seed=self.args.seed)
        sc.tl.umap(self.adata, random_state=self.args.seed)
        fig = self.plot_trajectory(pr_res, self.adata.uns['paga'], self.adata.obs[self.predict_key])
        fig.savefig(os.path.join(self.args.img_path, f'{self.args.dataname}_palantir.png'))
        # plt.show()

    def caculate_metrics(self):
        cm_all = ClusteringMetrics(self.labels, self.adata.obs[self.predict_key].values)
        ari, nmi = cm_all.evaluationClusterModelFromLabel()
        print('ARI: ', ari, 'NMI: ', nmi)
        IM, OT, KT, SR = caculate_metric(self.adata, psedo_key="palantir_pseudotime", adj_key="paga")
        print("IM: {}, OT:{}, KT: {}, SR: {}".format(IM, OT, KT, SR))
        return IM, OT, KT, SR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default="real1")
    parser.add_argument('--adata_file', type=str, default="../dataset/scdata/")
    parser.add_argument('--img_path', type=str, default='../img/scimg/')
    parser.add_argument('--out_path', type=str, default='../result/')
    parser.add_argument('--n_comp', type=int, default=15)
    parser.add_argument('--k', type=int, default=20)
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
        experiment = Experiment(args)
        experiment.run_palantir()
        print(f'Running time: {time() - start}')
        IM, OT, KT, SR = experiment.caculate_metrics()
        IM_ls.append(IM)
        OT_ls.append(OT)
        KT_ls.append(KT)
        SR_ls.append(SR)
    print("IM: {}, OT: {}, KT: {}, SR: {}".format(np.mean(IM_ls), np.mean(OT_ls), np.mean(KT_ls), np.mean(SR_ls)))
    metric_path = os.path.join(args.out_path,
                               '{}_palantir_meanIM{:.5f}_std{:.5f}_meanOT{:.5f}_std{:.5f}_meanKT{:.5f}_std{:.5f}_meanSR{:.5f}_std{:.5f}'.format(
                                   args.dataname, np.mean(IM_ls), np.std(IM_ls), np.mean(OT_ls), np.std(OT_ls),
                                   np.mean(KT_ls), np.std(KT_ls), np.mean(SR_ls), np.std(SR_ls)))
    # open(metric_path, 'a').close()
