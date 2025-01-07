import sys
import os
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import pyVIA.core as via
from pyVIA.plotting_via import plot_piechart_viagraph, plot_atlas_view, plot_trajectory_curves, via_atlas_emb, \
    plot_trajectory_curves

sys.path.append('/mnt/c/Users/yingyinyu3/PycharmProjects/CASCAT')
from utils.Utils import run_leiden, preprocess, denoise
from utils.Metrics import caculate_metric, ClusteringMetrics


class Experiment:
    def __init__(self, args):
        self.args = args
        self.seed_everything(args.seed)
        self.adata = self.prepare_data(args.adata_file)
        self.labels = self.adata.obs['cluster']
        self.n_neighbors = args.n_neighbors
        self.nclasses = len(set(self.labels))
        self.resolution, _ = run_leiden(self.adata, self.nclasses, range_min=0, range_max=3,
                                        max_steps=30, cluster_key='metric_clusters', random_state=args.seed)
        self.start_idx = np.where(self.adata.obs_names == self.adata.uns['start_id'])[0][0]
        self.img_path = args.img_path

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def prepare_data(self, adata_file):
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
        adata = preprocess(adata)
        adata = denoise(adata)
        sc.tl.pca(adata, svd_solver='arpack', random_state=self.args.seed)
        sc.pp.neighbors(adata, n_neighbors=self.args.n_neighbors, random_state=self.args.seed, n_pcs=20)
        return adata

    def save_subfig(self, fig, ax, save_path):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.5, 1.5)
        extent = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(save_path, bbox_inches=extent)
        # plt.show()
        plt.close(fig)

    def get_group_frac(self, via_object):
        n_groups = len(set(via_object.labels))
        n_truegroups = len(set(via_object.true_label))
        sorted_col_ = list(set(via_object.true_label))
        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=sorted_col_)

        via_object.cluster_population_dict = {}
        set_labels = list(set(via_object.labels))
        set_labels.sort()
        for group_i in set_labels:
            loc_i = np.where(via_object.labels == group_i)[0]
            via_object.cluster_population_dict[group_i] = len(loc_i)
            true_label_in_group_i = list(np.asarray(via_object.true_label)[loc_i])
            ll_temp = list(set(true_label_in_group_i))
            for ii in ll_temp:
                group_frac[ii][group_i] = true_label_in_group_i.count(ii)
        return group_frac

    def set_cluster_text(self, ax, group_frac, node_pos):
        for i in range(len(group_frac)):
            x, y = node_pos[i]
            t = group_frac.iloc[i].idxmax()
            ax.text(x, y, str(t), fontsize=25, color='black', ha='right', va='bottom')
        return ax

    def run_via(self):
        start_ids = [self.start_idx]
        v0 = via.VIA(self.adata.obsm['X_pca'], self.labels,
                     knn=self.args.n_neighbors, too_big_factor=0.3, edgepruning_clustering_resolution=0.15,
                     edgepruning_clustering_resolution_local=1,
                     root_user=start_ids, random_seed=self.args.seed, resolution_parameter=self.resolution,
                     # preserve_disconnected_after_pruning=True #if run stop then change
                     )
        v0.run_VIA()
        self.adata.obs['metric_pseudotime'] = v0.single_cell_pt_markov
        self.adata.uns['metric_connectivities'] = v0.cluster_adjacency + v0.cluster_adjacency.T
        v0.embedding = via_atlas_emb(via_object=v0, init_pos='via')
        fig, ax, ax1 = plot_piechart_viagraph(v0, ax_text=False, dpi=300, cmap='plasma', show_legend=False,
                                              pie_size_scale=0, fontsize=20, alpha_edge=0.9, linewidth_edge=2,
                                              headwidth_arrow=0.3, size_node_notpiechart=2)
        fig.set_size_inches(21, 10)
        ax.set_title('')
        cbar = fig.colorbar(ax1.collections[0], ax=ax1)
        cbar.remove()
        ax1 = self.set_cluster_text(ax1, self.get_group_frac(v0), v0.graph_node_pos)
        self.save_subfig(fig, ax1, os.path.join(self.img_path, f'{self.args.dataname}_via.png'))
        return v0

    def caculate_metrics(self, v0):
        cm_all = ClusteringMetrics(self.labels, v0.labels)
        ari, nmi = cm_all.evaluationClusterModelFromLabel()
        print('ARI: ', ari, 'NMI: ', nmi)
        IM, OT, KT, SR = caculate_metric(self.adata)
        print(f'IM: {IM}, OT: {OT}, KT: {KT}, SR: {SR}')
        return IM, OT, KT, SR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default="real1")
    parser.add_argument('--adata_file', type=str, default="./dataset/scdata/")
    parser.add_argument('--img_path', type=str, default='./img/scimg/')
    parser.add_argument('--out_path', type=str, default='./result/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_neighbors', type=int, default=20)
    return parser.parse_args()


if __name__ == '__main__':
    from time import time

    start = time()
    args = parse_arguments()
    args.adata_file = args.adata_file + args.dataname + '/data.h5ad'
    IM_ls, OT_ls, KT_ls, SR_ls = [], [], [], []
    for seed in range(5):
        args.seed = seed + 1
        experiment = Experiment(args)
        v0 = experiment.run_via()
        print(f'Running time: {time() - start}')
        IM, OT, KT, SR = experiment.caculate_metrics(v0)
        IM_ls.append(IM)
        OT_ls.append(OT)
        KT_ls.append(KT)
        SR_ls.append(SR)
    # pesudo = v0.single_cell_pt_markov
    # np.save(os.path.join(args.out_path, 'via_pesudo.npy'), pesudo)
    print(f'IM: {np.mean(IM_ls)}, OT: {np.mean(OT_ls)}, KT: {np.mean(KT_ls)}, SR: {np.mean(SR_ls)}')
    metric_path = os.path.join(args.out_path,
                               '{}_via_meanIM{:.5f}_std{:.5f}_meanOT{:.5f}_std{:.5f}_meanKT{:.5f}_std{:.5f}_meanSR{:.5f}_std{:.5f}'.format(
                                   args.dataname, np.mean(IM_ls), np.std(IM_ls), np.mean(OT_ls), np.std(OT_ls),
                                   np.mean(KT_ls), np.std(KT_ls), np.mean(SR_ls), np.std(SR_ls)))
    open(metric_path, 'a').close()
