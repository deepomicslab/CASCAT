import random
import argparse
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/a/yingyingyu/CASCAT')

from utils.Utils import *
from utils.Metrics import caculate_metric, ClusteringMetrics


class Experiment:
    def __init__(self, args):
        self.args = args
        self.seed_everything(args.seed)
        self.n_neighbors = args.n_neighbors
        self.adata = self.prepare_data(args.adata_file)
        self.labels = self.adata.obs['cluster']
        self.nclasses = len(set(self.labels))
        self.predict_key = 'louvain'
        self.pesudotime_key = 'dpt_pseudotime'

    def seed_everything(self, seed):
        np.random.seed(seed)
        sc.settings.verbosity = 3
        random.seed(seed)

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
        return adata

    def cluster_analysis(self):
        gt_clusters = pd.DataFrame(self.labels, index=self.adata.obs_names)
        metric_clusters = pd.DataFrame(self.adata.obs[self.predict_key].values, index=self.adata.obs_names)
        cluster_df = pd.concat([gt_clusters, metric_clusters], axis=1)
        cluster_df.columns = ['gt_clusters', self.predict_key]
        cluster_df = cluster_df.groupby(self.predict_key)['gt_clusters'].value_counts(normalize=True)
        cluster_df = cluster_df[cluster_df > 0.1]
        print(cluster_df)
        cluster_map, idx, idx_list, pause = {}, 0, list(set([i[0] for i in cluster_df.index])), 0
        for idx_ in idx_list.copy():
            gt_cluster = cluster_df[idx_]
            if (len(gt_cluster)) == 1:
                cluster_map[idx_] = gt_cluster.index[0]
                cluster_df = cluster_df.drop((idx_, gt_cluster.index[0]))
                idx_list.remove(idx_)
        while len(idx_list) > 0:
            gt_cluster = cluster_df[idx_list[idx]]
            candidates = [i for i in gt_cluster.index if i not in cluster_map.values()]
            if len(candidates) == 0:
                cluster_map[idx_list[idx]] = gt_cluster.index[0]
                cluster_df = cluster_df.drop((idx_list[idx], gt_cluster.index[0]))
                idx_list.remove(idx_list[idx])
                pause = 0

            elif len(candidates) == 1:
                cluster_map[idx_list[idx]] = candidates[0]
                cluster_df = cluster_df.drop((idx_list[idx], candidates[0]))
                idx_list.remove(idx_list[idx])
                pause = 0
            else:
                pause += 1
                if pause > 1:
                    cluster_map[idx_list[idx]] = candidates[0]
                    cluster_df = cluster_df.drop((idx_list[idx], candidates[0]))
                    idx_list.remove(idx_list[idx])
                    pause = 0
            if len(idx_list) > 0:
                idx = (idx + 1) % len(idx_list)
        print(cluster_map)
        return cluster_map

    def caculate_centriods(self, embedding):
        embedding_dict = {}
        for embedding, label in zip(embedding, self.adata.obs[self.predict_key + "_map"]):
            if label in embedding_dict:
                embedding_dict[label].append(embedding)
            else:
                embedding_dict[label] = [embedding]
        embedding_arrays = {label: np.array(embedding_list) for label, embedding_list in embedding_dict.items()}
        centroids = {label: np.mean(embedding_array, axis=0) for label, embedding_array in embedding_arrays.items()}
        centroids_ = np.array([centroids[label] for label in centroids.keys()])
        return centroids_

    def run_paga(self):
        sc.tl.pca(self.adata, svd_solver='arpack', random_state=self.args.seed)
        sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, random_state=self.args.seed, n_pcs=20)
        sc.tl.diffmap(self.adata, random_state=self.args.seed)
        sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, use_rep='X_diffmap', random_state=self.args.seed)
        _, self.adata = run_louvain(self.adata, self.nclasses, cluster_key=self.predict_key,
                                    random_state=self.args.seed)
        self.adata.obs[self.predict_key] = self.adata.obs[self.predict_key].astype('category')
        cm_all = ClusteringMetrics(self.labels, self.adata.obs[self.predict_key])
        ari, ami = cm_all.evaluationClusterModelFromLabel()
        if 'start_id' in self.adata.uns:
            self.adata.uns['iroot'] = np.where(self.adata.obs_names == self.adata.uns['start_id'])[0][0]
        sc.tl.dpt(self.adata)
        print(f'ARI: {ari}, AMI: {ami}')
        sc.tl.paga(self.adata, groups=self.predict_key)

    def plot_paga(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sc.tl.draw_graph(self.adata, random_state=self.args.seed)
        clustermap = self.cluster_analysis()
        self.adata.obs[self.predict_key + '_map'] = self.adata.obs[self.predict_key].map(clustermap)
        sc.pl.draw_graph(self.adata, color=self.pesudotime_key, show=False, ax=ax, colorbar_loc=None, title=None,
                         cmap='plasma')
        centriods = self.caculate_centriods(self.adata.obsm['X_draw_graph_fr'])
        sc.tl.paga(self.adata, groups=self.predict_key + '_map')
        sc.pl.paga(self.adata, color=self.pesudotime_key, pos=centriods, node_size_scale=0.5, edge_width_scale=1,
                   ax=ax, show=False, arrowsize=10, colorbar=False, fontsize=22, cmap='plasma',
                   random_state=self.args.seed)
        # cbar = plt.colorbar(ax.collections[0], ax=ax, fraction=0.03, pad=0.01)
        ax.set_title('', fontsize=16)
        ax.axis('off')
        plt.savefig(os.path.join(self.args.img_path, '{}_paga.png').format(self.args.dataname))
        # plt.show()

    def caculate_metrics(self):
        IM, OT, KT, SR = caculate_metric(self.adata, psedo_key='dpt_pseudotime', adj_key='paga')
        print(f'IM: {IM}, OT:{OT},KT: {KT}, SR: {SR}')
        return IM, OT, KT, SR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default="real1")
    parser.add_argument('--adata_file', type=str, default='../dataset/scdata/')
    parser.add_argument('--img_path', type=str, default='../img/scimg/')
    parser.add_argument('--out_path', type=str, default='../result/')
    parser.add_argument('--predict_key', type=str, default='louvain')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_neighbors', type=int, default=30)
    return parser.parse_args()


if __name__ == '__main__':
    from time import time

    start = time()
    args = parse_arguments()
    args.adata_file = args.adata_file + args.dataname + '/data.h5ad'
    args.img_path = args.img_path
    IM_ls, OT_ls, KT_ls, SR_ls = [], [], [], []
    for seed in range(5):
        args.seed = seed
        exp = Experiment(args)
        exp.run_paga()
        exp.plot_paga()
        print(f'Paga time: {time() - start}')
        IM, OT, KT, SR = exp.caculate_metrics()
        IM_ls.append(IM)
        OT_ls.append(OT)
        KT_ls.append(KT)
        SR_ls.append(SR)
    print(f'IM: {np.mean(IM_ls)},OT:{np.mean(OT_ls)}, KT: {np.mean(KT_ls)}, SR: {np.mean(SR_ls)}')
    metric_path = os.path.join(args.out_path,
                               '{}_paga_IM{:.5f}_std{:.5f}_meanOT{:.5f}_std{:.5f}_meanKT{:.5f}_std{:.5f}_meanSR{:.5f}_std{:.5f}'.format(
                                   args.dataname, np.mean(IM_ls), np.std(IM_ls), np.mean(OT_ls), np.std(OT_ls),
                                   np.mean(KT_ls), np.std(KT_ls), np.mean(SR_ls), np.std(SR_ls)))
    open(metric_path, 'a').close()
