import palantir
import argparse
import matplotlib.pyplot as plt
import random
import sys

# sys.path.append(r'C:\Users\yingyinyu3\PycharmProjects\STCMI')
# sys.path.append('/mnt/c/Users/yingyinyu3/PycharmProjects/STCMI')
sys.path.append('/home/a/yingyingyu/STCMI')
from utils.Utils import *
from utils.Metrics import caculate_metric, ClusteringMetrics
import os


class Experiment:
    def __init__(self, args):
        self.args = args
        self.num_dim = args.num_dim
        self.num_dim2 = args.num_dim2
        self.k = args.k
        self.predict_key = 'louvain'
        self.adata = self.load_data(args.data_path)
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

    def plot_trajectory(self, pr_res):
        def scatter_with_colorbar(ax, x, y, c, **kwargs):
            sc = ax.scatter(x, y, c=c, **kwargs)
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(sc, cax=cax, orientation="vertical")

        fig, ax = plt.subplots(figsize=(10, 10))
        embedding_data = pd.DataFrame(self.adata.obsm["X_umap"], index=self.adata.obs_names)
        scatter_with_colorbar(
            ax,
            embedding_data.iloc[:, 0],
            embedding_data.iloc[:, 1],
            pr_res.pseudotime[embedding_data.index]
        )
        ax.set_axis_off()
        return fig

    def run_palantir(self):
        sc.pp.normalize_per_cell(self.adata)
        palantir.preprocess.log_transform(self.adata)
        sc.pp.highly_variable_genes(self.adata, n_top_genes=1500, flavor="cell_ranger")
        sc.pp.pca(self.adata, random_state=self.args.seed)
        sc.pp.neighbors(self.adata, n_neighbors=self.k, random_state=self.args.seed, n_pcs=20)
        dm_res = palantir.utils.run_diffusion_maps(self.adata, n_components=15,
                                                   seed=self.args.seed)  # dm_res
        palantir.utils.determine_multiscale_space(self.adata)  # ms data
        self.adata.obsm['X_diffusion'] = dm_res['EigenVectors'].to_numpy()
        sc.pp.neighbors(self.adata, use_rep='X_diffusion', n_neighbors=self.k, random_state=self.args.seed)
        _, adata = run_louvain(self.adata, self.nclasses, cluster_key=self.predict_key,
                               random_state=self.args.seed)
        adata.obs[self.predict_key] = adata.obs[self.predict_key].astype('category')
        sc.tl.paga(adata, groups=self.predict_key)
        sc.tl.umap(adata, random_state=self.args.seed)
        pr_res = palantir.core.run_palantir(
            self.adata, self.start_cell, num_waypoints=1200, seed=self.args.seed)
        fig = self.plot_trajectory(pr_res)
        fig.savefig(self.args.img_path + 'palantir.png')
        # plt.show()

    def caculate_metrics(self):
        cm_all = ClusteringMetrics(self.labels, self.adata.obs[self.predict_key].values)
        ari, nmi = cm_all.evaluationClusterModelFromLabel()
        print('ARI: ', ari, 'NMI: ', nmi)
        IM, KT, SR = caculate_metric(self.adata, psedo_key="palantir_pseudotime", adj_key="paga")
        print("IM: {}, KT: {}, SR: {}".format(IM, KT, SR))
        return IM, KT, SR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default="linear2")
    parser.add_argument('--data_path', type=str, default="./dataset/scdata/")
    parser.add_argument('--img_path', type=str, default='./img/')
    parser.add_argument('--out_path', type=str, default='./result/')
    parser.add_argument('--num_dim', type=int, default=30)
    parser.add_argument('--num_dim2', type=int, default=20)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    args.data_path = args.data_path + args.dataname + '/data.h5ad'
    args.img_path = args.img_path + args.dataname + '/'
    args.out_path = args.out_path + args.dataname + '/'
    IM_ls, KT_ls, SR_ls = [], [], []
    for seed in range(1):
        args.seed = seed
        experiment = Experiment(args)
        experiment.run_palantir()
        IM, KT, SR = experiment.caculate_metrics()
        IM_ls.append(IM)
        KT_ls.append(KT)
        SR_ls.append(SR)
    print("IM: {}, KT: {}, SR: {}".format(np.mean(IM_ls), np.mean(KT_ls), np.mean(SR_ls)))
    # metric_path = os.path.join(args.out_path,
    #                            'palantir_meanIM{:.5f}_stdIM{:.5f}_meanKT{:.5f}_stdKT{:.5f}_meanSR{:.5f}_stdSR{:.5f}'.format(
    #                                np.mean(IM_ls), np.std(IM_ls), np.mean(KT_ls), np.std(KT_ls), np.mean(SR_ls),
    #                                np.std(SR_ls)))
    # open(metric_path, 'a').close()
