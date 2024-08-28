import stlearn as st
import argparse
import os
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib

import sys

sys.path.append('/mnt/c/Users/yingyinyu3/PycharmProjects/STCMI')

from utils.Metrics import ClusteringMetrics
from utils.Utils import *

rcParams['font.family'] = 'Times New Roman'
st.settings.set_figure_params(dpi=300)
from models.model_utils import get_CMI_connectivities


class Experiment:
    def __init__(self, args):
        self.args = args
        self.data = self.load_data(args.data_path)
        self.nclasses = len(set(self.data.obs["cluster"]))

    def load_data(self, path):
        adata = sc.read_h5ad(path)
        # spatial_matrix = pd.DataFrame(adata.obsm['spatial'][:, [1, 0]], columns=["imagecol", "imagerow"])
        spatial_matrix = pd.DataFrame(adata.obsm['spatial'], columns=["imagecol", "imagerow"])
        max_coor = np.max(adata.obsm["spatial"])
        scale = 2000 / max_coor
        adata.obs["imagecol"] = spatial_matrix["imagecol"].values * scale
        adata.obs["imagerow"] = spatial_matrix["imagerow"].values * scale
        adata.layers["raw_count"] = adata.X
        if self.args.hvg:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        st.pp.normalize_total(adata)
        st.pp.log1p(adata)
        if 'highly_variable' in adata.var.keys():
            adata = adata[:, adata.var['highly_variable']].copy()
        print('Processed adata:', adata, sep='\n')
        sc.tl.pca(adata, random_state=self.args.seed)
        st.pp.neighbors(adata, n_neighbors=self.args.n_neighbors, use_rep='X_pca', random_state=self.args.seed,
                        n_pcs=20)
        adata.obsp['connectivities'] = get_CMI_connectivities(adata, args.CMI_dir, percent=args.p)
        return adata

    def run_group_frac(self):
        predict_labels = self.data.obs['louvain'].astype(int)
        labels = self.data.obs['cluster']
        n_groups = len(set(predict_labels))
        n_truegroups = len(set(labels))
        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=list(set(labels)))
        set_labels = sorted(list(set(predict_labels)))
        for group_i in set_labels:
            loc_i = np.where(predict_labels == group_i)[0]
            true_label_in_group_i = list(np.asarray(labels)[loc_i])
            ll_temp = list(set(true_label_in_group_i))
            for ii in ll_temp:
                group_frac[ii][group_i] = true_label_in_group_i.count(ii)
        print(group_frac)
        return group_frac

    def plot_pesudotime(self, adata, cell_alpha=0.7, spot_size=15,
                        pseudotime_key="dpt_pseudotime", ax=None, margin=0, show_plot=True, dpi=300, fname=None):
        imagecol = adata.obs["imagecol"]
        imagerow = adata.obs["imagerow"]
        tmp = adata.obs
        fig, a = plt.subplots()
        if ax != None:
            a = ax
        dpt = adata.obs[pseudotime_key]
        vmin, vmax = min(dpt), max(dpt)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scale = scaler.fit_transform(tmp[pseudotime_key].values.reshape(-1, 1)).reshape(-1, 1)
        a.scatter(tmp["imagecol"], tmp["imagerow"], edgecolor="none", alpha=cell_alpha, s=spot_size,
                  marker="o", vmin=vmin, vmax=vmax, cmap=plt.get_cmap("viridis"), c=scale.reshape(1, -1)[0])
        a.axis("off")
        a.set_xlim(imagecol.min() - margin, imagecol.max() + margin)
        a.set_ylim(imagerow.min() - margin, imagerow.max() + margin)

        a.set_ylim(a.get_ylim()[::-1])
        cax = fig.add_axes([0.3, 0.1, 0.3, 0.02])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation="horizontal")
        if fname is not None:
            fig.savefig(fname, dpi=dpi, bbox_inches="tight", pad_inches=0)
        if show_plot == True:
            plt.show()

    def assign_color_labels(self, node_list):
        mean_dpt = []
        for node in node_list:
            mean_dpt.append(
                self.data.obs[self.data.obs['louvain'] == str(node)]["dpt_pseudotime"].median())
        order_nodes = list(np.array(node_list)[np.argsort(mean_dpt)])
        ordered_node_map = {node: str(i) for i, node in enumerate(order_nodes)}
        cmap_n = plt.get_cmap("Paired").N
        cmap_ = plt.get_cmap("Paired")
        self.data.uns['louvain' + "_colors"] = []
        for i, cluster in enumerate(sorted(self.data.obs['louvain'].unique())):
            self.data.uns['louvain' + "_colors"].append(
                matplotlib.colors.to_hex(cmap_(i / (cmap_n - 1))))
        self.data.uns['louvain' + "_colors"] = [
            self.data.uns['louvain' + "_colors"][(int(ordered_node_map[node]) + 1) % len(order_nodes)] for
            node in node_list]

    def plot_cluster(self):
        st.pl.cluster_plot(self.data, use_label='louvain', show_trajectories=True,
                           list_clusters=list(self.data.obs['louvain'].unique()),
                           show_subcluster=True, dpi=300, fname=self.args.img_path + "stlearn.png",
                           trajectory_edge_color="black", margin=1, size=self.args.node_size,
                           trajectory_arrowsize=self.args.trajectory_arrowsize,
                           trajectory_width=self.args.trajectory_width, cmap="Paired")

    def run_pseudotime(self):
        group_frac = self.run_group_frac()
        if type(group_frac.columns[0]) == np.int64:
            self.args.root = int(self.args.root)
        root = group_frac[group_frac[self.args.root] == group_frac[self.args.root].max()].index[0]

        self.data.uns["iroot"] = st.spatial.trajectory.set_root(self.data, use_label='louvain',
                                                                cluster=root)
        # The maximum distance between two samples for one to be considered as in the neighborhood
        st.spatial.trajectory.pseudotime(self.data, eps=1500, use_rep="X_pca", use_label='louvain')
        # there are bug for H_sub
        st.spatial.trajectory.pseudotimespace_global(self.data, use_label='louvain',
                                                     list_clusters=self.data.obs['louvain'].unique())
        self.assign_color_labels(list(self.data.obs['louvain'].unique()))
        # self.plot_cluster()

    def run_stlearn(self):
        resolution, _ = run_louvain(self.data, self.nclasses, cluster_key='louvain',
                                    random_state=self.args.seed)
        self.data.obs['louvain'] = self.data.obs['louvain'].astype('str').astype('category')
        cm_all = ClusteringMetrics(self.data.obs["cluster"], self.data.obs['louvain'])
        ari, ami = cm_all.evaluationClusterModelFromLabel()
        print(f'ARI: {ari}, AMI: {ami}')
        return ari, ami


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default="OSCC8")
    parser.add_argument('--data_dir', type=str, default='../dataset/stdata/')
    parser.add_argument('--img_dir', type=str, default='../img/')
    parser.add_argument('--out_dir', type=str, default='../result/')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--root', type=str, default='L1')
    parser.add_argument('--node_size', type=int, default=10)
    parser.add_argument('--trajectory_arrowsize', type=int, default=1)
    parser.add_argument('--trajectory_width', type=int, default=2)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--p', type=float, default=0.05)
    parser.add_argument('--CMI_dir', type=str, default='../cmi_result/')
    parser.add_argument('--analysis_dir', type=str, default='../analysis/ablation/p_analysis/')
    parser.add_argument('--hvg', type=bool, default=True)
    args = parser.parse_args()
    args.data_path = args.data_dir + args.dataname + '/data.h5ad'
    args.img_path = args.img_dir + args.dataname + '/'
    args.out_path = args.out_dir + args.dataname + '/'
    args.CMI_dir = args.CMI_dir + '/' + args.dataname + '/stlearn/' + str(args.n_neighbors) + '/'
    args.analysis_dir = args.analysis_dir + args.dataname + '_stlearn_'
    return args


if __name__ == '__main__':
    args = parse_arguments()
    result_df = []
    for p in np.arange(0.01, 0.305, 0.01):
        ari_ls, ami_ls = [], []
        for seed in range(5):
            args.seed = seed
            args.p = p
            exp = Experiment(args)
            ari, ami = exp.run_stlearn()
            # exp.run_pseudotime()
            ari_ls.append(ari)
            ami_ls.append(ami)
        print('Percent:{:.2f} Mean ARI: {:.5f}, Mean AMI: {:.5f}'.format(p, np.mean(ari_ls), np.mean(ami_ls)))
        result_df.append([p, np.mean(ari_ls), np.mean(ami_ls)])
    result_df = pd.DataFrame(result_df, columns=['Percent', 'ARI', 'AMI'])
    result_df.to_csv(args.analysis_dir + 'result.csv', index=False)
    # metric_path = os.path.join(args.out_path,
    #                            'stlearn_meanARI{:.5f}_stdARI{:.5f}_meanAMI{:.5f}_stdAMI{:.5f}'.format(
    #                                np.mean(ari_ls), np.std(ari_ls), np.mean(ami_ls), np.std(ami_ls)))
    # open(metric_path, 'a').close()
