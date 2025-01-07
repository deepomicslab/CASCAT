import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

sys.path.append('/home/a/yingyingyu/CASCAT')

from submodules.SpaceFlow_funs import SpaceFlow
from models.model_utils import get_CMI_connectivities
from utils.Metrics import ClusteringMetrics


class Experiment:
    def __init__(self, args):
        self.args = args
        self.sf, self.data = self.load_adata(args.data_path)
        self.nclasses = len(set(self.data.obs["cluster"]))
        self.percent = args.percent

    def load_adata(self, path):
        adata = sc.read_h5ad(path)
        sc.pp.filter_genes(adata, min_cells=3)
        sf = SpaceFlow(adata=adata)
        sf.preprocessing_data(n_top_genes=3000)
        print("Preprocessed data shape:", sf.adata_preprocessed.shape)
        return sf, sf.adata_preprocessed

    def assign_colors(self, node_list, colors="Paired"):
        self.sf.adata_preprocessed.obs["dpt_pseudotime"] = self.sf.pSM_values
        mean_dpt = []
        for node in node_list:
            mean_dpt.append(
                self.data.obs[self.sf.adata_preprocessed.obs["leiden"] == str(node)]["dpt_pseudotime"].median())
        order_nodes = list(np.array(node_list)[np.argsort(mean_dpt)])
        print("Trajectory:", order_nodes)
        ordered_node_map = {node: str(i) for i, node in enumerate(order_nodes)}
        cmap_n = plt.get_cmap(colors).N
        cmap_ = plt.get_cmap(colors)
        color_list = []
        for i, cluster in enumerate(sorted(node_list)):
            color_list.append(matplotlib.colors.to_hex(cmap_(i / (cmap_n - 1))))
        color_list = [color_list[(int(ordered_node_map[node]) + 1) % len(order_nodes)] for
                      node in node_list]
        return color_list

    def run_spaceflow(self):
        best_emb = self.sf.train(spatial_regularization_strength=0.1, z_dim=50, lr=1e-2, epochs=1000,
                                 max_patience=50,
                                 min_stop=100, random_seed=self.args.seed, gpu=0,
                                 regularization_acceleration=True,
                                 edge_subset_sz=1000000)
        self.data.obsm["X_emb"] = best_emb
        x_coord = self.data.obsm['s patial']
        self.data.obsm['spatial'] = x_coord[:, [0, 1]]
        # self.data.obsm['spatial'] = x_coord[:, [1, 0]]
        sc.pp.neighbors(self.data, n_neighbors=self.args.n_neighbors, use_rep='X_emb', random_state=self.args.seed)
        # for fast running
        # sc.pp.neighbors(self.data, n_neighbors=self.args.n_neighbors, random_state=self.args.seed)
        # self.data.obsp['connectivities'] = get_CMI_connectivities(self.data, self.args.CMI_dir, percent=self.percent)
        # resolution, _ = run_leiden(self.data, self.nclasses, random_state=self.args.seed)  # respect raw paper
        resolution = 1.0
        sc.tl.leiden(self.data, resolution=float(resolution))
        self.data.obs["celltype_mapped_refined"] = self.data.obs["leiden"].cat.codes
        self.sf.adata_preprocessed.obs["celltype_mapped_refined"] = self.data.obs["celltype_mapped_refined"]
        self.sf.adata_preprocessed.obs["leiden"] = self.data.obs["leiden"]
        sc.tl.paga(self.sf.adata_preprocessed, groups="leiden")
        cm_all = ClusteringMetrics(self.data.obs['cluster'].astype('category').values, self.data.obs['leiden'])
        ari, ami = cm_all.evaluationClusterModelFromLabel()
        self.sf.pseudo_Spatiotemporal_Map(pSM_values_save_filepath=self.args.out_path + "pSM_values.tsv",
                                          n_neighbors=self.args.n_neighbors, resolution=resolution)
        # colors = self.assign_colors(self.sf.adata_preprocessed.obs["leiden"].unique())
        # print(self.args.img_path + "spaceflow.png")
        # self.sf.plot_pSM(pSM_figure_save_filepath=self.args.img_path + "spaceflow.png", scatter_sz=self.args.s,
        #                  colormap=colors)
        return ari, ami


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Zhuang-ABCA-2.053')
    parser.add_argument('--data_dir', type=str, default='../dataset/stdata/')
    parser.add_argument('--img_dir', type=str, default='../img/')
    parser.add_argument('--out_dir', type=str, default='../result/')
    parser.add_argument('--clu_dir', type=str, default='../clu_result/')
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--CMI_dir', type=str, default='../cmi_result/')
    parser.add_argument('--analysis_dir', type=str, default='../analysis/p_analysis/')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--percent', type=float, default=0.01)
    parser.add_argument('--s', type=int, default=40, help='size of the scatter plot')
    args = parser.parse_args()
    args.data_path = args.data_dir + args.dataname + '/data.h5ad'
    args.img_path = args.img_dir + args.dataname + '/'
    args.out_path = args.out_dir + args.dataname + '/'
    args.clu_dir = args.clu_dir + args.dataname + '/'
    args.CMI_dir = args.CMI_dir + '/' + args.dataname + '/spaceflow/' + str(args.n_neighbors) + '/'
    args.analysis_dir = args.analysis_dir + args.dataname + '_spaceflow_'
    return args


if __name__ == '__main__':
    from time import time

    start = time()
    args = parse_arguments()
    exp = Experiment(args)
    ari, ami = exp.run_spaceflow()
    print(f'SpaceFlow time: {time() - start}')
    # print('ARI: {:.5f}, AMI: {:.5f}'.format(ari, ami))
    # result_df = []
    # for p in np.arange(0.01, 0.305, 0.01):
    #     ari_ls, ami_ls = [], []
    #     for seed in range(5):
    #         args.seed = seed
    #         args.percent = p
    #         exp = Experiment(args)
    #         ari, ami = exp.run_spaceflow()
    #         print('ARI: {:.5f}, AMI: {:.5f}'.format(ari, ami))
    #         ari_ls.append(ari)
    #         ami_ls.append(ami)
    #     print('Percent:{:.2f} Mean ARI: {:.5f}, Mean AMI: {:.5f}'.format(p, np.mean(ari_ls), np.mean(ami_ls)))
    #     result_df.append([p, np.mean(ari_ls), np.mean(ami_ls)])
    # result_df = pd.DataFrame(result_df, columns=['Percent', 'ARI', 'AMI'])
    # result_df.to_csv(args.analysis_dir + 'result.csv', index=False)
    # metric_path = os.path.join(args.out_path,
    #                            'spaceflow_meanARI{:.5f}_stdARI{:.5f}_meanAMI{:.5f}_stdAMI{:.5f}'.format(
    #                                np.mean(ari_ls), np.std(ari_ls), np.mean(ami_ls), np.std(ami_ls)))
    # open(metric_path, 'a').close()
