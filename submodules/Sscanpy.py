import argparse
import os
import sys

# sys.path.append(r'C:\Users\yingyinyu3\PycharmProjects\STCMI')
# sys.path.append('/mnt/c/Users/yingyinyu3/PycharmProjects/STCMI')
sys.path.append('/home/a/yingyingyu/STCMI')

from utils.data_loader import *
from utils.Metrics import ClusteringMetrics
from utils.Utils import run_leiden
from models.model_utils import generate_CMI_from_adj, get_CMI_connectivities, CMIdf2Adj


class Experiment:
    def __init__(self, args):
        self.args = args

    def load_data(self):
        adata = sc.read_h5ad(self.args.adata_file)
        adata = self.preprocess(adata)
        sc.pp.pca(adata, n_comps=20, random_state=self.args.seed)
        sc.pp.neighbors(adata, n_neighbors=args.k, n_pcs=20, random_state=self.args.seed)
        self.adata = adata

    def get_CMI_connectivities(self, dir, percent=0.2):
        adj = self.adata.obsp["connectivities"]
        if not os.path.exists(dir + "CMI.csv"):
            if not os.path.exists(dir):
                os.makedirs(dir)
            print('Start to generate ', dir, 'CMI ...')
            if not isinstance(self.adata.X, np.ndarray):
                self.adata.X = self.adata.X.toarray()
            df = generate_CMI_from_adj(adj.toarray(), self.adata.X, dir)
        else:
            df = pd.read_csv(dir + "CMI.csv")
        adj = CMIdf2Adj(df, adj, percent)
        return adj

    def preprocess(self, adata):
        if self.args.hvg:
            sc.pp.filter_genes(adata, min_cells=3)
            print('After flitering: ', adata.shape)
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        self.args.norm_target = float(self.args.norm_target) if self.args.norm_target is not None else None
        sc.pp.normalize_total(adata, target_sum=self.args.norm_target)
        sc.pp.log1p(adata)

        if 'highly_variable' in adata.var.keys():
            adata = adata[:, adata.var['highly_variable']].copy()
        if 'cluster' not in adata.obs.keys():
            gt_clusters = pd.Series(index=adata.obs_names)
            for obs_name in adata.obs_names:
                res = (adata.uns['milestone_percentages']['cell_id'] == obs_name)
                milestones = adata.uns['milestone_percentages'].loc[res, 'milestone_id']
                percentages = adata.uns['milestone_percentages'].loc[res, 'percentage']
                cluster_id = milestones.loc[percentages.idxmax()]
                gt_clusters.loc[obs_name] = cluster_id
            adata.obs['cluster'] = gt_clusters
        print('Processed adata:', adata, sep='\n')
        return adata


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default="Zhuang-ABCA-2.005")
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hvg', type=bool, default=False)
    parser.add_argument('--norm_target', type=float, default=None)
    parser.add_argument('--adata_file', type=str, default='../dataset/stdata/')
    parser.add_argument('--CMI_dir', type=str, default='../cmi_result/')
    parser.add_argument('--analysis_dir', type=str, default='../analysis/ablation/p_analysis/')
    parser.add_argument('--output_dir', type=str, default='../result/')
    parser.add_argument('--img_dir', type=str, default='../img/')
    args = parser.parse_args()
    print(args)
    args.CMI_dir = args.CMI_dir + '/' + args.dataname + '/scanpy/' + str(args.k) + '/'
    args.out_path = args.output_dir + args.dataname + '/'
    args.analysis_dir = args.analysis_dir + args.dataname + '_scanpy_'
    args.img_dir = args.img_dir + args.dataname + '/'
    args.adata_file = args.adata_file + args.dataname + '/data.h5ad'
    return args


if __name__ == '__main__':
    args = parse_arguments()
    result_df = []
    for p in np.arange(0.01, 0.305, 0.01):
        ari_ls, ami_ls = [], []
        for seed in [0, 1, 2, 3, 4]:
            args.seed = seed
            exp = Experiment(args)
            adata = exp.load_data()
            adata.obsp['connectivities'] = get_CMI_connectivities(adata, args.CMI_dir, percent=p)
            _, adata = run_leiden(adata, len(set(adata.obs['cluster'])), cluster_key='leiden', random_state=args.seed)
            cm_all = ClusteringMetrics(adata.obs['cluster'].astype('category').values, adata.obs['leiden'])
            ari, ami = cm_all.evaluationClusterModelFromLabel()
            ari_ls.append(ari)
            ami_ls.append(ami)
        print('Percent:{:.2f} Mean ARI: {:.5f}, Mean AMI: {:.5f}'.format(p, np.mean(ari_ls), np.mean(ami_ls)))
        result_df.append([p, np.mean(ari_ls), np.mean(ami_ls)])
    result_df = pd.DataFrame(result_df, columns=['Percent', 'ARI', 'AMI'])
    result_df.to_csv(args.analysis_dir + 'result.csv', index=False)
    # metric_path = os.path.join(args.out_path,
    #                            'scanpy_meanARI{:.5f}_stdARI{:.5f}_meanAMI{:.5f}_stdAMI{:.5f}'.format(
    #                                np.mean(ari_ls), np.std(ari_ls), np.mean(ami_ls), np.std(ami_ls)))
    # open(metric_path, 'a').close()
