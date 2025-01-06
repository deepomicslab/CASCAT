import argparse
import random
from utils.data_loader import *
from models.BinaryCMITree import BinaryCMITree
import networkx as nx
from models.model_utils import run_group_frac, connect_graph, CMIPlot, run_pseudotime, run_diffusion_maps, \
    determine_multiscale_space
from scipy.sparse.csgraph import dijkstra
from utils.Plot import plot_ground_truth
from utils.Metrics import caculate_metric


class InferExperiment:
    def __init__(self, args, adata=None):
        self.setup_seed(args.seed)
        self.seed = args.seed
        self.k = args.a_k
        self.ncomp = args.ncomp
        (self.img_path, self.predict_key, self.emb_key, self.cluster_key, self.pesudo_key, self.adata,
         self.save_path) = self.load_params(args, adata)
        print(f"Finish load Infer params")
        if 'threshold' not in args:
            args.threshold = 0.05
        self.tree = BinaryCMITree(self.adata, str(self.adata.uns["root"]), self.predict_key, args.threshold, args.seed,
                                  save_dir=self.save_path, kde=True)
        print(f"Finish initialize tree params")

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def get_root(self, adata, root, group_frac, predict_key):
        root = group_frac.columns[group_frac.loc[root, :] == group_frac.loc[root, :].max()][0]
        if 'spatial' in adata.obsm.keys():
            sub_adata_loc = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names)
            sub_adata_loc['label'] = adata.obs[predict_key]
            root_loc = sub_adata_loc[sub_adata_loc['label'] == root].iloc[:, :2].mean().values
            dist = np.linalg.norm(adata.obsm['spatial'] - root_loc, axis=1)
            idx = np.argmin(dist)
        else:
            sub_root = adata.obs[predict_key] == root
            dist = np.linalg.norm(adata.obsm['X_pca'] - adata.obsm['X_pca'][sub_root].mean(), axis=1)
            idx = np.argmin(dist)
        return root, idx

    def load_params(self, args, adata=None):
        args.img_dir = args.img_dir if 'img_dir' in args else './img/scimg/'
        img_path = args.img_dir + args.data_name + '_'
        predict_key, emb_key = 'cascat_clusters', 'cascat_embedding'
        cluster_key, pesudo_key = 'cascat_connectivities', "cascat_pseudotime"
        dist_key = "knn_adj"
        print(f"using keys are {predict_key}, {emb_key}, {cluster_key}, {pesudo_key}, {dist_key}")
        if adata is None:
            adata, labels, nclasses = load_adata_from_raw(args)
        if 'emb_path' in args and args.emb_path != 'None':
            if 'cluster' in adata.obs.keys():
                nclasses = len(set(adata.obs['cluster'].values))
            else:
                nclasses = args.nclass
            adata = load_labels_from_emb(args, nclasses, adata, label_key=predict_key,
                                         emb_key=emb_key)
        if "cluster" in adata.obs.keys() and predict_key in adata.obs.keys():
            group_frac = run_group_frac(adata.obs[predict_key], adata.obs["cluster"])
        elif predict_key in adata.obs.keys():
            group_frac = run_group_frac(adata.obs[predict_key], adata.obs[predict_key])
        elif "cluster" in adata.obs.keys():
            predict_key = "cluster"
            group_frac = run_group_frac(adata.obs["cluster"], adata.obs["cluster"])
        else:
            raise ValueError("No cluster label in adata")

        if 'root' not in args:
            raise ValueError("No root label in args")
        root = group_frac.columns[group_frac.loc[args.root, :] == group_frac.loc[args.root, :].max()][0]
        if "start_id" not in adata.uns:
            print("No start cell id, use truth label in cluster to find start cell")
            _, start_cell_idx = self.get_root(adata, args.root, group_frac, predict_key)
        else:
            start_cell_idx = np.where(adata.obs_names.values == adata.uns['start_id'])[0][0]
        adata.uns['start_cell_idx'] = start_cell_idx
        adata.uns['root'] = root
        adata.uns['group_frac'] = group_frac
        print(f"Root is {root}, Start cell idx is {start_cell_idx}")
        return img_path, predict_key, emb_key, cluster_key, pesudo_key, adata, args.job_dir

    def run_compute_pseudotime(self, ncomp=10):
        run_diffusion_maps(self.adata, pca_key=self.emb_key, n_components=ncomp, knn=self.k, seed=self.seed)
        determine_multiscale_space(self.adata)
        pseudotime = run_pseudotime(self.adata, self.adata.uns['start_id'],
                                    num_waypoints=min(1200, len(self.adata.obs_names)),
                                    seed=self.seed, knn=self.k)
        self.adata.obs[self.pesudo_key] = pseudotime

    def compute_pseudotime(self, is_undirected=True):
        emb = self.adata.obsm[self.emb_key]
        adjacency = np.zeros(shape=(self.adata.n_obs, self.adata.n_obs), dtype=np.float64)
        adj_dist = distance_matrix(emb, emb)
        neighbors_idx = np.argsort(adj_dist, axis=1)
        for i, n_idx in enumerate(neighbors_idx):
            n_idx = n_idx[n_idx != i][:args.a_k]
            adjacency[i, n_idx] = 1
            assert adjacency[i, i] != 1
        if is_undirected:
            print('Use undirected cell-cell communication graph')
            adjacency = ((adjacency + adjacency.T) > 0).astype(int)
        adj_dist = adjacency.copy()
        start_cell_idx = self.adata.uns['start_cell_idx']
        communities = self.adata.obs[self.predict_key]
        embedding = pd.DataFrame(self.adata.obsm[self.emb_key], index=self.adata.obs_names)
        clusters = {}
        for idx in np.unique(communities):
            cluster_idx = communities == idx
            clusters[idx] = cluster_idx
        adj_dist = pd.DataFrame(adj_dist, index=self.adata.obs_names, columns=self.adata.obs_names,
                                dtype=np.float64)
        p = dijkstra(adj_dist.to_numpy(), indices=[start_cell_idx], min_only=True)
        pseudotime = pd.Series(p, index=self.adata.obs_names)
        for _, cluster in clusters.items():
            p_cluster = pseudotime.loc[cluster]
            cluster_start_cell = p_cluster.idxmin()
            adj_sc = adj_dist.loc[cluster, cluster]
            adj_sc = connect_graph(adj_sc, embedding.loc[cluster, :],
                                   np.where(adj_sc.index == cluster_start_cell)[0][0])
            adj_dist.loc[cluster, cluster] = adj_sc
        p = dijkstra(adj_dist, indices=[start_cell_idx], min_only=True)
        pseudotime = pd.Series(p, index=self.adata.obs_names)
        print(f"Number of inf in pseudotime is {np.sum(pseudotime == np.inf)}")
        pseudotime = pseudotime / pseudotime[pseudotime != np.inf].max()
        pseudotime[pseudotime == np.inf] = 1
        self.adata.obs[self.pesudo_key] = pseudotime

    def infer(self, cluster_list=None, debug=False, debug_nodes=None):
        root = str(self.adata.uns['root'])
        self.tree.init_tree(cluster_list, debug=debug, debug_nodes=debug_nodes)
        self.tree.construct_tree(root)
        adj_unweight = nx.to_numpy_array(self.tree.tree, nodelist=sorted(self.tree.tree.nodes))
        cluster_ids = [node for node in sorted(self.tree.tree.nodes)]
        self.adata.uns[self.cluster_key] = pd.DataFrame(adj_unweight, index=cluster_ids, columns=cluster_ids)
        self.run_compute_pseudotime(self.ncomp)
        # self.compute_pseudotime()
        if "timecourse" in self.adata.uns and "milestone_network" in self.adata.uns:
            IM, OT, KT, SR = caculate_metric(self.adata, self.pesudo_key, self.cluster_key)
            print(f"IM:{IM}, OT:{OT}, KT:{KT}, SR:{SR}")
            return IM, OT, KT, SR
        else:
            print("No ground truth for evaluation")
            return None, None, None, None

    def plot(self, type: str = "st_emb", sorted_genes=None, marker_genes=None, order_layer=None, show=True,
             colors='Paired', show_traj=False):
        obj = CMIPlot(self.adata, save_path=self.img_path, start_cell_idx=self.adata.uns['start_cell_idx'],
                      root=self.adata.uns['root'], group_frac=self.adata.uns['group_frac'],
                      pesudo_key=self.pesudo_key, connect_key=self.cluster_key, predict_key=self.predict_key,
                      emb_key=self.emb_key)
        if type == "emb":
            obj.plot_embedding(show=show, colors=colors)
        elif type == "st_emb":
            obj.plot_st_embedding(show_trajectory=show_traj, colors=colors)
        elif type == "pesodutime":
            obj.plot_pseudotime(show=show)
        elif type == "st_pesodutime":
            obj.plot_st_pseudotime()
        elif type == "tree_mode":
            obj.plot_trajectory_tree(show=show)
        elif type == "subtype":
            obj.plot_subtype(show=show)
        elif type == "marker_heatmap":
            obj.plot_marker_heatmap(sorted_genes, order_layer, show=show)
        elif type == "marker_gene":
            obj.plot_marker_gene(marker_genes, order_layer, show=show)
        elif type == "ground_truth":
            plot_ground_truth(self.adata)
        else:
            print(
                "Select plot type from st_emb, st_pesodutime, subtype, marker_heatmap, marker_gene, tree_mode, emb, pesodutime, ground_truth")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_name", type=str, default="HER2ST")
    args.add_argument("--adata_file", type=str, default="./dataset/stdata/HER2ST/data.h5ad")
    args.add_argument("--root", type=str, default="Tumor region")
    args.add_argument("--job_dir", type=str, default="./result/causalLearn/")
    args = args.parse_args()
    exp = InferExperiment(args)
    exp.infer()
