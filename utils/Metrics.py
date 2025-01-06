import warnings
from ot.gromov import gromov_wasserstein
import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from sklearn import metrics


class ClusteringMetrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def evaluationClusterModelFromLabel(self):
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        ami = metrics.adjusted_mutual_info_score(self.true_label, self.pred_label)
        return ari, ami


def compute_gt_milestone_network(ad, uns_mn_key="milestone_network", mode="directed"):
    # NOTE: Since the dyntoy tool does not provide the spatial position
    # of the milestones, this function uses spring_layout to plot the
    # node positions. Hence the displayed graph is not guaranteed to produce
    # accurate spatial embeddings of milestones in the 2d plane.
    assert mode in ["directed", "undirected"]
    if uns_mn_key not in ad.uns_keys():
        raise Exception(f"Milestone network not found in uns.{uns_mn_key}")
    # Get a set of unique milestones
    mn = ad.uns[uns_mn_key]
    from_milestones = set(mn["from"].unique())
    to_milestones = set(mn["to"].unique())
    milestones = from_milestones.union(to_milestones)
    # Construct the milestone network
    milestone_network = nx.DiGraph() if mode == "directed" else nx.Graph()
    for milestone in milestones:
        milestone_network.add_node(milestone)
    for idx, (f, t) in enumerate(zip(mn["from"], mn["to"])):
        milestone_network.add_edge(f, t, weight=mn["length"][idx])
    return milestone_network


def compute_ranking_correlation(pseudotime1, pseudotime2):
    """Computes the ranking correlation between two pseudotime series.
    It is upto the user to ensure that the index of the two pseudotime series match

    Args:
        pseudotime1 ([np.ndarray, pd.Series]): Pseudotime series 1
        pseudotime2 ([np.ndarray, pd.Series]): Pseudotime series 2

    Returns:
        [dict]: A dictionary containing KT, Weighted KT and SR correlations.
    """
    import scipy.stats as ss
    kt = ss.kendalltau(pseudotime1, pseudotime2)
    weighted_kt = ss.weightedtau(pseudotime1, pseudotime2)
    sr = ss.spearmanr(pseudotime1, pseudotime2)
    return {"kendall": kt, "weighted_kendall": weighted_kt, "spearman": sr}


class BaseDistance:
    def __init__(self):
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        dist = -1  # compute the distance
        self.results["dist"] = dist  # store dist in self.results
        return dist  # return only one value!


class IpsenMikhailov(BaseDistance):
    """Compares the spectrum of the Laplacian matrices."""

    def __init__(self):
        super().__init__()
        seed = 0
        np.random.seed(seed)

    def dist(self, G1, G2, hwhm=0.08):
        # get the adjacency matrices
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)
        self.results["adjacency_matrices"] = adj1, adj2
        # get the IM distance
        dist = self._im_distance(adj1, adj2, hwhm)
        self.results["dist"] = dist
        return dist

    def _im_distance(self, adj1, adj2, hwhm):
        """Computes the Ipsen-Mikhailov distance for two symmetric adjacency matrices
        Base on this paper :
        https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.046109
        Note : this is also used by the file hamming_ipsen_mikhailov.py
        Parameters
        ----------
        adj1, adj2 (array): adjacency matrices.
        hwhm (float) : hwhm of the lorentzian distribution.
        Returns
        -------
        dist (float) : Ipsen-Mikhailov distance.
        """
        N_1 = len(adj1)
        N_2 = len(adj2)
        # get laplacian matrix
        L1 = laplacian(adj1, normed=False)
        L2 = laplacian(adj2, normed=False)
        w1 = np.sqrt(np.abs(eigh(L1)[0][1:]))
        w2 = np.sqrt(np.abs(eigh(L2)[0][1:]))

        norm1 = (N_1 - 1) * np.pi / 2 - np.sum(np.arctan(-w1 / hwhm))
        norm2 = (N_2 - 1) * np.pi / 2 - np.sum(np.arctan(-w2 / hwhm))
        density1 = lambda w: np.sum(hwhm / ((w - w1) ** 2 + hwhm ** 2)) / norm1
        density2 = lambda w: np.sum(hwhm / ((w - w2) ** 2 + hwhm ** 2)) / norm2
        func = lambda w: (density1(w) - density2(w)) ** 2
        return np.sqrt(quad(func, 0, np.inf, limit=100)[0])


class OTDistance:
    def __init__(self):
        np.random.seed(0)

    def dist(self, G1, G2):
        adj1 = nx.to_numpy_array(G1)
        adj1 = np.where(adj1 > 0, 1.0, 0)
        adj2 = nx.to_numpy_array(G2)
        h1 = np.ones(adj1.shape[0]) / adj1.shape[0]
        h2 = np.ones(adj2.shape[0]) / adj2.shape[0]
        OT, log = gromov_wasserstein(adj1, adj2, h1, h2, 'square_loss', symmetric=True, log=True,
                                     verbose=False, epsilon=5e-4, max_iter=1000)
        dist = log['gw_dist']
        return dist


def caculate_metric(adata, psedo_key="metric_pseudotime", adj_key="metric_connectivities"):
    if adj_key == "paga":
        adj = adata.uns["paga"]["connectivities"].toarray().copy()
    else:
        adj = adata.uns[adj_key].copy()
    if not np.allclose(adj, adj.T):
        adj = adj + adj.T

    dist_obj = IpsenMikhailov()
    net1 = compute_gt_milestone_network(adata, mode="undirected")
    net2 = nx.from_scipy_sparse_array(csr_matrix(adj))
    dist1 = dist_obj.dist(net1, net2)
    IM = max(round(dist1, 5), 0)
    dist_obj2 = OTDistance()
    dist2 = dist_obj2.dist(net1, net2)
    OT = max(round(dist2, 5), 0)
    if isinstance(adata.uns["timecourse"], np.ndarray):
        gt_pseudotime = pd.DataFrame(adata.uns["timecourse"], index=adata.obs_names)
    else:
        gt_pseudotime = adata.uns["timecourse"].reindex(adata.obs_names)
    res = compute_ranking_correlation(gt_pseudotime, adata.obs[psedo_key])
    KT, SR = round(res["kendall"][0], 5), round(res["spearman"][0], 5)
    return IM, OT, KT, SR


def caculate_R_metric(adata):
    adj = adata.uns["metric_connectivities"].copy()
    if not np.allclose(adj, adj.T):
        warnings.warn("The adjacency matrix is not symmetric. "
                      "It is recommended to use an undirected adjacency matrix.")

    dist_obj = IpsenMikhailov()
    net1 = compute_gt_milestone_network(adata, mode="undirected")
    net2 = nx.from_scipy_sparse_array(csr_matrix(adj))
    dist1 = dist_obj.dist(net1, net2)
    IM = max(round(dist1, 5), 0)
    dist_obj2 = OTDistance()
    dist2 = dist_obj2.dist(net1, net2)
    OT = max(round(dist2, 5), 0)
    gt_pseudotime = adata.uns["timecourse"]
    res = compute_ranking_correlation(gt_pseudotime, adata.obs.values)
    KT, SR = round(res["kendall"][0], 5), round(res["spearman"][0], 5)
    return IM, OT, KT, SR


def caculate_R_cluster_metric(predict_labels, adata):
    # predict_labels = adata.obs["metric_label"].copy()
    gt_labels = adata.obs["cluster"]
    cm = ClusteringMetrics(gt_labels, predict_labels)
    ari, ami = cm.evaluationClusterModelFromLabel()
    return ari, ami
