import io
import warnings
from typing import List
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import pydot
import scanpy as sc
from castle.algorithms import ICALiNGAM, PC, DirectLiNGAM, GES


def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    adata = sc.read_h5ad('../dataset/stdata/HER2ST/data.h5ad')
    if not isinstance(adata.X, np.ndarray):
        cluster_feature = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names,
                                       dtype=np.float64)
    else:
        cluster_feature = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names, dtype=np.float64)
    if 'cluster' not in adata.obs.keys():
        gt_clusters = pd.Series(index=adata.obs_names)
        for obs_name in adata.obs_names:
            res = (adata.uns['milestone_percentages']['cell_id'] == obs_name)
            milestones = adata.uns['milestone_percentages'].loc[res, 'milestone_id']
            percentages = adata.uns['milestone_percentages'].loc[res, 'percentage']
            cluster_id = milestones.loc[percentages.idxmax()]
            gt_clusters.loc[obs_name] = cluster_id
        adata.obs['cluster'] = gt_clusters
    cluster_feature['cluster'] = adata.obs['cluster']
    cluster_feature = cluster_feature.groupby('cluster').mean().T
    print(cluster_feature.shape)
    return cluster_feature


def draw_nx_graph(G, labels=None, figname: str = 'graph.png'):
    from networkx.drawing.nx_pydot import graphviz_layout
    """Draw nx_graph if skel = False and draw nx_skel otherwise"""
    warnings.filterwarnings("ignore", category=UserWarning)
    pos = graphviz_layout(G, prog="dot")
    plt.rcParams["figure.figsize"] = [12, 20]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    nx.draw_networkx_nodes(G, pos=pos, node_size=500)
    nx.draw_networkx_edges(G, pos=pos, arrows=True, ax=ax, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in G.nodes}, ax=ax, font_size=10)
    plt.axis('off')
    plt.savefig(figname, dpi=300)
    plt.show()


def draw_pydot_graph(G, labels: List[str], figname: str = "graph", dpi=200):
    nodes = list(G.nodes)
    edges = list(G.edges)
    if labels is not None:
        assert len(labels) == len(nodes)

    pydot_g = pydot.Dot("", graph_type="digraph", fontsize=18)
    pydot_g.obj_dict["attributes"]["dpi"] = dpi
    for i, node in enumerate(nodes):
        node_name = labels[i]
        pydot_g.add_node(pydot.Node(i, label=labels[node]))
        pydot_g.add_node(pydot.Node(i, label=node_name))

    for edge in edges:
        node1, node2 = edge[0], edge[1]
        node1_id = nodes.index(node1)
        node2_id = nodes.index(node2)
        dot_edge = pydot.Edge(node1_id, node2_id, dir='left', arrowtail='normal', arrowhead='normal')
        pydot_g.add_edge(dot_edge)
    tmp_png = pydot_g.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(f'../img/causal-learn/{figname}.png', dpi=300)
    plt.show()


def infer(cluster_list=None):
    import argparse
    from infer import InferExperiment
    from utils.Plot import plot_st_embedding
    args = argparse.ArgumentParser()
    args.add_argument("--data_name", type=str, default="HER2ST")
    args.add_argument("--adata_file", type=str, default="../dataset/stdata/HER2ST/data.h5ad")
    args.add_argument("--root", type=str, default="Normal glands")
    args.add_argument("--job_dir", type=str, default="../result/causalLearn/")
    args.add_argument("--threshold", type=float, default=0.42)  # 0.09
    args.add_argument("-kde", type=bool, default=True)  # hist or kde to cal CMI
    args = args.parse_args()
    exp = InferExperiment(args)
    # plot_st_embedding(exp.adata, exp.adata.obs['cluster'])
    exp.infer(cluster_list, debug=True, debug_nodes=['Fibrous tissue'])
    # exp.infer(cluster_list, debug=True, debug_nodes=['sEndF', 'sEndG'])
    adj = exp.adata.uns[exp.cluster_key]
    return adj


def load_sort(train=True):
    if train:
        # col_sort = [['sA', 'sB', 'sC', 'sCmid', 'sD', 'sE', 'sEmid', 'sEndD', 'sEndF', 'sEndG', 'sF', 'sG'],
        #             ['sEndD', 'sA', 'sC', 'sCmid', 'sD', 'sE', 'sEmid', 'sEndF', 'sEndG', 'sB', 'sF', 'sG'],
        #             ['sCmid', 'sA', 'sEndG', 'sB', 'sC', 'sD', 'sE', 'sEmid', 'sEndD', 'sEndF', 'sF', 'sG']]
        col_sort = [['Normal glands', 'Tumor region', 'Immune cells', 'Invasive cancer', 'Fibrous tissue'],
                    ['Fibrous tissue', 'Invasive cancer', 'Immune cells', 'Normal glands', 'Tumor region'],
                    ['Immune cells', 'Tumor region', 'Normal glands', 'Invasive cancer', 'Fibrous tissue']]
    else:
        col_sort = [['breast glands', 'tumor region', 'immune infiltrate', 'invasive cancer', 'connective tissue'],
                    ['connective tissue', 'invasive cancer', 'immune infiltrate', 'breast glands', 'tumor region'],
                    ['immune infiltrate', 'tumor region', 'breast glands', 'invasive cancer', 'connective tissue']]
    return col_sort


if __name__ == '__main__':
    # method = 'ICALiNGAM'
    # method = 'DirectLiNGAM'
    method = 'CASCAT'
    # method = 'PC'
    # method = 'GES'
    for idx in range(3):
        col_sort = load_sort()
        cluster_feature = load_data()
        print(cluster_feature.columns)
        cluster_feature = cluster_feature[col_sort[idx]]
        if method == 'CASCAT':
            adj = infer(col_sort[idx])
            adj = adj.loc[col_sort[idx]]
            adj = adj[col_sort[idx]].values
        elif method == 'ICALiNGAM':
            model = ICALiNGAM()
            model.learn(cluster_feature.values)
            adj = model.weight_causal_matrix
        elif method == 'DirectLiNGAM':
            model = DirectLiNGAM()
            model.learn(cluster_feature.values)
            adj = model.causal_matrix
        elif method == 'GES':
            model = GES()
            model.learn(cluster_feature.values)
            adj = model.causal_matrix
        elif method == 'PC':
            model = PC()
            model.learn(cluster_feature.values)
            adj = model.causal_matrix
        else:
            raise ValueError(f"Invalid method: {method}")
        # adata.obs['cluster'] = adata.obs['cluster'].map(
        #     {'Normal glands': 'breast glands', 'Tumor region': 'tumor region', 'Immune cells': 'immune infiltrate',
        #      'Fibrous tissue': 'connective tissue', 'Invasive cancer': 'invasive cancer'})
        adj[adj < 0.3] = 0
        G = nx.DiGraph(adj)
        col_sort = load_sort(train=False)
        draw_pydot_graph(G, labels=col_sort[idx], figname=f"{method}_{idx}")
