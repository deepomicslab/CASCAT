import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
import pydot
import io

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
import matplotlib.image as mpimg
import umap

rcParams['font.family'] = 'Times New Roman'


def plot_ground_truth(adata, gt_key='milestone_network', save_path=None, direction='horizontal'):
    def nudge(pos, x_shift, y_shift):
        return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}

    DG = nx.DiGraph()
    if gt_key not in adata.uns_keys():
        raise Exception(f"Ground truth key {gt_key} not found in adata.uns")
    if isinstance(adata.uns[gt_key], dict):
        for edges in adata.uns[gt_key].values():
            DG.add_edge(edges[0], edges[1])
    else:
        for edges in adata.uns[gt_key]:
            DG.add_edge(edges[0], edges[1])
    pos = nx.nx_pydot.pydot_layout(DG, prog="dot")
    if direction == 'vertical':
        pos = {n: (y, x) for n, (x, y) in pos.items()}
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    nx.draw_networkx_nodes(DG, pos=pos, node_size=300)
    nx.draw_networkx_edges(DG, pos=pos, arrows=True, ax=ax, arrowstyle='->', arrowsize=10)
    if direction == 'vertical':
        min_, max_ = ax.get_ylim()
    else:
        min_, max_ = ax.get_xlim()
    pos_nodes = nudge(pos, (max_ - min_) * 0.1, 0)
    nx.draw_networkx_labels(DG, pos_nodes, labels={i: i for i in DG.nodes}, ax=ax, font_size=20)
    if direction == 'vertical':
        ax.set_ylim((min_ + (max_ - min_) * 0.1, max_ + (max_ - min_) * 0.1))
    else:
        ax.set_xlim((min_ + (max_ - min_) * 0.1, max_ + (max_ - min_) * 0.1))
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_gt_milestone_network(ad, uns_mn_key="milestone_network", start_milestones="sA", start_node_color="red",
                              node_color="blue", figsize=(12, 12), node_size=800, font_size=9, save_path=None,
                              **kwargs):
    if uns_mn_key not in ad.uns_keys():
        raise Exception(f"Milestone network not found in uns.{uns_mn_key}")
    # Get a set of unique milestones
    mn = ad.uns[uns_mn_key]
    from_milestones = set(mn["from"].unique())
    to_milestones = set(mn["to"].unique())
    milestones = from_milestones.union(to_milestones)
    # Construct the milestone network
    milestone_network = nx.DiGraph()
    start_milestones = (
        [start_milestones]
        if isinstance(start_milestones, str)
        else list(start_milestones))
    color_map = []
    for milestone in milestones:
        milestone_network.add_node(milestone)
        if milestone in start_milestones:
            color_map.append(start_node_color)
        else:
            color_map.append(node_color)
    for idx, (f, t) in enumerate(zip(mn["from"], mn["to"])):
        milestone_network.add_edge(f, t, weight=mn["length"][idx])
    plt.figure(figsize=figsize)
    plt.axis("off")
    edge_weights = [1 + w for _, _, w in milestone_network.edges.data("weight")]
    nx.draw_networkx(milestone_network, pos=nx.spring_layout(milestone_network), node_size=node_size,
                     width=edge_weights, node_color=color_map, font_size=font_size, **kwargs)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_spatial(data, labels, save_path=None, show=True, s=10, ls=8, color='Paired', ax=None, **kwargs):
    if ax is None:
        ax, fig = plt.subplots(1, 1, figsize=(8, 6))
    if 'spatial' in data.obsm.keys():
        xy_coord = data.obsm['spatial']
    else:
        raise Exception("No spatial information found in data.obsm")
    if 'cluster' not in data.obs.keys():
        raise Exception("No cluster label found in data.obs")
    sns.scatterplot(x=xy_coord[:, 0], y=xy_coord[:, 1], hue=labels, s=s, palette=color, **kwargs)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=ls, frameon=False)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return ax


def plot_spatial_graph(adata, img_path=None, tissue_pos_path=None, save_path=None, scale_x=0, scale_y=4,
                       ls=8, s=6, labels=None, colors='Paired'):
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'spatial' in adata.obsm.keys():
        x_coord = adata.obsm['spatial']
    else:
        raise Exception("No spatial information found in data.obsm")
    img = mpimg.imread(img_path)
    tissue_pos = pd.read_csv(tissue_pos_path, header=None)
    ax.imshow(img, alpha=0.6)
    size_factor_y = img.shape[1] / (max(tissue_pos.iloc[:, 5]) - min(tissue_pos.iloc[:, 5])) / 1.2
    size_factor_x = img.shape[0] / (max(tissue_pos.iloc[:, 4]) - min(tissue_pos.iloc[:, 4])) / 1.2
    y_ = x_coord[:, 1] * size_factor_y
    x_ = x_coord[:, 0] * size_factor_x
    y_ = y_ + x_coord[:, 1].min() * size_factor_y * scale_y
    x_ = x_ + x_coord[:, 0].min() * size_factor_x * scale_x
    sns.scatterplot(x=y_, y=x_, hue=labels,
                    s=s, ax=ax, palette=colors, legend=True)
    sns.despine()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.9), fontsize=ls, frameon=False, ncol=1)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_pydot_graph(G, labels, save_path=None, show=True, direction='LR'):
    nodes, edges = list(G.nodes), list(G.edges)
    if labels is not None:
        assert len(labels) == len(nodes)
    pydot_g = pydot.Dot("", graph_type="digraph", fontsize=18, rankdir=direction)
    pydot_g.obj_dict["attributes"]["dpi"] = 200
    for i, node in enumerate(nodes):
        pydot_g.add_node(pydot.Node(i, label=labels[node]))

    for edge in edges:
        node1, node2 = edge[0], edge[1]
        node1_id = nodes.index(node1)
        node2_id = nodes.index(node2)
        dot_edge = pydot.Edge(node1_id, node2_id, dir='left', arrowtail='normal', arrowhead='normal')
        pydot_g.add_edge(dot_edge)
    tmp_png = pydot_g.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.rcParams["figure.figsize"] = [20, 8]
    plt.rcParams["figure.autolayout"] = True
    plt.axis('off')
    plt.imshow(img)
    if save_path != None:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


def plot_pesudotime(emd, group_frac, connectivities, predict_labels, pesudotime, label_map, show=True, save_path=None,
                    colors='Set2'):
    predict_str_labels = np.array([str(i) for i in predict_labels])
    graph = nx.from_pandas_adjacency(connectivities, create_using=nx.DiGraph)

    def get_centriods(emb, pesudotime, group_frac):
        center_pos = pd.DataFrame(np.zeros([group_frac.shape[1], 2]), columns=['x', 'y'], index=group_frac.columns)
        center_pesudotime = pd.DataFrame(np.zeros([group_frac.shape[1], 1]), columns=['pesudotime'],
                                         index=group_frac.columns)
        for i in group_frac.columns:
            idx = np.where(predict_str_labels == i)[0]
            center_pos.loc[i, :] = np.mean(emb[idx, :], axis=0)
            center_pesudotime.loc[i, :] = np.mean(pesudotime[idx])
        return center_pos, center_pesudotime

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    reducer = umap.UMAP(n_components=2, random_state=5)
    emd_2pc = reducer.fit_transform(emd)
    centroids, center_pesudotime = get_centriods(emd_2pc, pesudotime, group_frac)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    data_min, data_max = min(pesudotime), max(pesudotime)
    norm = Normalize(vmin=data_min, vmax=data_max)
    sm = ScalarMappable(cmap=plt.get_cmap('YlGn'), norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=.03)
    cbar.set_label('Pseudotime', fontsize=10)
    sns.scatterplot(x=emd_2pc[:, 0], y=emd_2pc[:, 1], c=[sm.to_rgba(i) for i in pesudotime], s=5, ax=ax, legend=False)
    node_colors_mapped = [sm.to_rgba(center_pesudotime.loc[i, :]) for i in sorted(center_pesudotime.index)]
    nx.draw_networkx_nodes(graph, pos={i: centroids.loc[i, :] for i in centroids.index}, node_size=200,
                           node_color=node_colors_mapped, cmap=colors, ax=ax)
    nx.draw_networkx_edges(graph, pos={i: centroids.loc[i, :] for i in centroids.index}, arrows=True, ax=ax,
                           arrowstyle="-|>", connectionstyle="arc3,rad=0.2", width=1)
    nx.draw_networkx_labels(graph, pos={i: centroids.loc[i, :] for i in centroids.index}, labels=label_map, ax=ax,
                            font_size=8)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        return ax


def plot_embedding(emd, group_frac, connectivities, predict_labels, label_map, show=True, save_path=None,
                   colors='Paired'):
    predict_str_labels = np.array([str(i) for i in predict_labels])
    graph = nx.from_pandas_adjacency(connectivities, create_using=nx.DiGraph)

    def get_centriods(emb, group_frac):
        center_pos = pd.DataFrame(np.zeros([group_frac.shape[1], 2]), columns=['x', 'y'], index=group_frac.columns)
        for i in group_frac.columns:
            idx = np.where(predict_str_labels == i)[0]
            center_pos.loc[i, :] = np.mean(emb[idx, :], axis=0)
        return center_pos

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    reducer = umap.UMAP(n_components=2, random_state=0)
    emd_2pc = reducer.fit_transform(emd)
    centroids = get_centriods(emd_2pc, group_frac)
    if type(colors) == str:
        colors = sns.color_palette(colors, len(set(predict_str_labels)))
    sns.scatterplot(x=emd_2pc[:, 0], y=emd_2pc[:, 1], c=[colors[i] for i in predict_labels], s=5, ax=ax)
    nx.draw_networkx_nodes(graph, pos={i: centroids.loc[i, :] for i in centroids.index}, node_size=600,
                           node_color=[colors[int(i)] for i in sorted(centroids.index)], ax=ax, alpha=0.8)
    nx.draw_networkx_edges(graph, pos={i: centroids.loc[i, :] for i in centroids.index}, arrows=True, ax=ax,
                           arrowstyle="-|>", connectionstyle="arc3,rad=0.2", width=1.5)
    nx.draw_networkx_labels(graph, pos={i: centroids.loc[i, :] for i in centroids.index}, labels=label_map, ax=ax,
                            font_size=10)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path + 'emb.png', dpi=300)
    if show:
        plt.show()
    else:
        return ax


def plot_st_embedding(adata, predict_labels, colors='tab10', save_path=None, show_trajectory=False, group_frac=None,
                      connectivities=None):
    # locations = adata.obsm['spatial'][:, [1, 0]]
    locations = adata.obsm['spatial'][:, [0, 1]]
    # locations[:, 0] = -locations[:, 0]
    locations[:, 1] = -locations[:, 1]
    fig, ax1 = plt.subplots(figsize=(6, 5))
    df = pd.DataFrame(locations, columns=['x', 'y'])
    df['cluster'] = predict_labels
    if isinstance(colors, str):
        colors = sns.color_palette(colors, len(set(predict_labels)))
    ax1.scatter(locations[:, 0], locations[:, 1], c=[colors[int(i)] for i in predict_labels], s=20)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=False)
    if show_trajectory and group_frac is not None and connectivities is not None:
        centriods = pd.DataFrame(np.zeros([len(group_frac), 2]), columns=['x', 'y'], index=group_frac.columns)
        G = nx.from_pandas_adjacency(connectivities, create_using=nx.DiGraph)
        for i in group_frac.columns:
            # idx = np.where(predict_labels == i)[0]
            idx = np.where(predict_labels == int(i))[0]
            center = np.mean(locations[idx, :], axis=0)
            dist = np.sum((locations[idx, :] - center) ** 2, axis=1)
            center = locations[idx[np.argmin(dist)]]
            centriods.loc[i, :] = center
        nx.draw_networkx_edges(G, pos={i: centriods.loc[i, :] for i in centriods.index}, arrows=True,
                               ax=ax1, arrowstyle="-|>", connectionstyle="arc3,rad=0.3", width=2)
    ax1.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_st_pesudotime(adata, pesudotime, save_path=None):
    # locations = adata.obsm['spatial'][:, [1, 0]]
    locations = adata.obsm['spatial'][:, [0, 1]]
    # locations[:, 0] = -locations[:, 0]
    locations[:, 1] = -locations[:, 1]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    data_min, data_max = min(pesudotime), max(pesudotime)
    norm = Normalize(vmin=data_min, vmax=data_max)
    sm = ScalarMappable(cmap=plt.get_cmap('YlGn'), norm=norm)
    ax1.scatter(locations[:, 0], locations[:, 1], c=[sm.to_rgba(i) for i in pesudotime], s=20)
    cbar = plt.colorbar(sm, ax=ax1, fraction=.05)
    cbar.set_label('Pseudotime', fontsize=10)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=False)
    ax1.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_subtype(cluster, subtype, colors='Paired', show=True, img_path=None, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    df = pd.DataFrame({'subtype': subtype, 'cluster': cluster})
    pivoted_df = df.pivot_table(index='cluster', columns='subtype', aggfunc='size', fill_value=0, observed=True)
    pivoted_df.plot(kind='barh', stacked=True, ax=ax, color=colors, legend=False)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xlabel('Count', fontsize=15)
    ax.set_ylabel('Layer', fontsize=15)
    if title is not None:
        ax.set_title(title, fontsize=15)
    if img_path is not None:
        plt.savefig(img_path, dpi=300)
    if show:
        plt.show()
    else:
        return fig, ax
