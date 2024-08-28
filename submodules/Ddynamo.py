# dynamo
import argparse
import os
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as scx
import seaborn as sns
import yaml
import dynamo as dyn
from sklearn.cluster import KMeans
import sys

sys.path.append('/home/a/yingyingyu/STCMI')
# sys.path.append('/mnt/c/Users/yingyinyu3/PycharmProjects/STCMI')

from cluster import Experiment
from infer import InferExperiment
from utils.Plot import plot_embedding


def get_genes(adata_template, ic50_dgidb, i, all_genes):
    # Get genes_up, genes_down, and nz_valid_genes
    nonzero_input = adata_template.layers["spliced"]
    nz_input_genesums = nonzero_input.sum(axis=0)
    valid_ind = np.logical_and(np.isfinite(nz_input_genesums), nz_input_genesums != 0)
    valid_ind = np.array(valid_ind).flatten()
    bad_genes = adata_template.var[~valid_ind]
    valid_genes = set(all_genes) - (set(all_genes) - set(adata_template.var.index.to_series().tolist()))
    nz_valid_genes = valid_genes - set(bad_genes.index.tolist()).intersection(valid_genes)
    genes_up = ic50_dgidb.loc[i]['genes_up']
    genes_down = ic50_dgidb.loc[i]['genes_down']
    genes_up = str(genes_up).replace('nan', '').split(" ")
    genes_up = set(genes_up).intersection(nz_valid_genes)
    genes_down = str(genes_down).replace('nan', '').split(" ")
    genes_down = set(genes_down).intersection(nz_valid_genes)
    if '' in genes_down:
        genes_down.remove('')
    if '' in genes_up:
        genes_up.remove('')
    nz_valid_genes = adata_template.var.index.to_series().tolist()
    return genes_up, genes_down, nz_valid_genes


def perturb_condition(adata_pt, genes_up, genes_down, genes_total, i, ic50_dgidb_scored, dataname):
    if (len(genes_up) >= 1 and len(genes_down) >= 1):
        print("up and down")
        expr_vals = [-200] * len(genes_down) + [200] * len(genes_up)
    elif (len(genes_up) >= 1 and len(genes_down) == 0):
        print("only up")
        expr_vals = [200] * len(genes_up)
    elif (len(genes_up) == 0 and len(genes_down) >= 1):
        print("only down")
        expr_vals = [-200] * len(genes_down)
    else:
        print("no perturb")
        return ic50_dgidb_scored
    # run_pertrubation(adata_pt, genes_total, expr_vals, i, dataname)
    ic50_dgidb_scored = run_perturb_analysis(adata_pt, genes_total, expr_vals, i, ic50_dgidb_scored, dataname)
    return ic50_dgidb_scored


def run_perturb_analysis(adata_pt, genes_total, expr_vals, i, ic50_dgidb_scored, dataname):
    try:
        print(i)
        print(expr_vals)
        print(genes_total)
        dyn.pd.perturbation(adata_pt, genes_total, expr_vals, emb_basis="umap")
        dyn.vf.VectorField(adata_pt, basis='umap_perturbation', M=50)
        dyn.pl.streamline_plot(adata_pt, basis="umap_perturbation", color=["cluster_annotations"],
                               color_key=["#4DBBD5FF", "#E64B35FF", "#F9E076"],
                               show_legend=False, calpha=1, pointsize=0.1,
                               save_show_or_return="save",
                               save_kwargs={"path": f"../img/state_graph/{dataname}_{i}_vector", "prefix": 'scatter',
                                            "dpi": 300, "ext": 'png', "transparent": False, "close": True,
                                            "verbose": True},
                               figsize=(5, 5), dpi=300)
        # adata_pt_tmp = adata_pt.copy()
        # adata_pt_tmp.layers["raw"] = adata_pt_tmp.X.copy()
        # adata_pt_tmp.X = adata_pt_tmp.layers['j_delta_x_perturbation'].copy()
        # scx.write(f"../dataset/stdata/{dataname}/{i}_perturb.h5ad", adata_pt_tmp)
        print("Generating state graph")
        dyn.pd.state_graph(adata_pt, group='cluster', basis='umap_perturbation',
                           transition_mat_key='perturbation_transition_matrix')
        Pl = adata_pt.uns["cluster_graph"]["group_graph"]
        flat_array = Pl.flatten()
        print(flat_array)
        ic50_dgidb_scored.at[i, "pertrub_mtx"] = list(flat_array)
        # print("making state graph plot")
        # dyn.pl.state_graph(adata_pt, color=['cluster'], color_key=["#4DBBD5FF", "#E64B35FF", "#F9E076"],
        #                    group='cluster_annotations', basis='umap_perturbation', save_show_or_return='save',
        #                    save_kwargs={"path": f"../img/state_graph/{dataname}_{i}_sg", "prefix": 'scatter',
        #                                 "dpi": 300, "ext": 'png', "transparent": False, "close": True, "verbose": True},
        #                    keep_only_one_direction=False,
        #                    figsize=(5, 5))
        return ic50_dgidb_scored
    except:
        return ic50_dgidb_scored


def run_perturbation_analysis(adata_pt, genes_up, genes_down, i, ic50_dgidb_scored, dataname):
    genes_total = list(genes_down) + list(genes_up)
    dyn.pp.recipe_monocle(adata_pt, genes_to_append=genes_total)
    dyn.tl.dynamics(adata_pt, cores=9)
    dyn.tl.reduceDimension(adata_pt)
    dyn.tl.cell_velocities(adata_pt, basis='pca')
    dyn.tl.cell_velocities(adata_pt, basis='umap')
    dyn.vf.VectorField(adata_pt, basis='pca', M=50)
    dyn.vf.VectorField(adata_pt, basis='umap', M=50)
    ic50_dgidb_scored = perturb_condition(adata_pt, genes_up, genes_down, genes_total, i, ic50_dgidb_scored, dataname)
    return ic50_dgidb_scored


def process_data(adata, ic50_dgidb, all_genes, dataname):
    ic50_dgidb_scored = ic50_dgidb.copy()
    for i in ic50_dgidb.index:
        # for i in ['Tanespimycin', 'Pictilisib', 'Paclitaxel', 'Tivantinib', 'Vorinostat', 'AZD8055', 'Entinostat',
        #           'Mitoxantrone', 'Alisertib', 'Irinotecan', 'Taselisib', 'Pevonedistat', 'Buparlisib', 'Afuresertib',
        #           'Talazoparib', 'Osimertinib', 'Dactolisib', 'Foretinib', 'Lovastatin', 'Tipifarnib', 'Sgx-523',
        #           'Pha-793887', 'Momelotinib', 'KX2-391', 'Belinostat', 'Onalespib']:
        adata_pt = adata.copy()
        genes_up, genes_down, nz_valid_genes = get_genes(adata_pt, ic50_dgidb, i, all_genes)
        ic50_dgidb_scored = run_perturbation_analysis(adata_pt, genes_up, genes_down, i,
                                                      ic50_dgidb, dataname)
    return ic50_dgidb_scored


def prepare_ic50_dgidb(ic50_dgidb):
    genes_down = set(re.sub(' +', ' ',
                            ic50_dgidb['genes_down'].to_string(index=False).strip().replace('\n', ' ').replace('NaN',
                                                                                                               '')).split(
        " "))
    genes_down = list(genes_down)
    genes_up = set(
        re.sub(' +', ' ', ic50_dgidb['genes_up'].to_string(index=False).strip().replace('\n', ' ').replace('NaN',
                                                                                                           '')).split(
            " "))
    genes_up = list(genes_up)
    all_genes = genes_down + genes_up
    all_genes.remove('')
    ic50_dgidb = ic50_dgidb.rename(index=ic50_dgidb.iloc[0:len(ic50_dgidb)]['Drug Name'])
    ic50_dgidb["pertrub_mtx"] = [[0, 0, 0, 0, 0, 0, 0, 0, 0]] * len(ic50_dgidb)
    ic50_dgidb = ic50_dgidb.dropna(subset=['genes_up', 'genes_down'], how='all')
    return ic50_dgidb, all_genes


def run_ic50_dgidb_score(ic50_dgidb, adata, dataname):
    ic50_dgidb, all_genes = prepare_ic50_dgidb(ic50_dgidb)
    ic50_dgidb_scored = process_data(adata, ic50_dgidb, all_genes, dataname)
    ic50_dgidb_scores = ic50_dgidb_scored[
        ic50_dgidb_scored['pertrub_mtx'].apply(lambda x: x != [0, 0, 0, 0, 0, 0, 0, 0, 0])]
    ic50_dgidb_scores['edge_outgoing'] = [x + y for x, y in zip([el[3] for el in ic50_dgidb_scores['pertrub_mtx']],
                                                                [el[5] for el in ic50_dgidb_scores['pertrub_mtx']])]

    ic50_dgidb_scores['edge_incoming'] = [x + y for x, y in zip([el[1] for el in ic50_dgidb_scores['pertrub_mtx']],
                                                                [el[7] for el in ic50_dgidb_scores['pertrub_mtx']])]

    ic50_dgidb_scores['core_incoming'] = [x + y for x, y in zip([el[2] for el in ic50_dgidb_scores['pertrub_mtx']],
                                                                [el[5] for el in ic50_dgidb_scores['pertrub_mtx']])]

    ic50_dgidb_scores['core_outgoing'] = [x + y for x, y in zip([el[6] for el in ic50_dgidb_scores['pertrub_mtx']],
                                                                [el[7] for el in ic50_dgidb_scores['pertrub_mtx']])]

    ic50_dgidb_scores['core_net'] = ic50_dgidb_scores['core_incoming'] - ic50_dgidb_scores['core_outgoing']
    ic50_dgidb_scores['edge_net'] = ic50_dgidb_scores['edge_incoming'] - ic50_dgidb_scores['edge_outgoing']
    ic50_dgidb_scored.to_csv("../analysis/ic50_dgidb_scored.csv")


def run_pertrubation(adata, genes, expression, drug, dataname):
    if type(genes) == str:
        genes = [genes]
    if type(expression) in [int, float]:
        expression = [expression]
    pca_genes = adata.var_names[adata.var.use_for_pca]
    valid_genes = pca_genes.intersection(genes)
    if len(valid_genes) == 0:
        raise ValueError("genes to perturb must be pca genes (genes used to perform the pca dimension reduction).")
    if len(expression) > 1:
        if len(expression) != len(valid_genes):
            raise ValueError(
                "if you want to set different values for different genes, you need to ensure those genes "
                "are included in the pca gene list and the length of those genes is the same as that of the"
                "expression.")
    X = adata.X
    gene_loc = [adata.var_names[adata.var.use_for_pca].get_loc(i) for i in valid_genes]
    X_perturb = X.copy()
    cells = np.arange(adata.n_obs)
    for i, gene in enumerate(gene_loc):
        X_perturb[cells, gene] = expression[i]
    adata.layers["perturbed"] = X_perturb
    scx.write(f"../dataset/stdata/{dataname}/{drug}_perturb.h5ad", adata)
    return adata


def run_cluster(args, adata):
    exp = Experiment(args)
    args.emb_path, ari, ami, args.job_dir = exp.train(verbose=True)
    if args.emb_path is None:
        scx.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        scx.tl.umap(adata, n_components=10, random_state=args.seed)
        embedding = adata.obsm['X_umap']
    else:
        embedding = np.load(args.emb_path, allow_pickle=True)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(embedding)
    # predict_labels = kmeans.predict(embedding)
    # adata.obs['cluster'] = predict_labels
    # adata.obs['cascat_clusters'] = predict_labels
    adata.obsm['embeddings'] = embedding
    return adata


def run_infer(args, adata):
    infer_exp = InferExperiment(args, adata)
    infer_exp.infer()


def plot_spatial(adata, args):
    fig, ax = plt.subplots()
    x_coord = adata.obsm['spatial'].copy()
    cat = args.data_name.replace("SCC", "")
    img = mpimg.imread(f"../dataset/stdata/OSCC{cat}/spatial/tissue_hires_image.png")
    tissue_pos = pd.read_csv(f"../dataset/stdata/OSCC{cat}/spatial/tissue_positions_list.csv", header=None)
    ax.imshow(img, alpha=0.6)
    size_factor_y = img.shape[1] / (max(tissue_pos.iloc[:, 5]) - min(tissue_pos.iloc[:, 5])) / 1.2
    size_factor_x = img.shape[0] / (max(tissue_pos.iloc[:, 4]) - min(tissue_pos.iloc[:, 4])) / 1.2
    x_coord[:, 1] = x_coord[:, 1] * size_factor_y
    x_coord[:, 0] = x_coord[:, 0] * size_factor_x
    x_coord[:, 0] = x_coord[:, 0] + 188 * size_factor_y * 4
    x_coord[:, 1] = x_coord[:, 1] + 188 * size_factor_x * 0
    colors = ["#F9E076", "#4DBBD5FF", "#E64B35FF"]
    sns.scatterplot(x=x_coord[:, 1], y=x_coord[:, 0], hue=adata.obs['cluster_annotations'],
                    s=6, ax=ax, palette=colors, legend=False)
    plt.savefig(f"../img/OSCC{cat}/scc.png", dpi=300)
    plt.title(args.data_name)
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='SCC')
    parser.add_argument('--yml_dir', type=str, default='../config/yaml/GL/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drug', type=str, default=None)
    args = parser.parse_args()
    args.yml_path = os.path.join(args.yml_dir, args.data_name + '.yml')
    with open(args.yml_path, 'r') as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)
    for key in dict.keys():
        args.__dict__[key] = dict[key]
    args.adata_file = args.adata_file.replace("./", "../")
    args.clu_dir = args.clu_dir.replace("./", "../")
    args.img_dir = args.img_dir.replace("./", "../")
    args.job_dir = args.job_dir.replace("./", "../")
    args.emb_path = args.emb_path.replace("./", "../")
    args.output_dir = args.output_dir.replace("./", "../")
    if args.drug is not None:
        args.adata_file = args.adata_file.replace("data.h5ad", args.drug + "_perturb.h5ad")
        args.job_dir = args.job_dir + args.drug + "_perturb/"
        args.output_dir = args.output_dir + args.drug + "_perturb/"
        args.clu_dir = args.clu_dir + args.drug + "_perturb/"
        args.job_dir = args.clu_dir.replace("clu_result", "clu_result/SCC_perturb")
    print(args)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    adata = scx.read_h5ad(args.adata_file)
    # # adata = run_cluster(args, adata)# set cluster by our model
    # adata.obs['cascat_clusters'] = adata.obs['cluster'].copy()
    # # plot_spatial(adata, args)
    # run_infer(args, adata)
    ic50_dgidb = pd.read_excel("../analysis/Trimmed_AAC_means.xlsx")
    # print(ic50_dgidb)
    run_ic50_dgidb_score(ic50_dgidb, adata, args.data_name)
