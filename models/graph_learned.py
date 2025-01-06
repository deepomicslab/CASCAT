import torch.nn.functional as F
import os
import torch.nn as nn
import torch.sparse
import pandas as pd
import numpy as np
from models.model_utils import normalize_adj_symm, df2tensor, generate_CMI_from_adj
from torch_geometric.nn import GCNConv


class GraphLearned(torch.nn.Module):
    def __init__(self, nlayers, isize, neighbor, gamma, adj, dis, device, omega, cmi_dir, expr, percent):
        super().__init__()
        self.adj = adj.to(device)
        d_matrix = torch.tensor(dis, dtype=torch.float32, device=device)
        d_sorted, _ = d_matrix.sort()
        c1 = d_matrix > 0
        d_cut = torch.median(d_sorted[:, neighbor])  # on the base of k to remove long distance
        c2 = d_matrix <= d_cut
        adj_mask = torch.logical_and(c1, c2)  # 1-k neighbor, no self-loop
        if percent == 0:
            self.adj_mask = adj_mask
        else:
            self.adj_mask = self.init_adj_mask(cmi_dir, adj_mask, device=device, exprs=expr, percent=percent)
        print('The number of useful edges is {}'.format(self.adj_mask.sum()))
        d_matrix = torch.where(self.adj_mask, d_matrix, torch.inf) / d_cut
        self.s_d = 1 / torch.exp(gamma * torch.pow(d_matrix, 2))

        self.convs = nn.ModuleList()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=isize, out_channels=isize)] +
            [GCNConv(in_channels=isize, out_channels=isize) for _ in range(nlayers - 1)])
        self.input_dim = isize
        self.omega = omega

    def init_adj_mask(self, path, adj, device, exprs, percent=0.5):
        if not os.path.exists(path + "CMI.csv"):
            if not os.path.exists(path):
                os.makedirs(path)
            generate_CMI_from_adj(adj_mask=adj, gene_exprs=exprs, save_path=path)
        conditional_mutual_info = pd.read_csv(path + "CMI.csv")
        group_CMI = conditional_mutual_info.groupby(['Cell1', 'Cell2'])['CMI'].mean()
        group_CMI = group_CMI.loc[group_CMI > -np.inf]
        theshold = group_CMI.quantile(percent)
        group_CMI = group_CMI.loc[group_CMI > theshold]
        group_CMI.fillna(np.mean(group_CMI.values), inplace=True)
        if group_CMI.min() < 0:
            group_CMI = group_CMI - group_CMI.min()
        adj = df2tensor(group_CMI.reset_index(), adj)
        adj = torch.Tensor(adj.toarray())
        adj = adj > 0
        adj = adj.to(device)
        return adj

    def internal_forward(self, h):
        for i, conv in enumerate(self.convs):
            h = conv(h, self.adj.t())
            if i != (len(self.convs) - 1):
                h = F.relu(h, inplace=True)
        return h

    def forward(self, features, eps=1e-8):
        h = self.internal_forward(features)
        h_norm = torch.linalg.vector_norm(h, ord=2, dim=1, keepdim=True)
        s1 = (h @ h.t()) / (h_norm @ h_norm.t() + eps)
        s1 = torch.where(s1 >= 0, s1, 0)  # symmetric, [0,1]

        s2 = self.s_d
        s = self.omega * s1 + (1 - self.omega) * s2
        s = torch.where(self.adj_mask, s, 0)
        s_norm = normalize_adj_symm(s)
        return s_norm, s
