import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .model_utils import *


class GraphLearner(nn.Module):
    def __init__(self, nlayers, isize, neighbor, gamma, adj, dis, device, omega):
        super().__init__()
        self.adj = adj.to(device)
        d_matrix = torch.tensor(dis, dtype=torch.float32, device=device)
        d_sorted, _ = d_matrix.sort()
        # mask
        c1 = d_matrix > 0
        d_cut = torch.median(d_sorted[:, neighbor])
        c2 = d_matrix <= d_cut
        self.adj_mask = torch.logical_and(c1, c2)  # 1-k neighbor, no self-loop
        print('The number of useful edges is {}'.format(self.adj_mask.sum()))
        d_matrix = torch.where(self.adj_mask, d_matrix, torch.inf) / d_cut
        self.s_d = 1 / torch.exp(gamma * torch.pow(d_matrix, 2))
        self.convs = nn.ModuleList()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=isize, out_channels=isize)] +
            [GCNConv(in_channels=isize, out_channels=isize) for _ in range(nlayers - 1)])
        self.input_dim = isize
        self.omega = omega

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
