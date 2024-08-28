import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from torch_sparse import SparseTensor
from .model_utils import get_feat_mask


class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_adj_p = dropout_adj
        self.relu = nn.ReLU(inplace=True)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(nlayers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, emb_dim))

        self.proj_head = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def dropout_adj(self, Adj):
        row, col, val = Adj.coo()

        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_mask = dropout_edge(edge_index, p=self.dropout_adj_p)

        num_nodes = Adj.size(0)
        val = val[edge_mask]
        Adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=val, sparse_sizes=(num_nodes, num_nodes))
        return Adj

    def forward(self, x, Adj, training):
        if training:
            Adj = self.dropout_adj(Adj)
        for conv in self.convs[:-1]:
            x = conv(x, Adj.t())
            x = self.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, Adj.t())
        z = self.proj_head(x)
        return x, z


class GCL(nn.Module):
    def __init__(self, nlayers, cell_feature_dim, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, margin,
                 bn):
        super().__init__()
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj)

        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_feature_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
        )
        if bn:
            self.cell_encoder.append(nn.BatchNorm1d(in_dim))

        self.graph_learner = None
        self.graph_learned = None
        self.margin_loss = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def get_cell_features(self, gene_exp):
        return self.cell_encoder(gene_exp)

    def get_learner_adj(self, cell_features):
        return self.graph_learner(cell_features)

    def get_learned_adj(self, cell_features):
        return self.graph_learned(cell_features)

    def forward(self, x_, Adj, maskfeat_rate=None, training=None):
        if maskfeat_rate is not None:
            mask = get_feat_mask(x_, maskfeat_rate)
            x = x_ * mask
        else:
            x = x_

        if training is None:
            training = self.training
        embedding, z = self.encoder(x, Adj, training)
        return embedding, z

    @staticmethod
    def sim_loss(x, x_aug, temperature, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1
