import torch
import torch.nn.functional as F
import random
import os
import numpy as np
from sklearn.cluster import KMeans
from models import GCL, GraphLearner, GraphLearned
from utils.data_loader import load_data_from_raw
from models.model_utils import refine_labels, normalize_adj_symm, dense2sparse
from utils.Metrics import ClusteringMetrics


class Experiment:
    def __init__(self, args):
        super().__init__()
        self.args = args
        gene_exp, labels, nclasses, adj_knn, dis, self.adata = load_data_from_raw(args)
        gene_exp = gene_exp.to(args.device)

        self.gene_exp = gene_exp
        self.labels = labels
        self.nclasses = nclasses
        self.adj_knn = adj_knn
        self.dis = dis
        self.dist_sort_idx = dis.argsort(axis=1)

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def evaluation(self, embedding):
        clu_model = self.args.clu_model
        if clu_model == 'kmeans':
            ari_ls, ami_ls = [], []
            for clu_trial in range(5):
                kmeans = KMeans(n_clusters=self.nclasses, random_state=clu_trial, n_init="auto").fit(embedding)
                predict_labels = kmeans.predict(embedding)
                if self.args.refine != 0:
                    predict_labels = refine_labels(predict_labels, self.dist_sort_idx, self.args.refine)
                    predict_labels = predict_labels.squeeze().copy()
                cm_all = ClusteringMetrics(self.labels.cpu().numpy(), predict_labels)
                ari, ami = cm_all.evaluationClusterModelFromLabel()
                ari_ls.append(ari)
                ami_ls.append(ami)
            ari, ami = np.mean(ari_ls), np.mean(ami_ls)
        elif clu_model == 'mclust':
            predict_labels = mclust_R(embedding, n_clusters=self.nclasses, random_state=0)
            if self.args.refine != 0:
                predict_labels = refine_labels(predict_labels, self.dist_sort_idx, self.args.refine)
            cm_all = ClusteringMetrics(self.labels.cpu().numpy(), predict_labels)
            ari, ami = cm_all.evaluationClusterModelFromLabel()
        else:
            raise Exception(f'Unknown cluster model {clu_model}')

        return ari, ami

    def train(self, verbose=True):
        args = self.args
        if not os.path.exists(f'{args.job_dir}/'):
            os.makedirs(f'{args.job_dir}/')
        with open(os.path.join(f'{args.job_dir}/', 'args.txt'), 'w') as f:
            print(args, file=f)
        best_ari, seed_record, ari_records, ami_records, best_emb = -np.inf, [], [], [], None
        self.setup_seed(args.seed)
        anchor_adj = normalize_adj_symm(self.adj_knn).to(args.device)
        bn = not args.no_bn
        model = GCL(nlayers=args.nlayers, cell_feature_dim=self.gene_exp.size(1), in_dim=args.exp_out,
                    hidden_dim=args.hidden_dim, emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                    dropout=args.dropout, dropout_adj=args.dropedge_rate, margin=args.margin, bn=bn)
        if args.learned_graph == 'CMI':
            model.graph_learned = GraphLearned(nlayers=args.nlayers, isize=args.exp_out, neighbor=args.a_k,
                                               gamma=args.gamma, adj=anchor_adj, dis=self.dis,
                                               device=args.device, omega=args.adj_weight,
                                               cmi_dir=args.CMI_dir + str(args.a_k) + '/', expr=self.gene_exp,
                                               percent=args.percent)
        else:
            model.graph_learner = GraphLearner(nlayers=args.nlayers, isize=args.exp_out, neighbor=args.a_k,
                                               gamma=args.gamma, adj=anchor_adj, dis=self.dis,
                                               device=args.device, omega=args.adj_weight)

        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        print(model)
        identity = dense2sparse(torch.eye(self.gene_exp.shape[0])).to(args.device)
        for epoch in range(1, 1 + args.epochs):
            optimizer.zero_grad()
            model.train()
            cell_features = model.get_cell_features(self.gene_exp)
            _, z1 = model(cell_features, anchor_adj, args.maskfeat_rate_anchor)
            if args.learned_graph == 'CMI':
                learned_adj, _ = model.get_learned_adj(cell_features)
            else:
                learned_adj, _ = model.get_learner_adj(cell_features)
            _, z2 = model(cell_features, learned_adj, args.maskfeat_rate_learner)
            idx = torch.randperm(self.gene_exp.shape[0])
            _, z1_neg = model(cell_features[idx], identity, args.maskfeat_rate_anchor, training=False)
            d_pos = F.pairwise_distance(z2, z1)
            d_neg = F.pairwise_distance(z2, z1_neg)
            margin_label = -1 * torch.ones_like(d_pos)
            sim_loss = model.sim_loss(z1, z2, args.temperature, sym=False) * args.sim_weight
            margin_loss = model.margin_loss(d_pos, d_neg, margin_label) * args.margin_weight
            loss = sim_loss + margin_loss
            loss.backward()
            optimizer.step()
            if args.c == 0 or epoch % args.c == 0:  # Structure Bootstrapping
                anchor_adj = dense2sparse(
                    anchor_adj.to_dense() * args.tau + learned_adj.detach().to_dense() * (1 - args.tau))
            model.eval()
            if (epoch - 1) % args.einterval == 0:
                with torch.no_grad():
                    embedding, _ = model(cell_features.detach(), learned_adj)
                embedding = embedding.cpu().detach().numpy()
                if verbose:
                    ari, ami = self.evaluation(embedding)
                    seed_record.append(args.seed)
                    ari_records.append(ari)
                    ami_records.append(ami)
                    print("Epoch {:05d} | CL Loss {:.5f} | Margin Loss {:.5f} | ARI {:5f}| AMI {:5f}".format(
                        epoch, sim_loss.item(), margin_loss.item(), ari, ami))
                    if ari > best_ari:
                        best_ari = ari
                        best_emb = embedding
                else:
                    print("Epoch {:05d} | CL Loss {:.5f} | Margin Loss {:.5f}".format(
                        epoch, sim_loss.item(), margin_loss.item()))
        emb_path = f'{args.job_dir}/trial{str(args.seed)}_ARI{best_ari:.5f}.npy'
        # np.save(emb_path, best_emb)
        np.save(emb_path, embedding)
        return seed_record, ari_records, ami_records

    def sc_cluster(self):
        sc.pp.pca(self.adata, random_state=self.args.seed)
        embedding = self.adata.obsm['X_pca']
        # self.adata.obsp['connectivities'] = get_CMI_connectivities(self.adata, self.args.CMI_dir, percent=self.args.percent)
        ari, ami = self.evaluation(embedding)
        print(f"ARI is {ari}, AMI is {ami}")
        if not os.path.exists(f'{self.args.job_dir}/'):
            os.makedirs(f'{self.args.job_dir}/')
        emb_path = f'{self.args.job_dir}/trial{str(self.args.seed)}_ARI{ari:.5f}.npy'
        np.save(emb_path, embedding)
        return emb_path, ari, ami
