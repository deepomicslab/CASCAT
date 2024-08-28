import copy
from sklearn.cluster import KMeans
import random
import torch.nn.functional as F
import datetime

from models import GCL, GraphLearner, GraphLearned
from utils.data_loader import *
from models.model_utils import *
from utils.Metrics import ClusteringMetrics


class Experiment:
    def __init__(self, args):
        super().__init__()
        self.args = args
        gene_exp, labels, nclasses, adj_knn, dis = load_data_from_raw(args)
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
        job_dir = args.clu_dir + '/' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        with open(os.path.join(job_dir, 'args.txt'), 'w') as f:
            print(args, file=f)

        record_ls = []
        for trial in range(args.ntrials):
            self.setup_seed(trial)
            anchor_adj = normalize_adj_symm(self.adj_knn).to(args.device)
            bn = not args.no_bn
            model = GCL(nlayers=args.nlayers, cell_feature_dim=self.gene_exp.size(1), in_dim=args.exp_out,
                        hidden_dim=args.hidden_dim, emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                        dropout=args.dropout, dropout_adj=args.dropedge_rate, margin=args.margin, bn=bn)
            if args.learned_graph == 'CMI':
                model.graph_learned = GraphLearned(nlayers=args.nlayers, isize=args.exp_out, neighbor=args.k,
                                                   gamma=args.gamma, adj=anchor_adj, dis=self.dis,
                                                   device=args.device, omega=args.adj_weight,
                                                   cmi_dir=args.CMI_dir, expr=self.gene_exp, percent=args.percent)
            else:
                model.graph_learner = GraphLearner(nlayers=args.nlayers, isize=args.exp_out, neighbor=args.k,
                                                   gamma=args.gamma, adj=anchor_adj, dis=self.dis,
                                                   device=args.device, omega=args.adj_weight)

            model = model.to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            print(model)
            best_ari = -np.inf
            best_embedding = None
            best_model = None
            ari_records = []
            ami_records = []
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
                        ari_records.append(ari)
                        ami_records.append(ami)
                        print("Epoch {:05d} | CL Loss {:.5f} | Margin Loss {:.5f} | ARI {:5f}| AMI {:5f}".format(
                            epoch, sim_loss.item(), margin_loss.item(), ari, ami))
                        if ari > best_ari:
                            best_ari = ari
                            best_embedding = embedding
                            best_model = copy.deepcopy(model)
                    else:
                        best_embedding = embedding
                        print("Epoch {:05d} | CL Loss {:.5f} | Margin Loss {:.5f}".format(
                            epoch, sim_loss.item(), margin_loss.item()))

            # save best embedding and model for each trial
            emb_path = os.path.join(job_dir, 'trial{}_ARI{:.5f}.npy'.format(trial, best_ari))
            np.save(emb_path, best_embedding)
            # model_path = os.path.join(job_dir, 'trial{}_model.pt'.format(trial))
            # torch.save(best_model.state_dict(), model_path)
            if verbose:
                df_record = pd.DataFrame({'trial': trial, 'ARI': ari_records, 'AMI': ami_records})
                record_ls.append(df_record)
        # save job args
        if verbose:
            best_ari_ls, best_ami_ls, best_trial_ls = [], [], []
            for df_ in record_ls:
                idx = df_['ARI'].argmax()
                best_ari_ls.append(df_['ARI'][idx])
                best_ami_ls.append(df_['AMI'][idx])
                best_trial_ls.append(df_['trial'][idx])
            best_ari = np.max(best_ari_ls)
            best_trial = best_trial_ls[np.argmax(best_ari_ls)]
            emb_path = os.path.join(job_dir, 'trial{}_ARI{:.5f}.npy'.format(best_trial, best_ari))
            all_record = pd.concat(record_ls, ignore_index=True)
            all_record.to_csv(os.path.join(job_dir, 'training_record.csv'), index=False)
            metric_path = os.path.join(job_dir,
                                       'metric_result_meanARI{:.5f}_stdARI{:.5f}_meanAMI{:.5f}_stdAMI{:.5f}'.format(
                                           np.mean(best_ari_ls), np.std(best_ari_ls),
                                           np.mean(best_ami_ls), np.std(best_ami_ls)))
            open(metric_path, 'a').close()
        else:
            best_ari_ls, best_ami_ls = [0], [0]
        return emb_path, np.mean(best_ari_ls), np.mean(best_ami_ls), job_dir
