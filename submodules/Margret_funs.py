import numpy as np
import pandas as pd
import scanpy as sc
import os
import copy
import time
from torch.utils.data import DataLoader

from functools import wraps
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import gc
import torch
from sklearn.manifold import TSNE
import umap
from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import networkx as nx


def compute_runtime(func):
    @wraps(func)
    def f(*args, **kwargs):
        start_time = time.time()
        r = func(*args, **kwargs)
        end_time = time.time()
        print(f"Runtime for {func.__name__}(): {end_time - start_time}")
        return r

    return f


@compute_runtime
def preprocess_recipe(
        adata,
        min_expr_level=None,
        min_cells=None,
        use_hvg=False,
        scale=False,
        n_top_genes=1500,
        pseudo_count=1.0,
):
    """A simple preprocessing recipe for scRNA data
    Args:
        adata (sc.AnnData): Input annotated data object
        min_expr_level (int, optional): Min expression level for each cell. Defaults to None.
        min_cells (int, optional): Min. expression level of a gene. Defaults to None.
        use_hvg (bool, optional): Whether to select highly variable genes for analysis. Defaults to False.
        scale (bool, optional): Whether to perform z-score normalization. Defaults to False.
        n_top_genes (int, optional): No of highly variable genes to select if use_hvg is True. Defaults to 1500.
        pseudo_count (float, optional): Pseudo count to use for log-normalization. Defaults to 1.0.

    Returns:
        [sc.AnnData]: Preprocessed copy of the input annotated data object
    """
    preprocessed_data = adata.copy()

    # Some type conversion!
    if isinstance(preprocessed_data.X, csr_matrix) or isinstance(
            preprocessed_data.X, csc_matrix
    ):
        preprocessed_data.X = preprocessed_data.X.todense()
    if isinstance(preprocessed_data.X, np.matrix):
        preprocessed_data.X = np.asarray(preprocessed_data.X)

    print("Preprocessing....")

    if min_expr_level is not None:
        sc.pp.filter_cells(preprocessed_data, min_counts=min_expr_level)
        print(f"\t->Removed cells with expression level<{min_expr_level}")

    if min_cells is not None:
        sc.pp.filter_genes(preprocessed_data, min_cells=min_cells)
        print(f"\t->Removed genes expressed in <{min_cells} cells")

    sc.pp.normalize_total(preprocessed_data)
    log_transform(preprocessed_data, pseudo_count=pseudo_count)
    print("\t->Normalized data")

    if use_hvg:
        sc.pp.highly_variable_genes(
            preprocessed_data, n_top_genes=n_top_genes, flavor="cell_ranger"
        )
        print(f"\t->Selected the top {n_top_genes} genes")

    if scale:
        print("\t->Applying z-score normalization")
        sc.pp.scale(preprocessed_data)
    print(f"Pre-processing complete. Updated data shape: {preprocessed_data.shape}")
    return preprocessed_data


def log_transform(data, pseudo_count=1):
    """Perform log-transformation of scRNA data

    Args:
        data ([sc.AnnData, np.ndarray, pd.DataFrame]): Input data
        pseudo_count (int, optional): [description]. Defaults to 1.
    """
    if type(data) is sc.AnnData:
        data.X = np.log2(data.X + pseudo_count) - np.log2(pseudo_count)
    else:
        return np.log2(data + pseudo_count) - np.log2(pseudo_count)


@compute_runtime
def run_pca(
        data, n_components=300, use_hvg=True, variance=None, obsm_key=None, random_state=0
):
    """Helper method to compute PCA of the data. Uses the sklearn
    implementation of PCA.

    Args:
        data (sc.AnnData): Input annotated data
        n_components (int, optional): Number of PCA components. Defaults to 300.
        use_hvg (bool, optional): Whether to use highly variable genes for PCA computation. Defaults to True.
        variance (float, optional): Variance to account for. Defaults to None.
        obsm_key (str, optional): An optional key to specify for which data to compute PCA for. Defaults to None.
        random_state (int, optional): Random state. Defaults to 0.

    Returns:
        [type]: [description]
    """
    if not isinstance(data, sc.AnnData):
        raise Exception(f"Expected data to be of type sc.AnnData found: {type(data)}")

    data_df = data.to_df()
    if obsm_key is not None:
        data_df = data.obsm[obsm_key]
        if isinstance(data_df, np.ndarray):
            data_df = pd.DataFrame(
                data_df, index=data.obs_names, columns=data.var_names
            )

    # Select highly variable genes if enabled
    X = data_df.to_numpy()
    if use_hvg:
        valid_cols = data_df.columns[data.var["highly_variable"] == True]
        X = data_df[valid_cols].to_numpy()

    if variance is not None:
        # Determine the number of components dynamically
        comps_ = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=comps_, random_state=random_state)
        pca.fit(X)
        try:
            n_comps = np.where(np.cumsum(pca.explained_variance_ratio_) > variance)[0][
                0
            ]
        except IndexError:
            n_comps = n_components
    else:
        n_comps = n_components

    # Re-run with selected number of components (Either n_comps=n_components or
    # n_comps = minimum number of components required to explain variance)
    pca = PCA(n_components=n_comps, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_, n_comps


@compute_runtime
def determine_cell_clusters(
        data,
        obsm_key="X_pca",
        backend="phenograph",
        cluster_key="clusters",
        nn_kwargs={},
        **kwargs,
):
    """Run clustering of cells"""
    if not isinstance(data, sc.AnnData):
        raise Exception(f"Expected data to be of type sc.AnnData found : {type(data)}")
    try:
        X = data.obsm[obsm_key]
    except KeyError:
        raise Exception(f"Either `X_pca` or `{obsm_key}` must be set in the data")
    if backend == "kmeans":
        kmeans = KMeans(**kwargs)
        clusters = kmeans.fit_predict(X)
        score = kmeans.inertia_
        data.obs[cluster_key] = clusters
    elif backend == "louvain":
        # Compute nearest neighbors
        sc.pp.neighbors(data, use_rep=obsm_key, **nn_kwargs)
        sc.tl.louvain(data, key_added=cluster_key, **kwargs)
        data.obs[cluster_key] = data.obs[cluster_key].to_numpy().astype(np.int64)
        clusters = data.obs[cluster_key]
        score = None
    elif backend == "leiden":
        # Compute nearest neighbors
        sc.pp.neighbors(data, use_rep=obsm_key, **nn_kwargs)
        sc.tl.leiden(data, key_added=cluster_key, **kwargs)
        data.obs[cluster_key] = data.obs[cluster_key].to_numpy().astype(np.int64)
        clusters = data.obs[cluster_key]
        score = None
    else:
        raise NotImplementedError(f"The backend {backend} is not supported yet!")
    return clusters, score


def get_start_cell_cluster_id(data, start_cell_ids, communities):
    start_cluster_ids = set()
    obs_ = data.obs_names
    for cell_id in start_cell_ids:
        start_cell_idx = np.where(obs_ == cell_id)[0][0]
        start_cell_cluster_idx = communities[start_cell_idx]
        start_cluster_ids.add(start_cell_cluster_idx)
    return start_cluster_ids


def prune_network_edges(communities, adj_sc, adj_cluster):
    cluster_ids = np.unique(communities)

    # Create cluster index
    clusters = {}
    for idx in cluster_ids:
        cluster_idx = communities == idx
        clusters[idx] = cluster_idx

    col_ids = adj_cluster.columns

    for c_idx in adj_cluster.index:
        cluster_i = clusters[c_idx]
        non_connected_clusters = col_ids[adj_cluster.loc[c_idx, :] == 0]
        for nc_idx in non_connected_clusters:
            if nc_idx == c_idx:
                continue
            cluster_nc = clusters[nc_idx]

            # Prune (remove the edges between two non-connected clusters)
            adj_sc.loc[cluster_i, cluster_nc] = 0

    return adj_sc


def connect_graph(adj, data, start_cell_id):
    # TODO: Update the heuristic here which involves using the
    # cell with the max distance to establish a connection with
    # the disconnected parts of the clusters.

    index = adj.index
    dists = pd.Series(dijkstra(adj, indices=start_cell_id), index=index)
    unreachable_nodes = index[dists == np.inf]
    if len(unreachable_nodes) == 0:
        return adj

    # Connect unreachable nodes
    while len(unreachable_nodes) > 0:
        farthest_reachable_id = dists.loc[index[dists != np.inf]].idxmax()
        # Compute distances to unreachable nodes
        unreachable_dists = pairwise_distances(
            data.loc[farthest_reachable_id, :].values.reshape(1, -1),
            data.loc[unreachable_nodes, :])
        unreachable_dists = pd.Series(
            np.ravel(unreachable_dists), index=unreachable_nodes
        )
        # Add edge between farthest reacheable and its nearest unreachable
        adj.loc[farthest_reachable_id, unreachable_dists.idxmin()] = unreachable_dists.min()
        # Recompute distances to early cell
        dists = pd.Series(dijkstra(adj, indices=start_cell_id), index=index)
        # Idenfity unreachable nodes
        unreachable_nodes = index[dists == np.inf]
    return adj


class MetricDataset(Dataset):
    def __init__(
            self,
            data,
            obsm_cluster_key="metric_clusters",
            obsm_data_key="X_pca",
            transform=None,
    ):
        # TODO: Update the argument `obsm_cluster_key` to `obs_cluster_key`
        """Used to sample anchor, positive and negative samples given a cluster assignment over cells.

        Args:
            data ([sc.AnnData]): An AnnData object containing cluster assignments over the data
            obsm_cluster_key (str, optional): Key corresponding to cluster assignments in the AnnData object. Defaults to "metric_clusters".
            obsm_data_key (str, optional): Key corresponding to cell embeddings. Defaults to "X_pca".
            transform ([type], optional): Transform (if any) to be applied to the data before returning. Defaults to None.

        Raises:
            Exception: If `data` argument is not of type sc.AnnData
            Exception: If `obsm_cluster_key` is not present in the data.obs
        """
        if not isinstance(data, sc.AnnData):
            raise Exception(
                f"Expected data to be of type sc.AnnData found : {type(data)}"
            )
        self.data = data
        try:
            self.cluster_inds = data.obs[obsm_cluster_key]
        except KeyError:
            raise Exception(f"`{obsm_cluster_key}` must be set in the data")
        self.X = self.data.obsm[obsm_data_key]
        self.unique_clusters = np.unique(self.cluster_inds)

        self.indices = np.arange(self.data.shape[0])
        self.num_clusters = len(self.unique_clusters)
        self.transform = transform

    def __getitem__(self, idx):
        # Sample the anchor and the positive class
        anchor, pos_class = torch.Tensor(self.X[idx]), self.cluster_inds[idx]
        positive_idx = idx

        while positive_idx == idx:
            positive_idx = np.random.choice(
                self.indices[self.cluster_inds == pos_class]
            )
        pos_sample = self.X[positive_idx]

        # Sample the negative label and sample
        neg_class = np.random.choice(list(set(self.unique_clusters) - set([pos_class])))
        neg_class_choices = self.cluster_inds == neg_class
        neg_sample = self.X[np.random.choice(self.indices[neg_class_choices])]

        if self.transform is not None:
            anchor = self.transform(anchor)
            pos_sample = self.transform(pos_sample)
            neg_sample = self.transform(neg_sample)
        return anchor, pos_sample, neg_sample

    def __len__(self):
        return self.data.shape[0]


class NpDataset(Dataset):
    def __init__(self, X):
        """Creates a torch dataset from  a numpy array

        Args:
            X ([np.ndarray]): Numpy array which needs to be mapped as a tensor
        """
        self.X = torch.Tensor(X)
        self.shape = self.X.shape

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return self.X.shape[0]


class MetricEncoder(nn.Module):
    def __init__(self, infeatures, code_size=10):
        super(MetricEncoder, self).__init__()
        self.infeatures = infeatures
        self.code_size = code_size

        # Encoder architecture
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # Encoder Architecture
        self.fc1 = nn.Linear(self.infeatures, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, self.code_size)
        self.bn3 = nn.BatchNorm1d(self.code_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(self.fc3(x))
        return x


class VAELoss(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, use_bce=True):
        super(VAELoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("Valid values for the reduction param are `mean`, `sum`")
        self.alpha = alpha
        self.use_bce = use_bce
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCELoss(reduction="mean")

    def forward(self, x, decoder_out, mu, logvar):
        # Reconstruction Loss:
        # TODO: Try out the bce loss for reconstruction
        if self.use_bce:
            reconstruction_loss = self.bce(decoder_out, x)
        else:
            reconstruction_loss = self.mse(decoder_out, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + self.alpha * kl_loss
        if self.reduction == "mean":
            return torch.mean(loss)
        else:
            return loss


_SUPPORTED_DEVICES = ["cpu", "cuda"]

import torch.optim as optim


def configure_device(device):
    if device not in _SUPPORTED_DEVICES:
        raise NotImplementedError(f"The device type `{device}` is not supported")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        return "cuda"

    return "cpu"


def get_optimizer(name, net, lr, **kwargs):
    optim_cls = getattr(optim, name, None)
    if optim_cls is None:
        raise ValueError(
            f"""The optimizer {name} is not supported by torch.optim.
            Refer to https://pytorch.org/docs/stable/optim.html#algorithms
            for an overview of the algorithms supported"""
        )
    return optim_cls(
        [{"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": lr}],
        lr=lr,
        **kwargs,
    )


def get_loss(name, **kwargs):
    weight = None
    if name == "mse":
        loss = nn.MSELoss(**kwargs)
    elif name == "vae":
        loss = VAELoss(**kwargs)
    else:
        raise NotImplementedError(f"The loss {name} has not been implemented yet!")
    return loss


def get_lr_scheduler(optimizer, num_epochs, sched_type="poly", **kwargs):
    if sched_type == "poly":
        # A poly learning rate scheduler
        lambda_fn = lambda i: pow((1 - i / num_epochs), 0.9)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
    elif sched_type == "step":
        # A Step learning rate scheduler
        step_size = kwargs["step_size"]
        gamma = kwargs["gamma"]
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif sched_type == "cosine":
        # Cosine learning rate annealing with Warm restarts
        T_0 = kwargs["t0"]
        T_mul = kwargs.get("tmul", 1)
        eta_min = 0
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult=T_mul, eta_min=eta_min
        )
    else:
        raise ValueError(
            f"The lr_scheduler type {sched_type} has not been implemented yet"
        )


class UnsupervisedTrainer:
    def __init__(
            self,
            train_dataset,
            model,
            train_loss,
            val_dataset=None,
            lr_scheduler="poly",
            batch_size=32,
            lr=0.01,
            eval_loss=None,
            log_step=10,
            optimizer="SGD",
            backend="gpu",
            random_state=0,
            optimizer_kwargs={},
            lr_scheduler_kwargs={},
            train_loader_kwargs={},
            val_loader_kwargs={},
            **kwargs,
    ):
        # Create the dataset
        self.lr = lr
        self.random_state = random_state
        self.device = configure_device(backend)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.log_step = log_step
        self.loss_profile = []
        self.batch_size = batch_size
        self.train_loader_kwargs = train_loader_kwargs
        self.val_loader_kwargs = val_loader_kwargs

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.train_loader_kwargs,
        )
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                **self.val_loader_kwargs,
            )
        self.model = model.to(self.device)

        # The parameter train_loss must be a callable
        self.train_criterion = train_loss

        # The parameter eval_loss must be a callable
        self.val_criterion = eval_loss

        self.optimizer = get_optimizer(
            optimizer, self.model, self.lr, **optimizer_kwargs
        )
        self.sched_type = lr_scheduler
        self.sched_kwargs = lr_scheduler_kwargs

        # Some initialization code
        torch.manual_seed(self.random_state)
        torch.set_default_tensor_type("torch.FloatTensor")
        if self.device == "gpu":
            # Set a deterministic CuDNN backend
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self, num_epochs, save_path, restore_path=None):
        self.lr_scheduler = get_lr_scheduler(
            self.optimizer, num_epochs, sched_type=self.sched_type, **self.sched_kwargs
        )
        start_epoch = 0
        if restore_path is not None:
            # Load the model
            self.load(restore_path)

        best_eval = 0.0
        tk0 = tqdm(range(start_epoch, num_epochs))
        for epoch_idx in tk0:
            avg_epoch_loss = self.train_one_epoch()

            # LR scheduler step
            self.lr_scheduler.step()

            # Build loss profile
            self.loss_profile.append(avg_epoch_loss)

            # Evaluate the model
            if self.val_criterion is not None:
                val_eval = self.eval()
                tk0.set_postfix_str(
                    f"Avg Loss for epoch: {avg_epoch_loss} Eval Loss: {val_eval}"
                )
                if epoch_idx == 0:
                    best_eval = val_eval
                    self.save(save_path, epoch_idx, prefix="best")
                else:
                    if best_eval > val_eval:
                        # Save this model checkpoint
                        self.save(save_path, epoch_idx, prefix="best")
                        best_eval = val_eval
            else:
                tk0.set_postfix_str(f"Avg Loss for epoch:{avg_epoch_loss}")
                if epoch_idx % 10 == 0:
                    # Save the model every 10 epochs anyways
                    self.save(save_path, epoch_idx)

    def eval(self):
        raise NotImplementedError()

    def train_one_epoch(self):
        raise NotImplementedError()

    def save(self, path, epoch_id, prefix=""):
        checkpoint_name = f"chkpt_{epoch_id}"
        path = os.path.join(path, prefix)
        checkpoint_path = os.path.join(path, f"{checkpoint_name}.pt")
        state_dict = {}
        model_state = copy.deepcopy(self.model.state_dict())
        model_state = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in model_state.items()
        }
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        for state in optim_state["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

        state_dict["model"] = model_state
        state_dict["optimizer"] = optim_state
        state_dict["scheduler"] = self.lr_scheduler.state_dict()
        state_dict["epoch"] = epoch_id + 1
        state_dict["loss_profile"] = self.loss_profile

        os.makedirs(path, exist_ok=True)
        for f in os.listdir(path):
            if f.endswith(".pt"):
                os.remove(os.path.join(path, f))
        torch.save(state_dict, checkpoint_path)
        del model_state, optim_state
        gc.collect()

    def load(self, load_path):
        state_dict = torch.load(load_path)
        iter_val = state_dict.get("epoch", 0)
        self.loss_profile = state_dict.get("loss_profile", [])
        if "model" in state_dict:
            print("Restoring Model state")
            self.model.load_state_dict(state_dict["model"])

        if "optimizer" in state_dict:
            print("Restoring Optimizer state")
            self.optimizer.load_state_dict(state_dict["optimizer"])
            # manually move the optimizer state vectors to device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if "scheduler" in state_dict:
            print("Restoring Learning Rate scheduler state")
            self.lr_scheduler.load_state_dict(state_dict["scheduler"])

    def update_dataset(self, dataset):
        self.train_dataset = dataset
        # Update the training loader with the new dataset
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.train_loader_kwargs,
        )


class MetricTrainer(UnsupervisedTrainer):
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = self.train_loader
        for idx, (anchor, pos, neg) in enumerate(tk0):
            self.optimizer.zero_grad()
            anchor = anchor.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)
            X_anchor = self.model(anchor.float())
            X_pos = self.model(pos.float())
            X_neg = self.model(neg.float())
            loss = self.train_criterion(X_anchor, X_pos, X_neg)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)


def train_metric_learner(
        adata,
        n_episodes=10,
        n_metric_epochs=10,
        code_size=10,
        obsm_data_key="X_pca",
        device="cuda",
        random_state=0,
        save_path=os.getcwd(),
        backend="kmeans",
        nn_kwargs={},
        trainer_kwargs={},
        cluster_kwargs={},
        loss_kwargs={},
):
    X = adata.obsm[obsm_data_key]
    clustering_scores = []
    cluster_record = []

    # Generate initial clusters
    print("Generating initial clusters")
    communities, score = determine_cell_clusters(
        adata,
        obsm_key=obsm_data_key,
        backend=backend,
        cluster_key="metric_clusters",
        nn_kwargs=nn_kwargs,
        **cluster_kwargs,
    )
    clustering_scores.append(score)

    # Dataset
    dataset = MetricDataset(
        adata, obsm_data_key=obsm_data_key, obsm_cluster_key="metric_clusters"
    )
    cluster_record.append(dataset.num_clusters)

    # Train Loss
    train_loss = nn.TripletMarginLoss(**loss_kwargs)

    # Model
    infeatures = X.shape[-1]
    model = MetricEncoder(infeatures, code_size=code_size).to(device)

    # Trainer
    trainer = MetricTrainer(
        dataset,
        model,
        train_loss,
        random_state=random_state,
        backend=device,
        **trainer_kwargs,
    )
    if n_episodes == 0:
        adata.obsm["metric_embedding"] = adata.obsm[obsm_data_key]

    for episode_idx in range(n_episodes):
        print(f"Training for episode: {episode_idx + 1}")
        epoch_start_time = time.time()
        trainer.train(n_metric_epochs, save_path)

        # Generate embeddings
        embedding = []
        embedding_dataset = NpDataset(X)

        model.eval()
        with torch.no_grad():
            for data in tqdm(embedding_dataset):
                data = data.to(device)
                embedding.append(model(data.unsqueeze(0)).squeeze().cpu().numpy())
        X_embedding = np.array(embedding)

        adata.obsm["metric_embedding"] = X_embedding

        # Generate new cluster assignments using the obtained embedding\
        print(f"Re-generating clusters for episode: {episode_idx + 1}")
        communities, score = determine_cell_clusters(
            adata,
            obsm_key="metric_embedding",
            backend=backend,
            cluster_key="metric_clusters",
            nn_kwargs=nn_kwargs,
            **cluster_kwargs,
        )
        clustering_scores.append(score)

        # Update the dataset as the cluster assignments have changed
        dataset = MetricDataset(
            adata, obsm_data_key=obsm_data_key, obsm_cluster_key="metric_clusters"
        )
        cluster_record.append(dataset.num_clusters)
        trainer.update_dataset(dataset)
        print(f"Time Elapsed for epoch: {time.time() - epoch_start_time}s")

    # Add the modularity score estimates to the adata
    adata.uns["metric_clustering_scores"] = clustering_scores
    adata.uns["metric_n_cluster_records"] = cluster_record


@compute_runtime
def compute_undirected_cluster_connectivity(
        communities, adj, z_threshold=1.0, conn_threshold=None
):
    N = communities.shape[0]
    n_communities = np.unique(communities).shape[0]

    # Create cluster index
    clusters = {}
    for idx in np.unique(communities):
        cluster_idx = communities == idx
        clusters[idx] = cluster_idx

    undirected_cluster_connectivity = pd.DataFrame(
        np.zeros((n_communities, n_communities)),
        index=np.unique(communities),
        columns=np.unique(communities),
    )
    undirected_z_score = pd.DataFrame(
        np.zeros((n_communities, n_communities)),
        index=np.unique(communities),
        columns=np.unique(communities),
    )
    cluster_outgoing_edges = {}
    for i in np.unique(communities):
        cluster_i = clusters[i]

        # Compute the outgoing edges from the ith cluster
        adj_i = adj[cluster_i, :]
        adj_ii = adj_i[:, cluster_i]
        e_i = np.sum(adj_i) - np.sum(adj_ii)
        n_i = np.sum(cluster_i)
        cluster_outgoing_edges[i] = e_i

        for j in np.unique(communities):
            if i == j:
                continue
            # Compute the outgoing edges from the jth cluster
            cluster_j = clusters[j]
            adj_j = adj[cluster_j, :]
            adj_jj = adj_j[:, cluster_j]
            e_j = np.sum(adj_j) - np.sum(adj_jj)
            n_j = np.sum(cluster_j)

            # Compute the number of inter-edges from the ith to jth cluster
            adj_ij = adj_i[:, cluster_j]
            e_ij = np.sum(adj_ij)

            # Compute the number of inter-edges from the jth to ith cluster
            adj_ji = adj_j[:, cluster_i]
            e_ji = np.sum(adj_ji)
            e_sym = e_ij + e_ji

            # Compute the random assignment of edges from the ith to the jth
            # cluster under the PAGA binomial model
            e_sym_random = (e_i * n_j + e_j * n_i) / (N - 1)

            # Compute the cluster connectivity measure
            std_sym = (e_i * n_j * (N - n_j - 1) + e_j * n_i * (N - n_i - 1)) / (
                    N - 1
            ) ** 2
            undirected_z_score.loc[i, j] = (e_sym - e_sym_random) / std_sym

            # Only add non-spurious edges based on a threshold
            undirected_cluster_connectivity.loc[i, j] = (e_sym - e_sym_random) / (
                    e_i + e_j - e_sym_random
            )
            if conn_threshold is not None:
                if undirected_cluster_connectivity.loc[i, j] < conn_threshold:
                    undirected_cluster_connectivity.loc[i, j] = 0
            elif undirected_z_score.loc[i, j] < z_threshold:
                undirected_cluster_connectivity.loc[i, j] = 0
    return undirected_cluster_connectivity, undirected_z_score


def plot_embeddings(
        X,
        figsize=(12, 8),
        save_path=None,
        title=None,
        show_legend=False,
        show_colorbar=False,
        axis_off=True,
        hover_labels=None,
        labels=None,
        ax=None,
        legend_kwargs={},
        cb_axes_pos=None,
        cb_kwargs={},
        save_kwargs={},
        picker=False,
        **kwargs,
):
    def annotate(axis, text, x, y):
        text_annotation = Annotation(text, xy=(x, y), xycoords="data")
        axis.add_artist(text_annotation)

    def onpick(event):
        ind = event.ind

        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata

        # Take only the first of many indices returned
        label = X[ind[0], :]
        if hover_labels is not None:
            label = hover_labels[ind[0]]

        # Create Text annotation
        annotate(ax, label, label_pos_x, label_pos_y)

        # Redraw the figure
        ax.figure.canvas.draw_idle()

    assert X.shape[-1] == 2

    # Set figsize
    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    # ax = plt.gca()

    # Set title (if set)
    if title is not None:
        plt.title(title)

    # Plot
    scatter = ax.scatter(X[:, 0], X[:, 1], picker=picker, **kwargs)

    if show_legend:
        if labels is None:
            raise ValueError("labels must be provided when plotting legend")
        # Create legend
        legend = ax.legend(*scatter.legend_elements(num=len(labels)), **legend_kwargs)
        # Replace default labels with the provided labels
        text = legend.get_texts()
        assert len(text) == len(labels)

        for t, label in zip(text, labels):
            t.set_text(label)
        ax.add_artist(legend)

    if axis_off:
        ax.set_axis_off()
    if show_colorbar:
        cax = None
        if cb_axes_pos is not None:
            cax = fig.add_axes(cb_axes_pos)
        plt.colorbar(scatter, cax=cax, **cb_kwargs)

    # Pick Event handling (useful for selecting start cells)
    if picker is True:
        fig.canvas.mpl_connect("pick_event", onpick)

    # Save
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    # plt.show()


def plot_connectivity_graph(
        embeddings,
        communities,
        cluster_connectivities,
        start_cell_ids=None,
        mode="undirected",
        cmap="YlGn",
        figsize=(12, 12),
        node_size=800,
        labels=None,
        font_color="black",
        start_node_color=None,
        node_color=None,
        title=None,
        offset=0,
        **kwargs,
):
    g, node_positions = compute_connectivity_graph(
        embeddings, communities, cluster_connectivities, mode=mode
    )

    start_cluster_ids = list(set([communities[id] for id in start_cell_ids]))
    colors = np.unique(communities)
    if node_color is not None:
        colors = []
        for c_id in np.unique(communities):
            if c_id in start_cluster_ids and start_node_color is not None:
                colors.append(start_node_color)
            else:
                colors.append(node_color)
    # Draw the graph
    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        plt.title(title)
    ax.axis("off")
    edge_weights = [offset + w for _, _, w in g.edges.data("weight")]
    nx.draw_networkx(
        g,
        pos=node_positions,
        cmap=cmap,
        labels=labels,
        node_color=colors,
        font_color=font_color,
        node_size=node_size,
        width=edge_weights,
        **kwargs,
    )
    return ax


@compute_runtime
def compute_connectivity_graph(
        embeddings, communities, cluster_connectivities, mode="undirected"
):
    assert mode in ["directed", "undirected"]
    g = nx.Graph() if mode == "undirected" else nx.DiGraph()
    node_positions = {}
    cluster_ids = np.unique(communities)
    for i in cluster_ids:
        g.add_node(i)
        # determine the node pos for the cluster
        cluster_i = communities == i
        node_pos = np.mean(embeddings[cluster_i, :], axis=0)
        node_positions[i] = node_pos

    n_nodes = len(cluster_ids)
    for row_id, i in enumerate(cluster_ids):
        for col_id, j in enumerate(cluster_ids):
            if cluster_connectivities.loc[i, j] > 0:
                g.add_edge(
                    cluster_ids[row_id],
                    cluster_ids[col_id],
                    weight=cluster_connectivities.loc[i, j],
                )
    return g, node_positions


def compute_pseudotime(
        ad,
        start_cell_ids,
        adj_dist,
        adj_cluster,
        comm_key="metric_clusters",
        data_key="metric_embedding",
):
    communities = ad.obs[comm_key]
    cluster_ids = np.unique(communities)
    data = pd.DataFrame(ad.obsm[data_key], index=ad.obs_names)
    # Create cluster index
    clusters = {}
    for idx in cluster_ids:
        cluster_idx = communities == idx
        clusters[idx] = cluster_idx
    # Prune the initial adjacency matrix
    adj_dist = pd.DataFrame(
        adj_dist.todense(), index=ad.obs_names, columns=ad.obs_names
    )
    adj_dist_pruned = prune_network_edges(communities, adj_dist, adj_cluster)
    # Pseudotime computation on the pruned graph
    if type(start_cell_ids) is str:
        start_cell_ids = [start_cell_ids]
    p = dijkstra(adj_dist_pruned.to_numpy(), indices=start_cell_ids, min_only=True)
    pseudotime = pd.Series(p, index=ad.obs_names)

    for _, cluster in clusters.items():
        p_cluster = pseudotime.loc[cluster]
        cluster_start_cell = p_cluster.idxmin()
        adj_sc = adj_dist_pruned.loc[cluster, cluster]
        adj_sc = connect_graph(
            adj_sc,
            data.loc[cluster, :],
            np.where(adj_sc.index == cluster_start_cell)[0][0],
        )

        # Update the cluster graph with
        adj_dist_pruned.loc[cluster, cluster] = adj_sc

    # Recompute the pseudotime with the updated graph
    p = dijkstra(adj_dist_pruned, indices=start_cell_ids, min_only=True)
    pseudotime = pd.Series(p, index=ad.obs_names)

    # Set the pseudotime for unreachable cells to 0
    pseudotime[pseudotime == np.inf] = 0

    # Add pseudotime to annotated data object
    ad.obs["metric_pseudotime"] = pseudotime
    return ad


def plot_pseudotime(
        adata,
        embedding_key="X_met_embedding",
        pseudotime_key="metric_pseudotime_v2",
        cmap=None,
        figsize=None,
        ax=None,
        cb_axes_pos=None,
        save_path=None,
        save_kwargs={},
        cb_kwargs={},
        **kwargs,
):
    # An alias to plotting embeddings with pseudotime projected on it
    pseudotime = adata.obs[pseudotime_key]
    X_embedded = adata.obsm[embedding_key]
    # Plot
    plot_embeddings(
        X_embedded,
        c=pseudotime,
        cmap=cmap,
        ax=ax,
        figsize=figsize,
        show_colorbar=True,
        cb_axes_pos=cb_axes_pos,
        save_path=save_path,
        save_kwargs=save_kwargs,
        cb_kwargs=cb_kwargs,
        **kwargs,
    )


@compute_runtime
def generate_plot_embeddings(X, method="tsne", **kwargs):
    if method == "tsne":
        tsne = TSNE(n_components=2, **kwargs)
        X_tsne = tsne.fit_transform(X)
        return X_tsne
    elif method == "umap":
        u = umap.UMAP(n_components=2, **kwargs)
        X_umap = u.fit_transform(X)
        return X_umap
    else:
        raise ValueError(f"Unsupported embedding method type: {method}")
