<div style="display: flex; width: 100%;">
  <div style="width: 25%; padding: 10px;">
    <img src="doc/logo.png" alt="Model" style="width: 100%; height: auto; border-radius:40%;" />
  </div>
  <div style="width: 90%; padding: 10px;">
    <h1>inferring Causal trAjectories from SpAtial Transcriptomics data Using CASCAT</h1>
  </div>
</div>

**CASCAT** is a **tree-shaped structural causal model** with the local Markov property between clusters and conditional independences to infer a unique cell
differentiation trajectory, overcoming Markov equivalence in high-dimensional, non-linear data.
**CASCAT** **eliminates redundant links** between spatially close but independent cells,
creating a causal cell graph that enhances the accuracy of existing spatial clustering algorithms. 


<br/>
<p align="center">
    <img width="85%" src="doc/model.png" alt="Model">
</p>
<br/>

## Installation & Setup

This step can be finished within a few minutes.

1. Install [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) if not already available.
2. Create a new cascat environment, activate it, and install the basic packages.

```bash
conda create -n cascat python==3.10 -y 
conda activate cascat
```

3. Install PyTorch and PyG.
   To select the appropriate versions, you may refer to the official websites of
   [PyTorch](https://pytorch.org/get-started/previous-versions/) and
   [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
   The following commands are for CUDA 11.8.

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install scanpy matplotlib networkx scikit-misc numpy pydot numba==0.57.1 PyYAML numba-scipy
```

4. (optinal) Install R to generate simulated data.

```bash
conda create -n r_env r-essentials r-base -y; 
conda activate r_env
conda install r-mclust
export R_HOME='/home/yourname/miniconda3/envs/r_env/lib/R'
export rScript = '/home/yourname/miniconda3/envs/r_env/bin/Rscript'
```

## Dataset

We provide a simulated dataset **linear1** under ./data/linear1 as an example dataset. All the ten simulated datasets used in the paper can be accessed from [Google drive](https://drive.google.com/drive/folders/1Ycm_e7EtX07cjuw0a5vbCs_dIswT-n7n?usp=sharing).

## Run CASCAT

CASCAT takes a standard AnnData (adata) object as input. 
The observations `obs` are cells/spots and variables `var` are genes. 

If true cluster labels are stored in `adata.obs['cluster']`, set `verbose=True` when training the cluster GSL model.

To run CASCAT get **cluster** result, you can execute following code:

`python main.py --yml_path ./config/linear1.yml --mode train --verbose True`

If true trajectory labels are stored in `adata.obs['cluster']`, set `verbose=True` when inferring.

To run CASCAT get **trajectory** result, you can execute following code:

`python main.py --yml_path ./config/linear1.yml --mode infer --verbose True`


The output of CASCAT is a new adata object, with the following information stored
within it:

- `adata.obs['cascat_clusters']` The predicted cluster labels.
- `adata.obsm['cascat_embedding']` The generated low-dimensional cell embeddings.
- `adata.uns['cascat_connectivities']` The inferred trajecory topology connectivities.
- `adata.uns['CMI']` The inferred conditional mutual information matrix for each cluster.



### CASCAT Parameters

All the parameters are stored in the yaml file. 
You can modify the parameters in the **yaml** file in ./config to fit your data.

#### Specify parameters in the yaml file

- `n_clusters` The number of clusters.
- `learned_graph` CMI or GL to learn the graph.
- `hvg` Whether to use highly variable genes.
- `percent` The percentage of CMI filter.
- `output_dir` The directory to store the output.
- `clu_dir` The directory to store cluster result.
- `CMI_dir` The directory to store the CMI file.
- `root` The root of the trajectory.

#### Default parameters

- `clu_model` The clustering model, including KMeans, Louvain, Leiden, and Spectral.
- `threshold` The number of neighbors.
- `k` The number of neighbors.
- `temperature` The number of PCs.