library(SpatialPCA)
library(ggplot2)
library(slingshot)
library(reticulate)
set.seed(1234)

run_spatialPCA <- function(counts_, xy_coord, n_class)
{
  LIBD <- CreateSpatialPCAObject(counts = counts_, location = xy_coord, project = "SpatialPCA", gene.type = "spatial", sparkversion = "sparkx", customGenelist = NULL, min.loctions = 0, min.features = 0)
  LIBD <- SpatialPCA_buildKernel(LIBD, kerneltype = "gaussian", bandwidthtype = "SJ", bandwidth.set.by.user = NULL)
  LIBD <- SpatialPCA_EstimateLoading(LIBD, fast = FALSE, SpatialPCnum = 20)
  LIBD <- SpatialPCA_SpatialPCs(LIBD, fast = FALSE)
  embedding <- LIBD@SpatialPCs
  clusterlabel <- walktrap_clustering(clusternum = n_class, latent_dat = LIBD@SpatialPCs, knearest = 10)
  clusterlabel_refine <- refine_cluster_10x(clusterlabels = clusterlabel, location = LIBD@location, shape = "hexagon")

  p_UMAP <- plot_RGB_UMAP(LIBD@location, LIBD@SpatialPCs, pointsize = 2, textsize = 15) # nearer than clolor more similar
  p_UMAP$figure
  sim <- SingleCellExperiment(assays = counts_)
  reducedDims(sim) <- SimpleList(DRM = t(LIBD@SpatialPCs))
  colData(sim)$clusterlabel <- factor(clusterlabel_refine)

  sim <- slingshot(sim, clusterLabels = 'clusterlabel', reducedDim = 'DRM', start.clus = "4")
  summary(sim@colData@listData)
  pseudotime_traj1 <- sim@colData@listData$slingPseudotime_1 # in this data only one trajectory was inferred
  return(list(pseudotime_traj1 = pseudotime_traj1, LIBD = LIBD, clusterlabel_refine = clusterlabel_refine))
}

load_data <- function(args)
{
  adata <- py_anndata$read_h5ad(args$data_path)
  truth <- as.vector(as.matrix(np$array(adata$obs$cluster$values)))
  n_class <- length(unique(truth))
  xy_coord <- as.matrix(adata$obsm['spatial'])
  # xy_coord <- data.frame(
  #   x_coord = xy_coord[, 1],
  #   y_coord = -1 * xy_coord[, 2])
  xy_coord <- data.frame(
    x_coord = xy_coord[, 2],
    y_coord = -1 * xy_coord[, 1])
  # check the datatype
  # if (typeof(adata$X) == "environment")
  # {
  #   # counts_ <- t(as.matrix(adata$X$toarray()))
  #   counts_ <- t(as.matrix(adata$X))
  # } else
  # {
  #   counts_ <- t(as.matrix(adata$X))
  # }
  counts_ <- tryCatch({
    t(as.matrix(adata$X$A))
  }, error = function(e) {
    t(as.matrix(adata$X))
  })

  counts_ <- as(as.matrix(counts_), "dgCMatrix")
  print(dim(counts_)) # The count matrix
  print(dim(xy_coord)) # The x and y coordinates. We flipped the y axis for visualization.
  xy_coord <- as.matrix(xy_coord)
  colnames(counts_) <- as.list(adata$obs$index$values)
  rownames(counts_) <- as.list(adata$var$index$values)
  rownames(xy_coord) <- colnames(counts_)
  return(list(counts_ = counts_, xy_coord = xy_coord, n_class = n_class))
}

plot_result <- function(pseudotime_traj1, args, color_in, clusterlabel_refine, res_)
{
  gridnum <- res_$n_class
  color_in <- assign_colors()
  p_traj1 <- plot_trajectory(pseudotime_traj1, res_$xy_coord, clusterlabel_refine, gridnum, color_in, pointsize = 1, arrowlength = 0.4, arrowsize = 0.6, textsize = 15)
  plot <- p_traj1$Arrowoverlay1
  # plot <- p_traj1$Pseudotime
  png(args$img_path, width = 6, height = 6, units = "in", res = 300)
  print(plot)
  dev.off()
}

assign_colors <- function()
{
  # cbp <- c("#9C9EDE", "#5CB85C", "#E377C2", "#4DBBD5", "#FED439", "#FF9896", "#FFDC91")
  cbp <- c('#fb9a99', '#1f78b4', '#33a02c', '#b2df8a', '#e31a1c', '#a6cee3', "#FF9896", "black")
  #cbp <- c("#9C9EDE", "#5CB85C", "#E377C2", "#4DBBD5", "#FED439", "#FF9896", "#FFDC91")
  # cbp <- c("#5CB85C", "#9C9EDE", "#FFDC91", "#4DBBD5", "#FF9896", "#FED439", "#E377C2", "#FED439")
  return(cbp)
}

gene_order <- function(res, pca_res)
{
  # only for Zhuang-ABCA-2.005
  source_python("../analysis/Fig2_gene_order.py")
  exprs <- as.matrix(res$counts_)
  pseudotime <- round(pca_res$pseudotime_traj1 / 100, 2)
  exprs <- rbind(exprs, pseudotime)
  exprs <- t(exprs)
  expr <- pd$DataFrame(exprs)
  adata <- py_anndata$read_h5ad(args$data_path)
  expr$columns <- append(as.array(adata$var$gene_symbol$values), "pseudotime")
  expr$index <- colnames(res$counts_)
  plot_gene_expression_trajectory(expr)
}

use_python("C:/Users/yingyinyu3/AppData/Local/anaconda3/")
py_anndata <- import("anndata", convert = FALSE)
source_python("utils/Metrics.py")
np <- import("numpy", convert = FALSE)
pd <- import("pandas", convert = FALSE)
# args <- list(data_path = 'dataset/stdata/BZ5/data.h5ad', img_path = "img/BZ5/spatialpca.png", out_path = "result/BZ5", num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/BZ9/data.h5ad', img_path = "img/BZ9/spatialpca.png", out_path = "result/BZ9",num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/BZ14/data.h5ad', img_path = "img/BZ14/spatialpca.png", out_path = "result/BZ14", num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/starmap/data.h5ad', img_path = "img/starmap/spatialpca.png", out_path = "result/starmap", num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/HER2ST/data.h5ad', img_path = "img/HER2ST/spatialpca.png", out_path = "result/HER2ST", num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/Zhuang-ABCA-2.005/data.h5ad', img_path = "img/Zhuang-ABCA-2.005/spatialpca.png", out_path = "result/Zhuang-ABCA-2.005",num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/Zhuang-ABCA-2.053/data.h5ad', img_path = "img/Zhuang-ABCA-2.053/spatialpca.png", out_path = "result/Zhuang-ABCA-2.053", num_dim = 30, k = 30)
# args <- list(data_path = 'dataset/stdata/OSCC1/data.h5ad', img_path = "img/OSCC1/spatialpca.png", out_path = "result/OSCC1", num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/OSCC2/data.h5ad', img_path = "img/OSCC2/spatialpca.png", out_path = "result/OSCC2",num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/OSCC3/data.h5ad', img_path = "img/OSCC3/spatialpca.png", out_path = "result/OSCC3",num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/OSCC5/data.h5ad', img_path = "img/OSCC5/spatialpca.png", out_path = "result/OSCC5",num_dim = 30, k = 10)
# args <- list(data_path = 'dataset/stdata/OSCC7/data.h5ad', img_path = "img/OSCC7/spatialpca.png", out_path = "result/OSCC7",num_dim = 30, k = 10)
args <- list(data_path = 'dataset/stdata/OSCC8/data.h5ad', img_path = "img/OSCC8/spatialpca.png", out_path = "result/OSCC8", num_dim = 30, k = 10)
res <- load_data(args)
pca_res <- run_spatialPCA(res$counts_, res$xy_coord, res$n_class)
adata <- py_anndata$read_h5ad(args$data_path)
res <- caculate_R_cluster_metric(as.vector(pca_res$clusterlabel_refine), adata)
ARI <- res[1]
AMI <- res[2]
formatted_string <- sprintf("spatialPCA_meanARI%.5f_stdARI%.5f_meanAMI%.5f_stdAMI%.5f", ARI, 0, AMI, 0)
metric_path <- paste(args$out_path, formatted_string, sep = "/", collapse = "")
print(metric_path)
writeLines(formatted_string, metric_path)
# plot_result(pca_res$pseudotime_traj1, args, assign_colors(), pca_res$clusterlabel_refine, res)



