library(monocle3)
library(Seurat)
library(SeuratWrappers)
library(reticulate)
library(dplyr)
library(igraph)
library(optparse)
library(ggplot2)
use_condaenv("r-reticulate")

run_leiden <- function(cds, n_cluster, reduction_method = "UMAP", k = 10) {
  this_min <- 0
  this_max <- 3
  this_step <- 0
  max_steps <- 30
  while (this_step < max_steps) {
    this_resolution <- this_min + ((this_max - this_min) / 2)
    cds <- cluster_cells(cds = cds, k = k, reduction_method = reduction_method, verbose = FALSE, cluster_method = "leiden", resolution = this_resolution)
    partition_list <- `@`(cds, clusters)[[reduction_method]]$clusters
    this_clusters <- length(unique(as.integer(as.numeric(partition_list))))
    message(sprintf(" %d clusters , error %.3f ", this_clusters, (this_clusters - n_cluster)))
    if ((this_clusters - n_cluster) > 0) {
      this_max <- this_resolution
    } else if ((this_clusters - n_cluster) < 0) {
      this_min <- this_resolution
    } else {
      message(sprintf("Succeed to find %d clusters at resolution %.3f", this_clusters, this_resolution))
      return(cds)
    }
    this_step <- this_step + 1
  }
  partition_list <- `@`(cds, clusters)[[reduction_method]]$partitions
  this_clusters <- length(unique(as.integer(as.numeric(partition_list))))
  message(sprintf("Find %d clusters at resolution %.3f", this_clusters, this_resolution))
  return(cds)
}

generate_adj_matrix <- function(cds, reduction_method = "UMAP") {
  ica_space_df <- t(`@`(cds, principal_graph_aux)[[reduction_method ]]$dp_mst) %>%
    as.data.frame() %>%
    dplyr::select(prin_graph_dim_1 = 1, prin_graph_dim_2 = 2) %>%
    dplyr::mutate(sample_name = rownames(.), sample_state = rownames(.))
  dp_mst <- `@`(cds, principal_graph)[[reduction_method ]]
  edge_df <- dp_mst %>%
    igraph::as_data_frame() %>%
    dplyr::select(source = "from", target = "to") %>%
    dplyr::left_join(ica_space_df %>% dplyr::select(source = "sample_name", source_prin_graph_dim_1 = "prin_graph_dim_1", source_prin_graph_dim_2 = "prin_graph_dim_2"), by = "source") %>%
    dplyr::left_join(ica_space_df %>% dplyr::select(target = "sample_name", target_prin_graph_dim_1 = "prin_graph_dim_1", target_prin_graph_dim_2 = "prin_graph_dim_2"), by = "target")
  vertex_names <- unique(c(edge_df$source, edge_df$target))
  graph <- make_empty_graph(n = length(vertex_names), directed = FALSE)
  graph <- add_edges(graph, cbind(match(edge_df$source, vertex_names), match(edge_df$target, vertex_names)))
  adj_matrix <- get.adjacency(graph, sparse = FALSE)
  return(adj_matrix)
}


run_monocle3_func <- function(args)
{
  adata <- py_anndata$read_h5ad(args$data_path)
  n_classes <- length(unique.matrix(as.matrix(adata$
                                                uns$
                                                milestone_percentages$
                                                milestone_id$
                                                values)))
  counts <- t(as.matrix(adata$X))
  # counts <- t(as.matrix(adata$X$A))
  cell_info <- as.matrix(adata$obs$index$values)
  gene_info <- as.matrix(adata$var$index$values)

  data <- CreateSeuratObject(
    counts = counts,
    assay = "RNA")

  data.cds <- as.cell_data_set(data)
  data.cds@rowRanges@elementMetadata@listData[["gene_short_name"]] <- gene_info
  colnames(data.cds) <- cell_info
  data.cds <- data.cds[, Matrix::colSums(exprs(data.cds)) != 0]
  data.cds <- estimate_size_factors(data.cds)
  data.cds <- preprocess_cds(data.cds, num_dim = args$num_dim, verbose = TRUE)
  data.cds <- reduce_dimension(data.cds, reduction_method = "UMAP", preprocess_method = "PCA")
  data.cds <- run_leiden(data.cds, n_classes, k = args$k)
  data.cds <- learn_graph(data.cds, use_partition = FALSE, verbose = FALSE)
  data.cds <- order_cells(data.cds, root_cells = as.character(adata$uns$start_id))
  metrics_pseudotime <- `@`(data.cds, principal_graph_aux)[["UMAP"]]$pseudotime
  metrics_cluster <- generate_adj_matrix(data.cds, reduction_method = "UMAP")
  adata$obs <- data.frame(pseudotime = metrics_pseudotime)
  adata$uns$metric_connectivities <- metrics_cluster
  res <- caculate_R_metric(adata)
  IM <- res[1]
  OT <- res[2]
  KT <- res[3]
  SR <- res[4]
  formatted_string <- sprintf("monocle3_meanIM%.5f_std%.5f_meanOT%.5f_std%.5f_meanKT%.5f_std%.5f_meanSR%.5f_std%.5f", IM, 0, OT, 0, KT, 0, SR, 0)
  formatted_string <- paste(args$dataname, formatted_string, sep = "_")
  metric_path <- paste(args.out_path, formatted_string, sep = "/", collapse = "")
  print(metric_path)
  writeLines(formatted_string, metric_path)
  plot <- plot_cells(data.cds, color_cells_by = "pseudotime", label_cell_groups = FALSE, graph_label_size = 3, cell_size = 1, label_leaves = FALSE, label_branch_points = FALSE)
  # png(args$img_path, width = 6, height = 6, units = "wswin", res = 300)
  ggsave(filename = paste(args$dataname, "monocle3.png", sep = "_"), path = args$img_path, plot, width = 6, height = 6, dpi = 300)
}

option_list <- list(
  make_option(c("-f", "--dataname"), type = "character", default = "real1"),
  make_option(c("-k", "--k"), type = "integer", default = 20),
  make_option(c("-d", "--num_dim"), type = "integer", default = 20));

opt_parser <- OptionParser(option_list = option_list);
opt <- parse_args(opt_parser);
opt$data_path <- paste("dataset/scdata/", opt$dataname, "/data.h5ad", sep = "")
opt$img_path <- paste("img/scimg/", "", sep = "")
args.out_path <- paste("result/", "", sep = "")
py_anndata <- import("anndata", convert = FALSE)
np <- import("numpy", convert = FALSE)
source_python("utils/Metrics.py")
start_time <- Sys.time() # 记录初始时间
run_monocle3_func(opt)
end_time <- Sys.time() # 记录结束时间
print(end_time - start_time) # 打印时间差