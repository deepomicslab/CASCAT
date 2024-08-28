import networkx as nx
import pandas as pd
import itertools
import os
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from InformationMeasure.KDEMeasures import *
from InformationMeasure.Measures import *


class BinaryCMITree:

    def __init__(self, adata, root, predict_key, threshold=0.095, save_dir=None, kde=True):
        self.tree = nx.DiGraph()
        self.root = root
        self.save_dir = save_dir
        self.theshold = threshold
        self.predict_key = predict_key
        self.adata = adata
        self.kde = kde
        self.debug = False

    def read_express(self, cluster_node=None):
        if not isinstance(self.adata.X, np.ndarray):
            express = pd.DataFrame(self.adata.X.toarray(), index=self.adata.obs.index, columns=self.adata.var.index,
                                   dtype=np.float64)
        else:
            express = pd.DataFrame(self.adata.X, index=self.adata.obs.index, columns=self.adata.var.index,
                                   dtype=np.float64)
        express = pd.concat([express, self.adata.obs[self.predict_key]], axis=1)
        express = express.groupby(self.predict_key).mean()
        express[self.predict_key] = express.index
        express = express.drop(self.predict_key, axis=1)
        express.columns = express.columns.astype(str)
        express.index = express.index.astype(str)
        if cluster_node != None:
            express = express.loc[cluster_node]
        else:
            cluster_node = list(set(express.index.values.tolist()))
        return express, cluster_node

    def update_tree(self, nodes):
        for node in nodes:
            self.tree.nodes[node]['CMI'] = self.adata.uns["CMI"][node]
            CMInodes = [n for n in nodes if n != node]
            candidate_CMI = self.tree.nodes[node]['CMI'].loc[CMInodes, CMInodes].copy()
            self.generate_attr(node, candidate_CMI)

    def init_tree(self, cluster_list=None, debug=False, debug_nodes=None):
        self.debug = debug
        express, nodes = self.read_express(cluster_list)
        print(f"tree nodes: {nodes}")
        self.tree.add_nodes_from(nodes)
        for node in nodes:
            self.tree.nodes[node]['express'] = express.loc[node].values
            self.tree.nodes[node]['fix'] = 0
            self.tree.add_edge(self.root, node)
        if (self.root, self.root) in self.tree.edges:
            self.tree.remove_edge(self.root, self.root)
        self.tree.nodes[self.root]['fix'] = 1
        self.adata.uns["CMI"] = {node: self.cal_CMI_matrix(node) for node in nodes}
        self.update_tree(nodes)
        if self.debug:
            if debug_nodes is not None:
                error_nodes = debug_nodes
            else:
                error_nodes = [node for node in nodes if self.tree.nodes[node]['compNum'] > 3]
            for node in error_nodes:
                self.tree.nodes[node]['fix'] = 2
            print(f"update tree attrs, with errors: {len(error_nodes)}")
            nodes = [node for node in nodes if self.tree.nodes[node]['fix'] != 2]
            self.update_tree(nodes)
        self.get_unfix_nodes_express()

    def construct_tree(self, root):
        self.find_sub_root([root])
        while len(self.unfix_nodes) > 0:
            dict = self.get_next_dict()
            print("unfix nodes group:", {i: list(dict[i]) for i in range(len(dict))})
            for i in dict:
                self.find_sub_root(i)
        error_nodes = [node for node in self.tree.nodes if self.tree.nodes[node]['fix'] == 2]
        if self.debug:
            for node in error_nodes:
                print(f"fix error node: {node}")
                CMI = self.tree.nodes[root]['CMI']
                CMI = CMI.loc[node, [i for i in CMI.columns if i in list(self.tree.nodes) and i not in error_nodes]]
                nearest = CMI.idxmax()
                self.tree.add_edge(nearest, node)
                self.tree.remove_edge(root, node)
                self.tree.nodes[node]['fix'] = 1
        for edge in self.tree.edges:
            self.tree.edges[edge]['weight'] = 1

    def transCopm2list(self, comps_arr):
        compDict = {key: [] for key in list(set(comps_arr[0]))}
        for i in range(len(comps_arr[0])):
            compDict[comps_arr[0][i]].append(comps_arr[1][i])
        return compDict

    def generate_attr(self, node, CMI):
        n_components, labels = self.find_matrix_connected_components(self.binary_CMI_matrix(CMI))
        print("Comp Num:", n_components, labels, CMI.columns.values, "based on node:", node)
        if node == self.root:
            dis2root = 0
        else:
            dis2root = self.items_in_root_cluster(labels, CMI.columns.values)
        self.tree.nodes[node]['compNum'] = n_components
        self.tree.nodes[node]['comps'] = (labels, CMI.columns.values)
        self.tree.nodes[node]['compDict'] = self.transCopm2list((labels, CMI.columns.values))
        self.tree.nodes[node]['dis2Root'] = dis2root

    def get_unfix_nodes_express(self):
        unfix_nodes = [node for node in self.tree.nodes if self.tree.nodes[node]['fix'] == 0]
        self.unfix_nodes = unfix_nodes
        print(f"refresh tree and unfix nodes: {unfix_nodes}")
        return unfix_nodes

    def binary_CMI_matrix(self, CMI_array, k=2):
        for i in range(len(CMI_array)):
            CMI_array.iloc[i, :] = [x if x in CMI_array.iloc[i, :].nlargest(k).values else 0 for x in
                                    CMI_array.iloc[i, :]]
        CMI_array = CMI_array[CMI_array > self.theshold].fillna(0)
        np.fill_diagonal(CMI_array.values, 0)
        CMI_array[CMI_array != 0] = 1
        return CMI_array

    def cal_CMI_matrix(self, root):
        CMI_path = self.save_dir + "/" + str(root) + '_CMI.csv'
        print(f"calculate CMI matrix based on root: {root} on {CMI_path}")
        if os.path.exists(CMI_path):
            CMI_array = pd.read_csv(self.save_dir + str(root) + '_CMI.csv', index_col=0, header=0)
            CMI_array.columns = CMI_array.columns.astype(str)
            CMI_array.index = CMI_array.index.astype(str)
        else:
            print(f"calculate CMI matrix based on root: {root}")
            leaves = [n for n in self.tree.nodes if n != root]
            CMI_array = np.zeros(shape=(len(leaves), len(leaves)))
            CMI_array = pd.DataFrame(CMI_array, index=leaves, columns=leaves)
            start = time.time()
            for leaf_i, leaf_j in itertools.combinations(leaves, 2):
                express_i = self.tree.nodes[leaf_i]['express']
                express_j = self.tree.nodes[leaf_j]['express']
                express_root = self.tree.nodes[root]['express']
                if leaf_i == leaf_j:
                    continue
                if self.kde:
                    CMI = get_kde_conditional_mutual_information(express_i, express_j, express_root)
                else:
                    CMI = get_conditional_mutual_information(valuesX=express_i, valuesY=express_j, valuesZ=express_root)
                # print(f'leaf {leaf_i}, leaf {leaf_j}, root {root}: {CMI}')
                CMI_array.loc[leaf_i, leaf_j] = CMI
                CMI_array.loc[leaf_j, leaf_i] = CMI
            print(f"calculate CMI matrix based on root: {root} takes {time.time() - start} seconds")
            CMI_array.to_csv(self.save_dir + "/" + str(root) + '_CMI.csv', index=True, header=True)
        return CMI_array

    def get_cluster(self, X):
        '''
        use cluster methods to split the nodes to two clusters
        we use spectral clustering here, so it should be even cluster size.
        AND we cluster_num = 2
        :param X: CMI matrix for nodes which will be split
        :return:two groups of nodes
        '''
        X = (X + X.T) / 2
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=2, n_init=100, affinity='precomputed')
        labels = spectral.fit_predict(X)
        dict = {i: X.columns.values[np.where(labels == i)].tolist() for i in range(2)}
        return dict

    def get_subroots(self):
        parents = [list(self.tree.predecessors(node))[0] for node in self.unfix_nodes]
        dict = {i: np.array(self.unfix_nodes)[np.where(np.array(parents) == i)[0]] for i in list(set(parents))}
        return dict

    def items_in_root_cluster(self, label, arr):
        r_label = label[np.where(arr == self.root)[0][0]]
        r_label_num = len([i for i in label if i == r_label])
        return r_label_num

    def sort_nodes_by_entropy(self, candidate_nodes):
        print("candidate nodes:", list(candidate_nodes))
        leaves = [item for item in candidate_nodes if self.tree.nodes[item]["compNum"] == 1]
        manifold_nodes = [item for item in candidate_nodes if item not in leaves]  # remove leaves
        manifold_nodes.sort(key=lambda x: self.tree.nodes[x]["dis2Root"])
        if len(manifold_nodes) > 1:
            dis1 = self.tree.nodes[manifold_nodes[0]]["dis2Root"]
            dis2 = self.tree.nodes[manifold_nodes[1]]["dis2Root"]
            comp1 = self.tree.nodes[manifold_nodes[0]]["compNum"]
            comp2 = self.tree.nodes[manifold_nodes[1]]["compNum"]
            if dis1 == dis2 and comp1 < comp2:
                manifold_nodes[0], manifold_nodes[1] = manifold_nodes[1], manifold_nodes[0]
        candidate_results = manifold_nodes + leaves
        print("sorted nodes:", candidate_results)
        return candidate_results

    def find_matrix_connected_components(self, CMI):
        graph = csr_matrix(CMI.values)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        return n_components, labels

    def find_sub_root(self, candidate_nodes):
        CompNumList = [self.tree.nodes[item]["compNum"] for item in candidate_nodes]
        if len(candidate_nodes) == 1:  # one leaf or root
            self.tree.nodes[candidate_nodes[0]]['fix'] = 1
            if candidate_nodes[0] == self.root and max(CompNumList) == 2:
                self.tree.nodes[self.root]['compNum'] = -1  # root should be binary
        elif len(candidate_nodes) == 2 and max(CompNumList) == 1:  # two leaves
            self.tree.nodes[candidate_nodes[0]]['fix'] = 1
            self.tree.nodes[candidate_nodes[1]]['fix'] = 1
        else:
            candidate_nodes = self.sort_nodes_by_entropy(candidate_nodes)
            self.tree.nodes[candidate_nodes[0]]['fix'] = 1
            for i in range(1, len(candidate_nodes)):
                candidate_edge = [edge for edge in self.tree.edges if
                                  candidate_nodes[i] in edge]  # fix have been removed
                self.tree.remove_edge(*candidate_edge[0])
                self.tree.add_edge(candidate_nodes[0], candidate_nodes[i])
        self.get_unfix_nodes_express()

    def get_next_dict(self):
        paList, dict = self.get_subroots(), []
        for pa in paList:
            if len(paList[pa]) > 1 and (self.tree.nodes[pa]['compNum'] > 2 or self.tree.nodes[pa]['compNum'] == -1):
                CMI = self.tree.nodes[self.root]['CMI']
                candidates = [i for i in self.unfix_nodes if i in paList[pa]]
                CMI = CMI.loc[candidates, candidates]
                CMI = (CMI - CMI.min()) / (CMI.max() - CMI.min())
                CMI = CMI.fillna(0)
                Cluster = self.get_cluster(CMI)
                dict.append(Cluster[0])
                dict.append(Cluster[1])
                print("subroot:", pa, "cluster:", Cluster[0], " and ", Cluster[1])
            else:
                dict.append(paList[pa])
        return dict
