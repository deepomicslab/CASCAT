from numba import prange, njit
from InformationMeasure.Measures import *
import pandas as pd
import numpy as np
import torch
import tqdm
from scipy.stats import entropy
import scipy.special as sp


# pip install numba-scipy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.entr.html#scipy.special.entr

def get_entropy_matrix(nodes, expres):
    if isinstance(expres, torch.Tensor):
        expres = expres.tolist()
    result = []
    for node in tqdm.tqdm(nodes):
        exp = get_entropy(valuesX=np.array(expres[node]))
        result.append(exp)
    result = pd.DataFrame(result, index=nodes, columns=['entropy'])
    print(f'The number of entropy items: {result.shape[0]}')
    return result


def get_conditional_mutual_info_matrix(tri_pairs, entropy, dual_entropy, triplet_entropy):
    entropy = np.array(entropy, dtype=np.float64)
    dual_entropy = np.array(dual_entropy, dtype=np.float64)
    triplet_entropy = np.array(triplet_entropy, dtype=np.float64)
    result = pd.DataFrame(tri_pairs, columns=['Cell1', 'Cell2', 'Cell3'])
    tri_pairs = np.array(tri_pairs, dtype=np.int64)
    CMI_List = apply_conditional_mutual_info(tri_pairs, entropy, dual_entropy, triplet_entropy)
    result['CMI'] = CMI_List
    return result


@njit(parallel=True, cache=True)
def apply_conditional_mutual_info(tri_pairs, entropy, dual_entropy, triplet_entropy):
    CMIs = np.empty(shape=(len(tri_pairs),), dtype=np.float64)
    for i in prange(len(tri_pairs)):
        xz, yz, xyz = sorted([tri_pairs[i, 0], tri_pairs[i, 2]]), sorted([tri_pairs[i, 1], tri_pairs[i, 2]]), sorted(
            tri_pairs[i])
        entropy_xz = dual_entropy[(dual_entropy[:, 0] == xz[0]) & (dual_entropy[:, 1] == xz[1])][0][2]
        entropy_yz = dual_entropy[(dual_entropy[:, 0] == yz[0]) & (dual_entropy[:, 1] == yz[1])][0][2]
        entropy_xyz = triplet_entropy[(triplet_entropy[:, 0] == xyz[0]) & (triplet_entropy[:, 1] == xyz[1]) & (
                triplet_entropy[:, 2] == xyz[2])][0][3]
        entropy_z = entropy[tri_pairs[i, 2]][0]
        CMIs[i] = entropy_xz + entropy_yz - entropy_xyz - entropy_z
    return CMIs


def get_dual_joint_entropy_matrix(pairs, expres):
    if isinstance(expres, torch.Tensor):
        expres = np.array(expres.tolist(), dtype=np.float64)
    result = pd.DataFrame(pairs, columns=['Cell1', 'Cell2'])
    pairs = np.array(pairs, dtype=np.int64)
    dual_entropys = get_fast_dual_entropy(pairs, expres)
    result['entropy'] = dual_entropys
    print(f'The number of dual entropy pairs: {result.shape[0]}')
    return result


@njit(parallel=True)
def get_fast_dual_entropy(pairs, express):
    dual_entropy = np.zeros(shape=(len(pairs),), dtype=np.float64)
    for p in prange(len(pairs)):
        valuesX = express[pairs[p, 0]]
        valuesY = express[pairs[p, 1]]
        number_of_bins_x = int(np.ceil(np.sqrt(len(valuesX))))
        number_of_bins_y = int(np.ceil(np.sqrt(len(valuesY))))
        if number_of_bins_x > 30:
            number_of_bins_x = number_of_bins_y = 30
        bins_edges_x = np.linspace(min(valuesX), max(valuesX), number_of_bins_x)
        bins_ids_x = np.digitize(valuesX, bins_edges_x)
        bins_edges_y = np.linspace(min(valuesY), max(valuesY), number_of_bins_y)
        bins_ids_y = np.digitize(valuesY, bins_edges_y)
        frequencies = np.zeros((number_of_bins_x, number_of_bins_y))
        for i in prange(len(bins_ids_x)):
            frequencies[int(bins_ids_x[i] - 1)][int(bins_ids_y[i] - 1)] += 1
        frequencies = frequencies.flatten().astype(np.float64)
        probs = (frequencies / sum(frequencies)).astype(np.float64)
        for j in prange(len(probs)):
            dual_entropy[p] += sp.entr(probs[j])
        dual_entropy[p] /= np.log(2)
    return dual_entropy


def get_triple_joint_entropy_matrix(pairs, expres):
    if isinstance(expres, torch.Tensor):
        expres = np.array(expres.tolist(), dtype=np.float64)
    result = pd.DataFrame(pairs, columns=['Cell1', 'Cell2', 'Cell3'])
    pairs = np.array(pairs, dtype=np.int64)
    triplet_entropys = get_fast_triple_entropy(pairs, expres)
    result['entropy'] = triplet_entropys
    print(f'The number of triple entropy pairs: {result.shape[0]}')
    return result


@njit(parallel=True)
def get_fast_triple_entropy(pairs, express):
    triplet_entropy = np.zeros(shape=(len(pairs),), dtype=np.float64)
    for p in prange(len(pairs)):
        valuesX = express[pairs[p, 0]]
        valuesY = express[pairs[p, 1]]
        valuesZ = express[pairs[p, 2]]
        number_of_bins_x = int(np.ceil(np.sqrt(len(valuesX))))
        number_of_bins_y = int(np.ceil(np.sqrt(len(valuesY))))
        number_of_bins_z = int(np.ceil(np.sqrt(len(valuesZ))))
        if number_of_bins_x > 30:
            number_of_bins_x = number_of_bins_y = number_of_bins_z = 30
        bins_edges_x = np.linspace(min(valuesX), max(valuesX), number_of_bins_x)
        bins_ids_x = np.digitize(valuesX, bins_edges_x)
        bins_edges_y = np.linspace(min(valuesY), max(valuesY), number_of_bins_y)
        bins_ids_y = np.digitize(valuesY, bins_edges_y)
        bins_edges_z = np.linspace(min(valuesZ), max(valuesZ), number_of_bins_z)
        bins_ids_z = np.digitize(valuesZ, bins_edges_z)
        frequencies = np.zeros((number_of_bins_x, number_of_bins_y, number_of_bins_z))
        for i in prange(len(bins_ids_x)):
            frequencies[int(bins_ids_x[i] - 1)][int(bins_ids_y[i] - 1)][int(bins_ids_z[i] - 1)] += 1
        frequencies = frequencies.flatten().astype(np.float64)
        probs = (frequencies / sum(frequencies)).astype(np.float64)
        for j in prange(len(probs)):
            triplet_entropy[p] += sp.entr(probs[j])
        triplet_entropy[p] /= np.log(2)
    return triplet_entropy
