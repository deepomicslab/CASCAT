import cupy as cp
import numpy as np
import pandas as pd
import cupyx.scipy.stats as stats
import torch
import tqdm
from InformationMeasure.Measures import get_entropy


def get_entropy_matrix(nodes, expres):
    if isinstance(expres, torch.Tensor):
        expres = expres.tolist()
    result = []
    for node in tqdm.tqdm(nodes):
        exp = get_entropy(valuesX=np.array(expres[node]))
        result.append(exp)
    result = pd.DataFrame(result, index=nodes, columns=['entropy'])
    print(f'Entropy matrix shape: {result.shape}')
    return result


def get_dual_joint_entropy_matrix(pairs, expres):
    if isinstance(expres, torch.Tensor):
        expres = cp.array(expres.tolist(), dtype=cp.float64)
    result = pd.DataFrame(pairs, columns=['Cell1', 'Cell2'])
    pairs = cp.array(pairs, dtype=cp.int64)
    dual_entropys = get_fast_dual_entropy(pairs, expres)
    result['entropy'] = dual_entropys.get()
    print(f'Dual Entropy matrix shape: {result.shape}')
    return result


def get_fast_dual_entropy(pairs, express, batch_size=30000):
    dual_entropy = cp.zeros(shape=(len(pairs),), dtype=cp.float64)
    for i in tqdm.tqdm(range(0, len(pairs), batch_size)):
        index = cp.array(pairs[i: int(cp.minimum(i + batch_size, len(pairs)))], dtype=cp.int64)
        valuesX_batch = express[index[:, 0]]
        valuesY_batch = express[index[:, 1]]
        number_of_bins_x = cp.ceil(cp.sqrt(valuesX_batch.shape[1])).astype(cp.int64)
        number_of_bins_y = cp.ceil(cp.sqrt(valuesY_batch.shape[1])).astype(cp.int64)
        number_of_bins_x = int(cp.minimum(number_of_bins_x, 30))
        number_of_bins_y = int(cp.minimum(number_of_bins_y, 30))
        bins_edges_x = cp.linspace(cp.min(valuesX_batch), cp.max(valuesY_batch), number_of_bins_x)
        bins_edges_y = cp.linspace(cp.min(valuesY_batch), cp.max(valuesY_batch), number_of_bins_y)
        bins_ids_x_batch = cp.digitize(valuesX_batch, bins_edges_x)
        bins_ids_y_batch = cp.digitize(valuesY_batch, bins_edges_y)
        frequencies = cp.zeros((number_of_bins_x, number_of_bins_y), dtype=cp.float64)
        cp.add.at(frequencies, (bins_ids_x_batch, bins_ids_y_batch), 1)
        frequencies = frequencies.flatten().astype(cp.float64)
        probs = (frequencies / cp.sum(frequencies)).astype(cp.float64)
        dual_entropy[i:i + batch_size] = stats.entropy(probs)
    return dual_entropy


def get_triple_joint_entropy_matrix(pairs, expres):
    if isinstance(expres, torch.Tensor):
        expres = cp.array(expres.tolist(), dtype=cp.float64)
    result = pd.DataFrame(pairs, columns=['Cell1', 'Cell2', 'Cell3'])
    pairs = cp.array(pairs, dtype=cp.int64)
    triplet_entropys = get_fast_triple_entropy(pairs, expres)
    result['entropy'] = triplet_entropys.get()
    print(f'Triple Entropy matrix shape: {result.shape}')
    return result


def get_fast_triple_entropy(pairs, express, batch_size=30000):
    triplet_entropy = cp.zeros(shape=(len(pairs),), dtype=cp.float64)
    for i in tqdm.tqdm(range(0, len(pairs), batch_size)):
        index = cp.array(pairs[i: int(cp.minimum(i + batch_size, len(pairs)))], dtype=cp.int64)
        valuesX_batch = express[index[:, 0]]
        valuesY_batch = express[index[:, 1]]
        valuesZ_batch = express[index[:, 2]]
        number_of_bins_x = cp.ceil(cp.sqrt(valuesX_batch.shape[1])).astype(cp.int64)
        number_of_bins_y = cp.ceil(cp.sqrt(valuesY_batch.shape[1])).astype(cp.int64)
        number_of_bins_z = cp.ceil(cp.sqrt(valuesZ_batch.shape[1])).astype(cp.int64)
        number_of_bins_x = int(cp.minimum(number_of_bins_x, 30))
        number_of_bins_y = int(cp.minimum(number_of_bins_y, 30))
        number_of_bins_z = int(cp.minimum(number_of_bins_z, 30))
        bins_edges_x = cp.linspace(cp.min(valuesX_batch), cp.max(valuesY_batch), number_of_bins_x)
        bins_edges_y = cp.linspace(cp.min(valuesY_batch), cp.max(valuesY_batch), number_of_bins_y)
        bins_edges_z = cp.linspace(cp.min(valuesZ_batch), cp.max(valuesZ_batch), number_of_bins_z)
        bins_ids_x_batch = cp.digitize(valuesX_batch, bins_edges_x)
        bins_ids_y_batch = cp.digitize(valuesY_batch, bins_edges_y)
        bins_ids_z_batch = cp.digitize(valuesZ_batch, bins_edges_z)
        frequencies = cp.zeros((number_of_bins_x, number_of_bins_y, number_of_bins_z), dtype=cp.float64)
        cp.add.at(frequencies, (bins_ids_x_batch, bins_ids_y_batch, bins_ids_z_batch), 1)
        frequencies = frequencies.flatten().astype(cp.float64)
        probs = (frequencies / cp.sum(frequencies)).astype(cp.float64)
        triplet_entropy[i:i + batch_size] = stats.entropy(probs)
    return triplet_entropy


def get_conditional_mutual_info_matrix(tri_pairs, entropy, dual_entropy, triplet_entropy):
    entropy = cp.array(entropy, dtype=cp.float64)
    dual_entropy = cp.array(dual_entropy, dtype=cp.float64)
    triplet_entropy = cp.array(triplet_entropy, dtype=cp.float64)
    result = pd.DataFrame(tri_pairs, columns=['Cell1', 'Cell2', 'Cell3'])
    tri_pairs = cp.array(tri_pairs, dtype=np.int64)
    CMI_List = apply_conditional_mutual_info(tri_pairs, entropy, dual_entropy, triplet_entropy)
    result['CMI'] = CMI_List.get()
    print(f'Conditional Mutual Information matrix shape: {result.shape}')
    return result


def encode_pairs(pairs, max_value):
    return pairs[:, 0] * max_value + pairs[:, 1]


def encode_tri_pairs(indices, max_value, max_squared):
    return indices[:, 0] * max_squared + indices[:, 1] * max_value + indices[:, 2]


def apply_conditional_mutual_info(tri_pairs, entropy, dual_entropy, triplet_entropy, batch_size=30000):
    CMIs = cp.zeros(shape=(len(tri_pairs),), dtype=cp.float64)
    max_value = cp.max(tri_pairs) + 1
    max_squared = max_value ** 2
    encoded_dual_entropy = encode_pairs(cp.array(dual_entropy[:, :2], dtype=cp.int64), max_value)
    encoded_triplet_entropy = encode_tri_pairs(cp.array(triplet_entropy[:, :3], dtype=cp.int64), max_value, max_squared)
    for i in tqdm.tqdm(range(0, len(tri_pairs), batch_size)):
        CMIs[i:i + batch_size] = get_fast_cond_mutual_info(tri_pairs[i: int(cp.minimum(i + batch_size, len(tri_pairs)))]
                                                           , entropy, dual_entropy, triplet_entropy,
                                                           encoded_dual_entropy, encoded_triplet_entropy, max_value,
                                                           max_squared)

    return CMIs


def get_fast_cond_mutual_info(tri_pairs, entropy, dual_entropy, triplet_entropy, encode_dual_entropy,
                              encode_triplet_entropy, max_value, max_squared):
    sorted_tri_pairs = cp.sort(tri_pairs, axis=1)
    xz_pairs = sorted_tri_pairs[:, [0, 2]]
    yz_pairs = sorted_tri_pairs[:, [1, 2]]
    xyz_pairs = sorted_tri_pairs
    xz_pairs = encode_pairs(xz_pairs, max_value)
    yz_pairs = encode_pairs(yz_pairs, max_value)
    xyz_pairs = encode_tri_pairs(xyz_pairs, max_value, max_squared)
    xz_indices = cp.searchsorted(encode_dual_entropy, xz_pairs, side='left')
    yz_indices = cp.searchsorted(encode_dual_entropy, yz_pairs, side='left')
    xyz_indices = cp.searchsorted(encode_triplet_entropy, xyz_pairs, side='left')
    entropy_xz = dual_entropy[xz_indices, 2]
    entropy_yz = dual_entropy[yz_indices, 2]
    entropy_xyz = triplet_entropy[xyz_indices, 3]
    entropy_z = entropy[tri_pairs[:, 2], 0]
    CMIs = entropy_xz + entropy_yz - entropy_xyz - entropy_z
    return CMIs
