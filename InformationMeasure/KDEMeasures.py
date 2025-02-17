from InformationMeasure.Estimators import *
from InformationMeasure.Formulas import *
from numba import jit


def get_kde_mutual_information(valuesX, valuesY):
    entropy_x = get_1D_kde_density(valuesX)
    entropy_y = get_1D_kde_density(valuesY)
    entropy_xy = get_2D_kde_density(valuesX, valuesY)
    return apply_mutual_information_formula(entropy_x, entropy_y, entropy_xy)


def get_kde_conditional_mutual_information(valuesX, valuesY, valuesZ):
    entropy_xz = get_2D_kde_density(valuesX, valuesZ)
    entropy_yz = get_2D_kde_density(valuesY, valuesZ)
    entropy_xyz = get_3D_kde_density(valuesX, valuesY, valuesZ)
    entropy_z = get_1D_kde_density(valuesZ)
    return apply_conditional_mutual_information_formula(entropy_xz, entropy_yz, entropy_xyz, entropy_z)


def get_kde_cond_mutual_info(nodes, samples, express):
    (node_1, node_2, root) = nodes
    if (node_1, root) not in samples:
        pdf_values = get_2D_pdf(express[node_1]['express'], express[root]['express'])
        samples[(node_1, root)] = pdf_values
    entropy_xz = caculate_entropy(samples[(node_1, root)])
    if (node_2, root) not in samples:
        pdf_values = get_2D_pdf(express[node_2]['express'], express[root]['express'])
        samples[(node_2, root)] = pdf_values
    entropy_yz = caculate_entropy(samples[(node_2, root)])
    if (node_1, node_2, root) not in samples:
        pdf_values = get_3D_pdf(express[node_1]['express'], express[node_2]['express'],
                                express[root]['express'])
        samples[(node_1, node_2, root)] = pdf_values
    entropy_xyz = caculate_entropy(samples[(node_1, node_2, root)])
    if root not in samples:
        pdf_values = get_1D_pdf(express[root]['express'])
        samples[root] = pdf_values
    entropy_z = caculate_entropy(samples[root])
    cond_mutual_info = apply_conditional_mutual_information_formula(entropy_xz, entropy_yz, entropy_xyz, entropy_z)
    return cond_mutual_info, samples


def get_kde_conditional_mutual_information2(valuesX, valuesY, valuesZ):
    '''
    通过express数据计算条件互信息
    :param valuesX:
    :param valuesY:
    :param valuesZ:
    :return:
    '''
    entropy_xz = _mutual_information(valuesX, valuesZ)
    entropy_yz = _mutual_information(valuesY, valuesZ)
    entropy_xyz = get_3D_kde_density(valuesX, valuesY, valuesZ)
    entropy_z = get_1D_kde_density(valuesZ)
    return apply_conditional_mutual_information_formula(entropy_xz, entropy_yz, entropy_xyz, entropy_z)


@jit(cache=True, parallel=True, nopython=True)
def _mutual_information(x1, x2):
    """Calculate the mutual informations."""
    kappa, sigma = 2e-2, 1.0  # under 1000 nodes
    n = len(x1)
    X1 = np.repeat(x1, n).reshape(-1, n)
    K1 = np.exp(-1 / (2 * sigma ** 2) * (X1 ** 2 + X1.T ** 2 - 2 * X1 * X1.T))
    X2 = np.repeat(x2, n).reshape(-1, n)
    K2 = np.exp(-1 / (2 * sigma ** 2) * (X2 ** 2 + X2.T ** 2 - 2 * X2 * X2.T))

    tmp1 = K1 + n * kappa * np.identity(n) / 2
    tmp2 = K2 + n * kappa * np.identity(n) / 2
    K_kappa = np.vstack((np.hstack((tmp1 @ tmp1, K1 @ K2)), np.hstack((K2 @ K1, tmp2 @ tmp2))))
    D_kappa = np.vstack((np.hstack((tmp1 @ tmp1, np.zeros((n, n)))), np.hstack((np.zeros((n, n)), tmp2 @ tmp2))))
    sigma_K, _, _ = np.linalg.svd(K_kappa)
    sigma_D, _, _ = np.linalg.svd(D_kappa)
    ans = (-1 / 2) * (np.sum(np.log(sigma_K)) - np.sum(np.log(sigma_D)))
    return ans
