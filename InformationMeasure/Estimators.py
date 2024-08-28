import numpy as np
from multiprocessing import Pool

'''
Estimators are discribed in the paper:
"Entropy Inference and the James-Stein Estimator, with Application to Nonlinear Gene Association Networks"
httpr://arxiv.org/abs/0811.3579
'''
from numba import jit
from scipy.stats import gaussian_kde, qmc


def get_probabilities_dirichlet(frequencies, prior):
    prior_freq = np.ones(len(frequencies)) * prior
    return (frequencies + prior_freq) / (np.sum(frequencies) + np.sum(prior_freq))


def get_probabilities_maximum_likelihood(frequencies):
    return frequencies / np.sum(frequencies)


def get_probabilities_shrinkage(frequencies, lamda=None):
    target = get_uniform_distribution(frequencies)
    normalized_frequencies = get_normalized_frequencies(frequencies)
    caculated_lamda = lamda if lamda is not None else get_lambda(normalized_frequencies, target, np.sum(frequencies))
    return apply_shrinkage_formula(normalized_frequencies, target, caculated_lamda)


def get_uniform_distribution(frequencies):
    return 1 / len(frequencies) ** frequencies.ndim


def get_normalized_frequencies(frequencies):
    return frequencies / np.sum(frequencies)


def get_lambda(normalized_frequencies, target, n):
    if n == 1 or n == 0:
        return 1
    # Unbiased estimator of var of u
    var = normalized_frequencies * (1 - normalized_frequencies) / (n - 1)
    msp = np.sum((normalized_frequencies - target) ** 2)
    lamb = 1.0 if msp == 0 else np.sum(var) / msp
    return 1.0 if lamb > 1.0 else (lamb if lamb > 0.0 else 0.0)


def apply_shrinkage_formula(normalized_frequencies, target, lamda):
    return (1 - lamda) * normalized_frequencies + lamda * target


def get_1D_kde_density(array, sample_size=20000):
    kernel = gaussian_kde(array)
    bounds = np.array([[array.min(), array.max()]])
    sampler = qmc.Halton(d=1, seed=0)
    samples = qmc.scale(sampler.random(sample_size), bounds[:, 0], bounds[:, 1])
    pdf_values = kernel(samples.T)
    integrand_values = caculate_integrand_values(pdf_values)  # we cannot sum because the sum of pdf is not 1
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    entropy = volume * np.mean(integrand_values)  # loss info
    return entropy


def get_2D_kde_density(array1, array2, sample_size=20000):
    kernel = gaussian_kde(np.vstack([array1, array2]))
    bounds = np.array([[array1.min(), array1.max()], [array2.min(), array2.max()]])
    sampler = qmc.Halton(d=2, seed=0)
    samples = qmc.scale(sampler.random(sample_size), bounds[:, 0], bounds[:, 1])
    pdf_values = kernel(samples.T)
    integrand_values = caculate_integrand_values(pdf_values)
    entropy = caculate_entropy(np.array(integrand_values), bounds)
    return entropy


def get_3D_kde_density(array1, array2, array3, sample_size=200000):
    kernel = gaussian_kde(np.vstack([array1, array2, array3]))
    bounds = np.array([[array1.min(), array1.max()], [array2.min(), array2.max()], [array3.min(), array3.max()]])
    sampler = qmc.Halton(d=3, seed=0)
    samples = qmc.scale(sampler.random(sample_size), bounds[:, 0], bounds[:, 1])
    pdf_values = kernel(samples.T)
    integrand_values = caculate_integrand_values(pdf_values)
    entropy = caculate_entropy(np.array(integrand_values), bounds)
    return entropy


@jit(nopython=True, cache=True)
def caculate_integrand_values(pdf_values):
    # cache = True is important lessen the time of compilation
    integrand_values = []
    for i in pdf_values:
        if i > 0:
            integrand_values.append(-i * np.log2(i))
        else:
            integrand_values.append(0)
    return integrand_values


@jit(parallel=True, cache=True, nopython=True)
def caculate_entropy(integrand_values, bounds):
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    entropy = volume * np.mean(integrand_values)
    return entropy
