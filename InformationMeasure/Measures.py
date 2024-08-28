from InformationMeasure.Formulas import *
from InformationMeasure.Discretization import *
from InformationMeasure.Estimators import *


def discretize_values(discretize="uniform_width", number_of_bins=0, get_number_of_bins='sqrt', valuesX=None,
                      valuesY=None, valuesZ=None):
    assert isinstance(valuesX, np.ndarray)
    if number_of_bins == 0 and type(get_number_of_bins) == str:
        number_of_bins = get_number_of_bins
    elif number_of_bins == 0 and callable(get_number_of_bins):
        number_of_bins = get_number_of_bins(valuesX)
    return get_frequencies(number_of_bins, discretize, valuesX, valuesY, valuesZ)


def get_probabilities(estimator, frequencies, lamda=None, prior=1):
    if estimator == "dirichlet":
        probabilities = get_probabilities_dirichlet(frequencies, prior)
    elif estimator == "shrinkage":
        probabilities = get_probabilities_shrinkage(frequencies, lamda)
    else:
        # "maximum_likelihood" or "miller_madow":
        probabilities = get_probabilities_maximum_likelihood(frequencies)
    return probabilities


def get_mutual_information(probabilites=False, estimator='maximum_likelihood', base=2, discretize="uniform_width",
                           number_of_bins=0, get_number_of_bins='sqrt', lamda=None, prior=1, valuesX=None,
                           valuesY=None):
    if isinstance(probabilites, np.ndarray):
        prob_xy = probabilites
    else:
        freq_xy = discretize_values(discretize, number_of_bins, get_number_of_bins, valuesX, valuesY)
        prob_xy = get_probabilities(estimator, freq_xy, lamda, prior)
    prob_x, prob_y = prob_xy.sum(axis=1), prob_xy.sum(axis=0)
    entropy_x = apply_entropy_formula(prob_x, base=base)
    entropy_y = apply_entropy_formula(prob_y, base=base)
    entropy_xy = apply_entropy_formula(prob_xy, base=base)
    print(entropy_x, entropy_y, entropy_xy)
    if estimator == "miller_madow":
        entropy_xy += ((len(np.nonzero(prob_xy)[0]) - 1) / (2.0 * len(prob_y) * len(prob_x)))
        entropy_x += (len(np.nonzero(prob_x)[0]) - 1) / (2.0 * len(prob_x))
        entropy_y += (len(np.nonzero(prob_y)[0]) - 1) / (2.0 * len(prob_y))
    return apply_mutual_information_formula(entropy_x, entropy_y, entropy_xy)


def get_entropy(probabilites=False, estimator='maximum_likelihood', base=2, discretize="uniform_width",
                number_of_bins=0, get_number_of_bins='sqrt', lamda=None, prior=1, valuesX=None, valuesY=None,
                valuesZ=None):
    if isinstance(probabilites, np.ndarray):
        probabilites = probabilites.copy()
    else:
        freq = discretize_values(discretize, number_of_bins, get_number_of_bins, valuesX, valuesY, valuesZ)
        probabilites = get_probabilities(estimator, freq, lamda, prior)
    entropys = apply_entropy_formula(probabilites, base=base)
    if estimator == "miller_madow":
        entropys += (len(np.nonzero(probabilites)[0]) - 1) / (2.0 * len(probabilites))
    return entropys + 0.0


def get_conditional_mutual_information(probabilites=False, estimator='maximum_likelihood', base=2,
                                       discretize="uniform_width", number_of_bins=0, get_number_of_bins='sqrt',
                                       lamda=None, prior=1, valuesX=None, valuesY=None, valuesZ=None):
    if isinstance(probabilites, np.ndarray):
        prob_xyz = probabilites
    else:
        freq_xyz = discretize_values(discretize, number_of_bins, get_number_of_bins, valuesX=valuesX, valuesY=valuesY,
                                     valuesZ=valuesZ)
        prob_xyz = get_probabilities(estimator, freq_xyz, lamda, prior)
    prob_xz = prob_xyz.sum(axis=1)
    prob_yz = prob_xyz.sum(axis=0)
    prob_z = prob_xyz.sum(axis=(0, 1))
    entropy_xz = apply_entropy_formula(prob_xz, base=base)
    entropy_yz = apply_entropy_formula(prob_yz, base=base)
    entropy_xyz = apply_entropy_formula(prob_xyz.flatten(), base=base)
    entropy_z = apply_entropy_formula(prob_z, base=base)
    if estimator == "miller_madow":
        entropy_xz += (len(np.nonzero(prob_xz)[0]) - 1) / (2.0 * len(prob_z) * len(prob_z))
        entropy_yz += (len(np.nonzero(prob_yz)[0]) - 1) / (2.0 * len(prob_z) * len(prob_z))
        entropy_xyz += (len(np.nonzero(prob_xyz)[0]) - 1) / (2.0 * len(prob_z) * len(prob_z) * len(prob_z))
        entropy_z += (len(np.nonzero(prob_z)[0]) - 1) / (2.0 * len(prob_z))
    return apply_conditional_mutual_information_formula(entropy_xz, entropy_yz, entropy_xyz, entropy_z)


def get_conditional_entropy(probabilites=False, estimator='maximum_likelihood',
                            base=2, discretize="uniform_width", number_of_bins=0,
                            get_number_of_bins='sqrt', lamda=None, prior=1, valuesX=None, valuesY=None):
    if isinstance(probabilites, np.ndarray):
        prob_xy = probabilites
    else:
        freq_xy = discretize_values(discretize, number_of_bins, get_number_of_bins, valuesX=valuesX, valuesY=valuesY)
        prob_xy = get_probabilities(estimator, freq_xy, lamda, prior)
    prob_y = prob_xy.sum(axis=0)
    entropy_y = apply_entropy_formula(prob_y, base=base)
    entropy_xy = apply_entropy_formula(prob_xy.flatten(), base=base)
    if estimator == "miller_madow":
        entropy_y += (len(np.nonzero(prob_y)[0]) - 1) / (2.0 * len(prob_y))
        entropy_xy += (len(np.nonzero(prob_xy)[0]) - 1) / (2.0 * len(prob_y) * len(prob_y))
    return apply_conditional_entropy_formula(entropy_xy, entropy_y)


def get_total_correlation(valuesX, valuesY, valueZ, probabilites=False, estimater='maximum_likelihood', base=2,
                          discretize="uniform_width", number_of_bins=0, get_number_of_bins='sqrt', lamda=None,
                          prior=1):
    if probabilites != False:
        prob_xyz = probabilites
    else:
        freq_xyz = discretize_values(discretize, number_of_bins, get_number_of_bins, valuesX=valuesX, valuesY=valuesY,
                                     valuesZ=valueZ)
        prob_xyz = get_probabilities(estimater, freq_xyz, lamda, prior)
    prob_xy = prob_xyz.sum(axis=2)
    prob_x, prob_y = prob_xy.sum(axis=1), prob_xy.sum(axis=0)
    prob_z = prob_xyz.sum(axis=(0, 1))
    entropy_xyz = apply_entropy_formula(prob_xyz.flatten(), base=base)
    entropy_x = apply_entropy_formula(prob_x, base=base)
    entropy_y = apply_entropy_formula(prob_y, base=base)
    entropy_z = apply_entropy_formula(prob_z, base=base)
    return apply_total_correlation_formula(entropy_x, entropy_y, entropy_z, entropy_xyz)


def get_partical_correlation(valuesX, valuesY, valuesZ):
    import statsmodels.api as sm
    Z = sm.add_constant(valuesZ)
    modelX = sm.OLS(valuesX, Z).fit()
    modelY = sm.OLS(valuesY, Z).fit()
    resid_X = modelX.resid
    resid_Y = modelY.resid
    partical_correlation = np.corrcoef(resid_X, resid_Y)[0, 1]
    return partical_correlation


def get_correlation(valuesX, valuesY):
    return np.corrcoef(valuesX, valuesY)[0, 1]
