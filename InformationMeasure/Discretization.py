import numpy as np
from InformationMeasure.src.DiscretizeUniformWidth import DiscretizeUniformWidth
from InformationMeasure.src.DiscretizeUniformCount import DiscretizeUniformCount
from InformationMeasure.src.DiscretizeBayesianBlocks import DiscretizeBayesianBlocks


def get_frequencies(number_of_bins, discretize, valuesX, valuesY=None, valuesZ=None):
    if valuesY is None and valuesZ is None:
        number_of_bins, bins_ids = get_bin_ids(valuesX, discretize, number_of_bins)
        frequencies = np.zeros(number_of_bins)
        for i in bins_ids:
            frequencies[int(i - 1)] += 1
    elif valuesZ is None:
        number_of_bins, bins_ids_x = get_bin_ids(valuesX, discretize, number_of_bins)
        number_of_bins, bins_ids_y = get_bin_ids(valuesY, discretize, number_of_bins)
        frequencies = np.zeros((number_of_bins, number_of_bins))
        for i in range(len(bins_ids_x)):
            frequencies[int(bins_ids_x[i] - 1)][int(bins_ids_y[i] - 1)] += 1
    else:
        number_of_bins, bins_ids_x = get_bin_ids(valuesX, discretize, number_of_bins)
        number_of_bins, bins_ids_y = get_bin_ids(valuesY, discretize, number_of_bins)
        number_of_bins, bins_ids_z = get_bin_ids(valuesZ, discretize, number_of_bins)
        frequencies = np.zeros((number_of_bins, number_of_bins, number_of_bins))
        for i in range(len(bins_ids_x)):
            frequencies[int(bins_ids_x[i] - 1)][int(bins_ids_y[i] - 1)][int(bins_ids_z[i] - 1)] += 1
    return frequencies


def get_bin_ids(values, discretize, number_of_bins):
    global bins_edges
    min, max = np.min(values), np.max(values)
    if min == max:
        return 1, np.ones(len(values))
    elif discretize == "binary":
        number_of_bins = 2
        bins_edges = [0]
    elif discretize == "uniform_count":
        Disc = DiscretizeUniformCount(number_of_bins)
        bins_edges = Disc.binedges(values)
        number_of_bins = Disc.nbins
    elif discretize == "bayesian_blocks":
        Disc = DiscretizeBayesianBlocks()
        bins_edges = Disc.binedges(values)
        number_of_bins = Disc.nbins
    else:
        # uniform width
        Disc = DiscretizeUniformWidth(number_of_bins)
        bins_edges = Disc.binedges(values)
        number_of_bins = Disc.nbins
    # print(f'number of bins: {number_of_bins}')
    bins_ids_ = np.digitize(values, bins_edges)
    bins_ids_ = [bins_ids_[i] - 1 if bins_ids_[i] > number_of_bins else bins_ids_[i] for i in
                 range(len(bins_ids_))]
    return number_of_bins, bins_ids_
