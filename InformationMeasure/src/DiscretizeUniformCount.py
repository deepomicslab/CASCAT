import numpy as np

from InformationMeasure.src.DiscretizationAlgorithm import DiscretizationAlgorithm


class DiscretizeUniformCount(DiscretizationAlgorithm):
    def __init__(self, nbins: int):
        super().__init__(nbins)

    # I think the implement of Integers is not correct, so i only implement the Floats
    def binedges(self, data: list):
        if isinstance(self.nbins, str):
            self.get_nbins(data, self.nbins)
        n = len(data)
        assert n >= self.nbins
        p = np.argsort(data)
        counts_per_bins, remainder = n // self.nbins, n % self.nbins  # count number in one bin is the same
        bins_edges = np.zeros(self.nbins + 1)
        bins_edges[0] = data[p[0]]
        bins_edges[-1] = data[p[-1]]
        ind = 0
        for i in range(self.nbins - 1):
            counts = counts_per_bins + 1 if remainder > 0 else counts_per_bins
            remainder -= 1
            ind += counts
            bins_edges[i + 1] = (data[p[ind - 1]] + data[p[ind]]) / 2
            assert bins_edges[i + 1] != bins_edges[i]
        return bins_edges
