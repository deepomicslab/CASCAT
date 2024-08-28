import numpy as np

from InformationMeasure.src.DiscretizationAlgorithm import DiscretizationAlgorithm


class DiscretizeUniformWidth(DiscretizationAlgorithm):
    def __init__(self, nbins: int):
        super().__init__(nbins)

    def binedges(self, data: list):
        min_, max_ = min(data), max(data)
        if isinstance(self.nbins, str):
            self.nbins = self.get_nbins(data, self.nbins)
        edges = np.linspace(min_, max_, self.nbins + 1)
        return edges
