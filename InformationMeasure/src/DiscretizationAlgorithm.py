import numpy as np


class DiscretizationAlgorithm():
    def __init__(self, nbins: int):
        self.nbins = nbins

    def get_nbins(self, data: list, alg: str = 'sturges'):
        n = len(data)
        if alg == 'sturges':
            # R’s default method, only accounts for data size.
            # Only optimal for gaussian data and underestimates number of bins for large non-gaussian datasets.
            # It implicitly bases the bin sizes on the range of the data and can perform poorly if n < 30,
            # because the number of bins will be small—less than seven—and unlikely to show trends in the data well.
            # It may also perform poorly if the data are not normally distributed.
            self.nbins = np.ceil(np.log2(n)).astype(int) + 1
        elif alg == 'sqrt':
            # Square root (of data size) estimator, used by Excel and other programs for its speed and simplicit
            self.nbins = np.ceil(np.sqrt(n)).astype(int)
        elif alg == 'rice':
            # Estimator does not take variability into account, only data size.
            # Commonly overestimates number of bins required.
            self.nbins = np.ceil(2 * n ** (1 / 3)).astype(int)
        elif alg == 'doane':
            # An improved version of Sturges’ estimator that works better with non-normal datasets.
            from scipy.stats import skew, sem
            sigma_g1 = sem(data, axis=None, ddof=0)  # Standard error of skewness
            self.nbins = np.ceil(1 + np.log2(n) + np.log2(1 + np.abs(skew(np.array(data))) / sigma_g1)).astype(int)
        elif alg == 'scott':
            # Less robust estimator that that takes into account data variability and data size
            sigma = np.std(data)
            binwidth = 3.5 * sigma / (n ** (1 / 3))
            self.nbins = np.ceil((max(data) - min(data)) / binwidth).astype(int)
        elif alg == 'fd':
            # reedman Diaconis Estimator
            # Robust (resilient to outliers) estimator that takes into account data variability and data size
            from scipy.stats import iqr
            iqr_value = iqr(data)
            binwidth = (2 * iqr_value) / (n ** (1 / 3))
            self.nbins = np.ceil((max(data) - min(data)) / binwidth).astype(int)
        else:
            iqr = np.percentile(data, 75) - np.percentile(data, 25)  # Calculate the interquartile range
            binwidth = (2 * iqr) / (n ** (1 / 3))
            max_, min_ = max(data), min(data)
            nbins_fd = np.ceil((max_ - min_) / binwidth).astype(int)
            nbins_sturges = np.ceil(np.log2(n)).astype(int) + 1
            self.nbins = max(nbins_fd, nbins_sturges)
        return self.nbins
