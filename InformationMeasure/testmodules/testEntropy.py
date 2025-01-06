import math
import random

import numpy as np

from InformationMeasure.Measures import *


def testEntropy():
    arr = np.array([random.random() for _ in range(1000)])
    sqrt_1000 = 31.622776601683793
    theshold_min, theshold_max = math.log2(sqrt_1000) - 0.05, math.log2(sqrt_1000) + 0.05
    assert theshold_min <= get_entropy(valuesX=arr) <= theshold_max
    print(f'Entropy with 1 array passed')

    assert theshold_min <= get_entropy(valuesX=arr, valuesY=arr) <= theshold_max
    print(f'Entropy with 2 arrays passed')

    assert theshold_min <= get_entropy(valuesX=arr, valuesY=arr, valuesZ=arr) <= theshold_max
    print(f'Entropy with 3 arrays passed')

    assert theshold_min <= get_entropy(valuesX=arr, estimator='uniform_width') <= theshold_max
    print(f'Entropy with 1 array and uniform width passed')

    assert math.log2(5) - 0.05 <= get_entropy(valuesX=arr, number_of_bins=5) <= math.log2(5) + 0.05
    print(f'Entropy with 1 array and 5 bins passed')

    def get_number(values):
        return 2

    assert math.log2(2) - 0.05 <= get_entropy(valuesX=arr, get_number_of_bins=get_number) <= math.log2(2) + 0.05
    print(f'Entropy with 1 array and get number of bins passed')

    assert theshold_min <= get_entropy(valuesX=arr, estimator='maximun_likelihood') <= theshold_max
    print(f'Entropy with 1 array and maximum likelihood passed')

    assert theshold_min <= get_entropy(valuesX=arr, estimator='shrinkage') <= theshold_max
    print(f'Entropy with 1 array and shrinkage passed')

    assert theshold_min <= get_entropy(valuesX=arr, estimator='shrinkage', lamda=0) <= theshold_max
    print(f'Entropy with 1 array and shrinkage with lamda=0 passed')

    assert theshold_min <= get_entropy(valuesX=arr, estimator='dirichlet') <= theshold_max
    print(f'Entropy with 1 array and dirichlet passed')

    assert theshold_min <= get_entropy(valuesX=arr, estimator='dirichlet', prior=1) <= theshold_max
    print(f'Entropy with 1 array and dirichlet with prior=1 passed')

    assert theshold_min <= get_entropy(valuesX=arr, estimator='miller_madow') <= theshold_max + 0.5
    print(f'Entropy with 1 array and miller madow passed')

    assert math.log(sqrt_1000) - 0.05 <= get_entropy(valuesX=arr, base=math.exp(1)) <= math.log(sqrt_1000) + 0.05
    print(f'Entropy with 1 array and base=e passed')

    assert theshold_min <= get_entropy(discretize_values(valuesX=arr)) <= theshold_max
    print(f'Entropy with 1 array and discretize values passed')
