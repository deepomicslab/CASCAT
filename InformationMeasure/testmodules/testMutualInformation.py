import random

import numpy as np
from InformationMeasure.Measures import *


def testGetMutualInfo():
    arr1 = np.array([random.random() for _ in range(1000)])
    arr2 = np.array([random.random() for _ in range(1000)])
    mi = get_entropy(valuesX=arr1) + get_entropy(valuesX=arr2) - get_entropy(valuesX=arr1, valuesY=arr2)
    assert mi - 0.05 <= get_mutual_information(valuesX=arr1, valuesY=arr2) <= mi + 0.05
    print(f'Mutual information with 2 arrays passed')


def testApplyMutualInfo():
    arr1 = np.array([random.random() for _ in range(1000)])
    arr2 = np.array([random.random() for _ in range(1000)])
    entorpy1 = get_entropy(valuesX=arr1)
    entorpy2 = get_entropy(valuesX=arr2)
    entorpy12 = get_entropy(valuesX=arr1, valuesY=arr2)
    mi = entorpy1 + entorpy2 - entorpy12
    assert mi - 0.05 <= apply_mutual_information_formula(entorpy1, entorpy2, entorpy12) <= mi + 0.05
    print(f'Mutual information with 2 arrays passed')
