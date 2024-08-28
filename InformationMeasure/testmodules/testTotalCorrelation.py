import numpy as np

from InformationMeasure.Measures import *
import random


def testGetTotalCor():
    arr1 = np.array([random.random() for _ in range(1000)])
    arr2 = np.array([random.random() for _ in range(1000)])
    arr3 = np.array([random.random() for _ in range(1000)])
    entropy1 = get_entropy(valuesX=arr1)
    entropy2 = get_entropy(valuesX=arr2)
    entropy3 = get_entropy(valuesX=arr3)
    entropy123 = get_entropy(valuesX=arr1, valuesY=arr2, valuesZ=arr3)
    total_cor = entropy1 + entropy2 + entropy3 - entropy123
    assert total_cor - 0.05 <= get_total_correlation(arr1, arr2, arr3) <= total_cor + 0.05
    print(f'Total correlation with 3 arrays passed')


def testApplyTotalCor():
    arr1 = np.array([random.random() for _ in range(1000)])
    arr2 = np.array([random.random() for _ in range(1000)])
    arr3 = np.array([random.random() for _ in range(1000)])
    entropy1 = get_entropy(valuesX=arr1)
    entropy2 = get_entropy(valuesX=arr2)
    entropy3 = get_entropy(valuesX=arr3)
    entropy123 = get_entropy(valuesX=arr1, valuesY=arr2, valuesZ=arr3)
    total_cor = entropy1 + entropy2 + entropy3 - entropy123
    assert total_cor - 0.05 <= apply_total_correlation_formula(entropy1, entropy2, entropy3,
                                                               entropy123) <= total_cor + 0.05
