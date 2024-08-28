import random
import numpy as np
from InformationMeasure.Measures import *


def testDiscretize():
    arr = np.array([random.random() for _ in range(100)])
    assert discretize_values(valuesX=arr).shape == (10,)
    print(f'Discretize with 1 array passed')
    assert discretize_values(valuesX=arr, valuesY=arr).shape == (10, 10)
    print(f'Discretize with 2 arrays passed')
    assert discretize_values(valuesX=arr, number_of_bins=5).shape == (5,)
    print(f'Discretize with 1 array and 2 bins passed')
    assert discretize_values(valuesX=arr, valuesY=arr, number_of_bins=2).shape == (2, 2)
    print(f'Discretize with 2 arrays and 2 bins passed')


def testProbability():
    arr = [1,1,4,4,4,4,3,3,3,3,2,2,2]
    freq = discretize_values(valuesX=arr, number_of_bins=4)
    prob = get_probabilities("maximum_likelihood", freq)
    print(prob)
    prob2 = get_probabilities("shrinkage", freq)
    print(prob2)
    prob3 = get_probabilities("dirichlet", freq)
    print(prob3)
    prob4 = get_probabilities("miller_madow", freq)
    print(prob4)


if __name__ == '__main__':
    # testDiscretize()
    testProbability()