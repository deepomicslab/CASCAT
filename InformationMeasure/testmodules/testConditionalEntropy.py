import numpy as np

from InformationMeasure.Measures import *
import random


def testGetConditionalEntropy():
    arr1 = np.array([random.random() for _ in range(1000)])
    arr2 = np.array([random.random() for _ in range(1000)])
    entropy1 = get_entropy(valuesX=arr1)
    entropy2 = get_entropy(valuesX=arr2)
    entropy12 = get_entropy(valuesX=arr1, valuesY=arr2)
    ci = entropy12 - entropy2
    assert ci - 0.05 <= get_conditional_entropy(valuesX=arr1, valuesY=arr2) <= ci + 0.05
    print(f'Conditional entropy with 2 arrays passed')
    assert - 0.05 <= get_conditional_entropy(valuesX=arr1, valuesY=arr1) <= 0.05
    print(f'Conditional entropy with the same array passed')


def testApplyConditionalEntropy():
    arr1 = np.array([random.random() for _ in range(1000)])
    arr2 = np.array([random.random() for _ in range(1000)])
    entropy1 = get_entropy(valuesX=arr1)
    entropy2 = get_entropy(valuesX=arr2)
    entropy12 = get_entropy(valuesX=arr1, valuesY=arr2)
    ci = entropy12 - entropy2
    assert ci - 0.05 <= apply_conditional_entropy_formula(entropy12, entropy2) <= ci + 0.05
    print(f'Conditional entropy with 2 arrays passed')


def testConditionalEntropy():
    np.random.seed(0)
    p_xyz = np.random.rand(3, 4, 5)
    p_xyz /= np.sum(p_xyz)
    p_yz = np.sum(p_xyz, axis=0)
    p_z = np.sum(p_xyz, axis=(0, 1))

    def H(x):
        return -np.sum(x * np.log(x))
    # H(Y|Z) using package
    print(get_conditional_entropy(p_yz, base=np.exp(1)))  # 1.3294926714567774
    # H(Y|Z) using formula
    print(H(p_yz) - H(p_z))  # 1.3294926714567774
    print(f'Conditional entropy is correct.')


if __name__ == '__main__':
    # testGetConditionalEntropy()
    # testApplyConditionalEntropy()
    testConditionalEntropy()
