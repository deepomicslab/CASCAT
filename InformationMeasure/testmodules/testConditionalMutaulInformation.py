import numpy as np

from InformationMeasure.Measures import *
import random


def testGetCMI():
    arr1 = np.array([random.random() for _ in range(1000)])
    arr2 = np.array([random.random() for _ in range(1000)])
    arr3 = np.array([random.random() for _ in range(1000)])
    entropy3 = get_entropy(valuesX=arr3)
    entropy23 = get_entropy(valuesX=arr2, valuesY=arr3)
    entropy13 = get_entropy(valuesX=arr1, valuesY=arr3)
    entropy123 = get_entropy(valuesX=arr1, valuesY=arr2, valuesZ=arr3)
    cmi = entropy13 + entropy23 - entropy123 - entropy3
    assert cmi - 0.05 <= get_conditional_mutual_information(valuesX=arr1, valuesY=arr2, valuesZ=arr3) <= cmi + 0.05
    print(f'Conditional mutual information with 3 arrays passed')


def testApplyCMI():
    arr1 = np.array([random.random() for _ in range(1000)])
    arr2 = np.array([random.random() for _ in range(1000)])
    arr3 = np.array([random.random() for _ in range(1000)])
    entropy3 = get_entropy(valuesX=arr3)
    entropy23 = get_entropy(valuesX=arr2, valuesY=arr3)
    entropy13 = get_entropy(valuesX=arr1, valuesY=arr3)
    entropy123 = get_entropy(valuesX=arr1, valuesY=arr2, valuesZ=arr3)
    cmi = entropy13 + entropy23 - entropy123 - entropy3
    assert cmi - 0.05 <= apply_conditional_mutual_information_formula(entropy13, entropy23, entropy123,
                                                                      entropy3) <= cmi + 0.05
    print(f'Conditional mutual information with 3 arrays passed')


def testConditionalMutualInformation():
    np.random.seed(10)
    p_xyz = np.random.rand(3, 4, 5)
    p_xyz /= np.sum(p_xyz)
    p_yz = np.sum(p_xyz, axis=0)
    p_xz = np.sum(p_xyz, axis=1)
    p_z = np.sum(p_xyz, axis=(0, 1))

    def H(x):
        return -np.sum(x * np.log(x))

    print(get_conditional_mutual_information(probabilites=p_xyz, base=np.exp(1)))  # 0.1066909094805184
    print(H(p_xz) + H(p_yz) - H(p_xyz) - H(p_z))  # 0.1066909094805184


if __name__ == '__main__':
    # testGetCMI()
    # testApplyCMI()
    testConditionalMutualInformation()
