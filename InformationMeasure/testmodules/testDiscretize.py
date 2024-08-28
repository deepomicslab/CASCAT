import numpy as np

from src.DiscretizeUniformCount import DiscretizeUniformCount
from src.DiscretizeUniformWidth import DiscretizeUniformWidth
from src.DiscretizeBayesianBlocks import DiscretizeBayesianBlocks


def testDiscUniformCount():
    # test when the number of bins is 2
    uniformcount = DiscretizeUniformCount(2)
    assert np.array_equal(uniformcount.binedges([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [1.0, 3.5, 6.0])
    print(f'DiscretizeUniformCount with 2 bins passed test case 1')
    assert np.array_equal(uniformcount.binedges([6.0, 2.0, 30.0, 4.0, 1.0, 1.0]), [1.0, 3.0, 30.0])
    print(f'DiscretizeUniformCount with 2 bins passed test case 2')
    assert np.array_equal(uniformcount.binedges([1.0, 2.0, 3.0, 4.0, 5.0]), [1.0, 3.5, 5.0])
    print(f'DiscretizeUniformCount with 2 bins passed test case 3')

    # test when the number of bins is 3
    uniformcount2 = DiscretizeUniformCount(3)
    assert np.array_equal(uniformcount2.binedges([1.0, 2.0, 3.0, 4.0, 5.0]), [1.0, 2.5, 4.5, 5.0])
    print(f'DiscretizeUniformCount with 3 bins passed test case 1')


def testDiscUniformWidth():
    uniformcount = DiscretizeUniformWidth(2)
    assert np.array_equal(uniformcount.binedges([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [1.0, 3.5, 6.0])
    print(f'DiscretizeUniformWidth with 2 bins passed test case 1')
    assert np.array_equal(uniformcount.binedges([6.0, 2.0, 30.0, 4.0, 1.0, 1.0]), [1, 15.5, 30])
    print(f'DiscretizeUniformWidth with 2 bins passed test case 2')
    assert np.array_equal(uniformcount.binedges([1, 2, 3, 4, 6]), [1.0, 3.5, 6.0])
    print(f'DiscretizeUniformWidth with 2 bins passed test case 3')
    uniformcount2 = DiscretizeUniformWidth(3)
    assert np.array_equal(uniformcount2.binedges([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7]), [1.0, 3.0, 5.0, 7.0])
    print(f'DiscretizeUniformWidth with 3 bins passed test case 1')


def testDiscBayesianBlocks():
    bayesianBlock = DiscretizeBayesianBlocks()
    assert np.array_equal(bayesianBlock.binedges([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [1.0, 6.0])
    print(f'DiscretizeBayesianBlocks passed test case 1')
    assert np.array_equal(bayesianBlock.binedges([6.0, 2.0, 30.0, 4.0, 1.0, 1.0]), [1.0, 1.5, 30.0])
    print(f'DiscretizeBayesianBlocks passed test case 2')
    assert np.array_equal(bayesianBlock.binedges([1, 2, 3, 4, 5, 6]), [1.0, 6.0])
    print(f'DiscretizeBayesianBlocks passed test case 3')
    assert np.array_equal(bayesianBlock.binedges([1, 2, 2, 2, 3]), [1.0, 3.0])
    print(f'DiscretizeBayesianBlocks passed test case 4')
