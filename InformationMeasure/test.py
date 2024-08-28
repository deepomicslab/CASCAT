from InformationMeasure.testmodules.testDiscretization import testDiscretize
from InformationMeasure.testmodules.testEntropy import testEntropy
from InformationMeasure.testmodules.testDiscretize import testDiscUniformCount, testDiscUniformWidth, \
    testDiscBayesianBlocks
from InformationMeasure.testmodules.testMutualInformation import testGetMutualInfo, testApplyMutualInfo
from InformationMeasure.testmodules.testConditionalMutaulInformation import testGetCMI, testApplyCMI
from InformationMeasure.testmodules.testConditionalEntropy import testGetConditionalEntropy, \
    testApplyConditionalEntropy, \
    testConditionalEntropy
from InformationMeasure.testmodules.testTotalCorrelation import testGetTotalCor, testApplyTotalCor

if __name__ == '__main__':
    testDiscretize()
    testEntropy()
    testDiscUniformCount()
    testDiscUniformWidth()
    testDiscBayesianBlocks()
    testGetMutualInfo()
    testApplyMutualInfo()
    testGetCMI()
    testApplyCMI()
    testGetConditionalEntropy()
    testApplyConditionalEntropy()
    testConditionalEntropy()
    testGetTotalCor()
    testApplyTotalCor()
