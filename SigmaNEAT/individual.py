'''
This class holds the logic for storing the genome, create the phenotype,
creating final network from substrate and calculating the output of
that network.
'''

import numba.cuda as cu
import neat
from config import Substrate, Params, System, cudaMethod
from activationFunctions import ActivationFunctions, activate
import numpy as np


@cu.jit
def _getValueKernel(input, output, network, neatData):
    pos = cu.grid(1)
    output[pos][0] = pos
    # fill the input

    # get value for each output node
    pass


@cudaMethod
def _getValueRecursive(network, neatData, element):
    if(network[element[0]][element[1]][1] is not None):
        return network[element[0]][element[1]][1]
    value = 0
    for prevElem in network[element[0]-1][:]:
        weight = neat.getValue(neatData, prevElem, element)
        if(abs(weight) < Params.weightThreshold):
            weight = 0
        # TODO: Activation functions
        value += _getValueRecursive(network, neatData, prevElem) * weight
    element[1] = activate(
        ActivationFunctions.TanH,
        value)
    return element[1]


class Individual:
    neatData = {}

    def __init__(self):
        self.neatData = neat.createDataStructure(
            Substrate.dimension*2, 1)

    def getOutput(self, input):
        '''
        The input should be 2D array of InputSize x TrialCount
        The Output will be the same as input only
        OutputSize x TrialCount dimensions
        '''
        trialCount = np.size(input, 1)
        network = np.array(self._createNetwork())
        output = np.zeros((Substrate.outputSize, trialCount))
        if(System.useGpu):
            blockspergrid = (trialCount +
                             (System.threadsPerBlock - 1)
                             ) // System.threadsPerBlock
            # [dict.copy(self.neatData) for x in range(len(input))]
            neatArray = np.array([1, 2, 3])
            _getValueKernel[
                blockspergrid, System.threadsPerBlock](
                input, output, network, neatArray)
        else:
            raise Exception("Not implemented")
        return output

    def crossOverAndMutate(parent1, parent2):
        child = neat.crossOver(parent1, parent2)
        neat.mutate(child)
        return child

    def _createNetwork(self):
        res = []
        for layer in Substrate.getSubstrate():
            res.append([[x, None] for x in layer])
        return res
