'''
This class holds the logic for storing the genome, create the phenotype,
creating final network from substrate and calculating the output of
that network.
'''

import numba.cuda as cu
import neat
from tools import cudaMethod
from config import Substrate, Params, System
from activationFunctions import ActivationFunctions, activate
import numpy as np


@cu.jit
def _getValueKernel(input, output, network, neat):
    pass


@cudaMethod
def _getValueRecursive(inputItem, outputItem, network, neat):
    if(network[element[0]][element[1]][1] is not None):
        return network[element[0]][element[1]][1]
    value = 0
    for prevElem in network[element[0]-1][:]:
        weight = neat.getValue(data, prevElem+element)
        if(abs(weight) < Params.weightThreshold):
            weight = 0
        # TODO: Activation functions
        value += _getValueKernel(data, prevElem, network) * weight
    element[1] = activate(
        ActivationFunctions.TanH,
        value)
    return element[1]


class Individual:
    neat

    def __init__(self):
        self.neat = neat.createDataStructure(
            Substrate.dimension*2, 1)

    def getOutput(self, input):
        '''
        The input should be an array of TrialCount size
        each containing InputSize element list
        The Output will be the same as input only
        OutputSize x TrialCount dimensions
        '''
        network = self._createNetwork()
        output = [np.zeros(Substrate.outputSize) for x in range(len(input))]
        if(System.useGpu):
            blockspergrid = (len(input) +
                             (System.threadsPerBlock - 1)
                             ) // System.threadsPerBlock
            neatArray = [dict.copy(neat) for x in range(len(input))]
            networkArray = [list.copy(network) for x in range(len(input))]
            _getValueKernel[
                blockspergrid, System.threadsPerBlock](
                input, output, networkArray, neatArray)
        else:
            raise Exception("Not implemented")
        return output

    def crossOverAndMutate(parent1, parent2):
        child = neat.crossOver(parent1, parent2)
        neat.mutate(child)
        return child

    def _createNetwork(self):
        res = []
        for layer in Substrate.nodes:
            res.append([[x, None] for x in layer])
        return res
