'''
This class holds the logic for storing the genome, create the phenotype,
creating final network from substrate and calculating the output of
that network.
'''

import numba.cuda as cu
import neat
import constants
from config import cudaMethod
import config
import activationFunctions
import numpy as np


@cu.jit
def _getValueKernel(input, output, innovation, nodeGenes, connectionGenes,
                    inputSize, outputSize):
    pos = cu.grid(1)
    output[pos][0] = pos+constants.NEATDATA__INNOVATION_NUMBER_INDEX
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
        if(abs(weight) < config.PARAMS__WEIGHT_THRESHOLD):
            weight = 0
        # TODO: Activation functions
        value += _getValueRecursive(network, neatData, prevElem) * weight
    element[1] = activationFunctions.activate(
        activationFunctions.ACTIVATION_FUNCTION__TANH,
        value)
    return element[1]


class Individual:
    neatData = {}

    def __init__(self):
        self.neatData = neat.createDataStructure(
            config.SUBSTRATE__DIMENSION*2, 1)

    def getOutput(self, input):
        '''
        The input should be 2D array of InputSize x TrialCount
        The Output will be the same as input only
        OutputSize x TrialCount dimensions
        '''
        trialCount = np.size(input, 1)
        output = np.zeros((config.SUBSTRATE__OUTPUT_SIZE, trialCount))
        if(config.SYSTEM__USE_GPU):
            # Remember this is different from the bpg in config
            blockspergrid = (trialCount +
                             (config.SYSTEM__THREADS_PER_BLOCK - 1)
                             ) // config.SYSTEM__THREADS_PER_BLOCK
            _getValueKernel[
                blockspergrid, config.SYSTEM__THREADS_PER_BLOCK](
                input,
                output,
                self.neatData[constants.NEATDATA__INNOVATION_NUMBER_INDEX],
                self.neatData[constants.NEATDATA__NODE_GENES_INDEX],
                self.neatData[constants.NEATDATA__CONNECTION_GENES_INDEX],
                self.neatData[constants.NEATDATA__INPUT_SIZE_INDEX],
                self.neatData[constants.NEATDATA__OUTPUT_SIZE_INDEX])
        else:
            raise Exception("Not implemented")
        return output

    def crossOverAndMutate(parent1, parent2):
        child = neat.crossOver(parent1, parent2)
        neat.mutate(child)
        return child
