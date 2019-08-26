import numba.types as typ
import numba.cuda as cu
import neat
import constants
from config import cudaMethod
import config
import activationFunctions
import numpy as np

OPTIONAL_FLOAT = typ.optional(typ.float64).type


@cu.jit
def _getValueKernel(input, output, innovation, nodeGenes, connectionGenes,
                    inputSize, outputSize):
    trialIndex = cu.grid(1)
    neatData = (innovation, nodeGenes, connectionGenes, inputSize, outputSize)

    # Create a network based on substrate
    network_input = cu.local.array(config.SUBSTRATE__INPUT_SIZE,
                                   OPTIONAL_FLOAT)

    network_hidden = cu.local.array((config.SUBSTRATE__LAYERS_COUNT,
                                     config.SUBSTRATE__LAYER_SIZE),
                                    OPTIONAL_FLOAT)

    network_output = cu.local.array(config.SUBSTRATE__OUTPUT_SIZE,
                                    OPTIONAL_FLOAT)

    # fill the input in the network
    for i in range(config.SUBSTRATE__INPUT_SIZE):
        network_input[i] = input[trialIndex][i]

    # get value for each output node
    for i in range(outputSize):
        output[trialIndex][i] = _getValueRecursive(
            network_input, network_hidden, network_output,
            neatData,
            (1+config.SUBSTRATE__LAYERS_COUNT, i))


@cudaMethod()
def _getValueRecursive(network_input, network_hidden, network_output,
                       neatData, element):
    if(element[0] < 1):
        # We are looking at the input
        return network_input[element[1]]

    elif (element[0] > config.SUBSTRATE__LAYERS_COUNT):
        # We are looking at the output
        if(network_output[element[1]] is not None):
            return network_output[element[1]]
    else:
        # We are looking at the hidden
        if(network_hidden[element[0]-1][element[1]] is not None):
            return network_hidden[element[0]-1][element[1]]

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
    '''
    This class holds the logic for storing the genome, create the phenotype,
    creating final network from substrate and calculating the output of
    that network.
    '''
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
        trialCount = np.size(input, 0)
        output = np.zeros((trialCount, config.SUBSTRATE__OUTPUT_SIZE))
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
