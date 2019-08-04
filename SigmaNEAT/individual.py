'''
This class holds the logic for storing the genome, create the phenotype,
creating final network from substrate and calculating the output of
that network.
'''

# import numba.cuda as cu
import neat
import config
from activationFunctions import ActivationFunctions, activate


@config.cudaMethod()
def createDataStructure():
    return neat.createDataStructure(
        config.substrate_dimension()*2, 1)


@config.cudaMethod()
def _createNetwork():
    res = []
    for layer in config.substrate_nodes():
        res.append([[x, None] for x in layer])
    return res


@config.cudaMethod()
def _getValueRecursive(data, element: list, network: list):
    if(network[element[0]][element[1]][1] is not None):
        return network[element[0]][element[1]][1]
    value = 0
    for prevElem in network[element[0]-1][:]:
        weight = neat.getValue(data, prevElem+element)
        if(abs(weight) < config.params_weightThreshold()):
            weight = 0
        # TODO: Activation functions
        value += _getValueRecursive(data, prevElem, network) * weight
    element[1] = activate(
        ActivationFunctions.TanH,
        value)
    return element[1]


@config.cudaMethod()
def getOutput(data, input: list):
    network = _createNetwork()
    for i in len(network[0]):
        network[0][i][1] = input[i]
    res = []
    for outputNode in network[-1]:
        res.append(_getValueRecursive(data, outputNode, network))
    return res


@config.cudaMethod()
def crossOverAndMutate(parent1, parent2):
    child = neat.crossOver(parent1, parent2)
    neat.mutate(child)
    return child
