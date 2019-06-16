import numba.cuda as cu
from neat import NEAT
from config import Config
from activationFunctions import ActivationFunctions, activate


class Individual:
    '''
    This class holds the logic for storing the genome, create the phenotype,
    creating final network from substrate and calculating the output of
    that network.
    '''
    neat: NEAT

    def __init__(self):
        self.neat = NEAT(Config.substrate["dimension"]*2, 1)

    @cu.jit(device=True)
    def _createNetwork(self):
        res = []
        for layer in Config.substrate["nodes"]:
            res.append([[x, None] for x in layer])
        return res

    @cu.jit(device=True)
    def _getValueRecursive(self, element: list, network: list):
        if(network[element[0]][element[1]][1] is not None):
            return network[element[0]][element[1]][1]
        value = 0
        for prevElem in network[element[0]-1][:]:
            weight = self.neat.getValue(prevElem+element)
            if(abs(weight)<Config.params["weightThreshold"]):
                weight=0
            # TODO: Activation functions
            value += self._getValueRecursive(prevElem, network) * weight
        element[1] = activate(
                ActivationFunctions.TanH,
                value)
        return element[1]

    @cu.jit(device=True)
    def getOutput(self, input: list):
        network=self._createNetwork()
        for i in len(network[0]):
            network[0][i][1]=input[i]
        res=[]
        for outputNode in network[-1]:
            res.append(self._getValueRecursive(outputNode,network))
        return res

    @cu.jit(device=True)
    def crossOverAndMutate(self,parent1,parent2):
        self.neat=NEAT.crossOver(parent1,parent2)
        self.neat.mutate()
