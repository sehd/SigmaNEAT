import numba.cuda as cu
from neat import NEAT
from config import Config


class Individual:
    '''
    This class holds the logic for storing the genome, create the phenotype,
    creating final network from substrate and calculating the output of
    that network.
    '''
    neat: NEAT

    def __init__(self):
        self.neat = NEAT(Config.substrate["dimension"], 1)

    @cu.jit(device=True)
    def createNetwork(self):
        pass

    @cu.jit(device=True)
    def getOutput(self, input: list):
        pass
