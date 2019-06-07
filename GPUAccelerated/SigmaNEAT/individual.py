import numba.cuda as cu
from config import Config
from network import Network


class Individual:
    '''
    This class holds the logic for storing the genome, create the phenotype,
    creating final network from substrate and calculating the output of
    that network.
    '''
    _network: Network

    @cu.jit(device=True)
    def createNetwork(self):
        print(Config.substrate)

    @cu.jit(device=True)
    def getOutpu(self):
        pass
