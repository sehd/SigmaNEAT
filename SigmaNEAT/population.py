import numba.cuda as cu
from config import Config, cudaMethod
from individual import Individual


class Population(object):
    '''
    A population is responsible to handle and control individuals.
    This includes mutation and cross-over and other GA operations.
    '''
    individuals = []

    def __init__(self):
        for i in range(Config.params["populationSize"]):
            self.individuals.append(Individual())

    @cudaMethod()
    def Run(self):
        print('yes')

    def temp(self):
        if(Config.system["useGpu"]):
            print("Running w/ GPU support")
            pos = cu.grid(1)
        else:
            print("Running w/o GPU support")
            pos = 0
        # shape = cu.gridsize(1)
        if(pos < Config.params["maxGenerationCount"]):
            # print(pos)
            print("Running generation {pos}")
            self._runGeneration()

    @cudaMethod()
    def _runGeneration(self):
        print("Generation ran")
