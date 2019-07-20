import numba.cuda as cu
from config import Config
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

    @cu.jit
    def Run(self):
        pos = cu.grid(1)
        # shape = cu.gridsize(1)
        if(pos < Config.params["maxGenerationCount"]):
            # print(pos)
            print("Running generation {pos}")
            self._runGeneration()

    @cu.jit(device=True)
    def _runGeneration():
        print("Generation ran")
