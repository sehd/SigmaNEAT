'''
A population is responsible to handle and control individuals.
This includes mutation and cross-over and other GA operations.
'''

from config import System, Params
from individual import Individual


class Population:
    individuals = []

    def __init__(self):
        for i in range(Params.populationSize):
            self.individuals.append(Individual())
        print('Population initiated.')

    # def _kernel(rng_states):
    #     pos = cu.grid(1)
    #     # shape = cu.gridsize(1)
    #     if(pos < config.params_populationSize()):
    #         print(pos)
    #         ind = individual.createDataStructure()
    #         _runIndividual(ind)

    def Run(self):
        if(System.useGpu):
            print("Running w/ GPU support")

        else:
            print("Running w/o GPU support")
