'''
A population is responsible to handle and control individuals.
This includes mutation and cross-over and other GA operations.
'''

import config
from individual import Individual
import numpy as np


class Population:
    individuals = []

    def __init__(self):
        for i in range(config.PARAMS__POPULATION_SIZE):
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
        if(config.SYSTEM__USE_GPU):
            print("Running w/ GPU support")
        else:
            print("Running w/o GPU support")
        input = np.zeros((2000, config.SUBSTRATE__INPUT_SIZE))
        for individual in self.individuals:
            output = individual.getOutput(input)
            print("len = "+str(len(output)))
            print("sum = "+str(sum(output)))
