'''
A population is responsible to handle and control individuals.
This includes mutation and cross-over and other GA operations.
'''

import numba.cuda as cu
from numba.cuda.random import create_xoroshiro128p_states
import config
import individual


@config.cudaMethod()
def _runIndividual(individual):
    print("Individual ran")


@cu.jit
def _kernel(rng_states):
    pos = cu.grid(1)
    # shape = cu.gridsize(1)
    if(pos < config.params_populationSize()):
        print(pos)
        ind = individual.createDataStructure()
        _runIndividual(ind)


def _host():
    for i in range(config.params_populationSize()):
        ind = individual.createDataStructure()
        _runIndividual(ind)


def Run():
    if(config.system_useGpu()):
        print("Running w/ GPU support")
        threadsperblock = 32
        blockspergrid = (config.system_maxGenerationCount() +
                         (threadsperblock - 1)) // threadsperblock
        rng_states = create_xoroshiro128p_states(
            threadsperblock * blockspergrid, seed=1)
        _kernel[blockspergrid, threadsperblock](rng_states)
    else:
        print("Running w/o GPU support")
        _host()
