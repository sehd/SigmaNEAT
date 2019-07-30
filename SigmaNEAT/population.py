'''
A population is responsible to handle and control individuals.
This includes mutation and cross-over and other GA operations.
'''

import numba.cuda as cu
from config import system, getConfigs, cudaMethod
import individual


@cudaMethod()
def _runIndividual(individual):
    print("Individual ran")


# @cudaMethod(isDevice=False)
@cu.jit
def _kernel():
    config = getConfigs()
    pos = cu.grid(1)
    # shape = cu.gridsize(1)
    if(pos < config["params"]["populationSize"]):
        # print(pos)
        print("Running individual {0}".format(pos))
        ind = individual.createDataStructure()
        _runIndividual(ind)


def _host():
    for i in range(getConfigs()["params"]["populationSize"]):
        ind = individual.createDataStructure()
        _runIndividual(ind)


def Run():
    if(system["useGpu"]):
        print("Running w/ GPU support")
        threadsperblock = 32
        blockspergrid = (system["maxGenerationCount"] +
                         (threadsperblock - 1)) // threadsperblock
        _kernel[blockspergrid, threadsperblock]()
    else:
        print("Running w/o GPU support")
        _host()
