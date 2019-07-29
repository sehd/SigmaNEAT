from population import Population
from config import Config, cudaMethod


@cudaMethod(isDevice=False)
def StartCuda(pop: Population):
    pop.Run()


def StartNoCuda():
    pop = Population()
    pop.Run()


if(Config.system["useGpu"]):
    threadsperblock = 32
    blockspergrid = (Config.params["maxGenerationCount"] +
                     (threadsperblock - 1)) // threadsperblock
    print('Entering kernel')
    pop = Population()
    StartCuda[blockspergrid, threadsperblock]()
else:
    StartNoCuda()
