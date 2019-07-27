from population import Population
from config import Config

pop = Population()
threadsperblock = 32
blockspergrid = (Config.params["maxGenerationCount"] +
                 (threadsperblock - 1)) // threadsperblock

if(Config.system["useGpu"]):
    pop.Run[blockspergrid, threadsperblock]()
else:
    pop.Run()
