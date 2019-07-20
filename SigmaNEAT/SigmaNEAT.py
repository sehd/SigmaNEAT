from population import Population
from config import Config

pop = Population()
threadsperblock = 32
blockspergrid = (Config.params["maxGenerationCount"] +
                 (threadsperblock - 1)) // threadsperblock

pop.Run[blockspergrid, threadsperblock]()
