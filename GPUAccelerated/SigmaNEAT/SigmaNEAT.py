# from individual import Individual
import numpy as np
import numba.cuda as cu
from neat import NEAT
from config import Config

print(Config.substrate["nodes"][:])


@cu.jit
def GetGenerationResults(input):
    pos = cu.grid(1)
    # shape = cu.gridsize(1)
    if(pos < input.size):
        # print(pos)
        input[pos] = add_one(input[pos])


@cu.jit(device=True)
def add_one(stuff):
    return stuff+1


genes = NEAT().createGenome(3, 1)
input = np.zeros(10)
output = np.zeros(10)

threadsperblock = 32
blockspergrid = (input.size + (threadsperblock - 1)) // threadsperblock

GetGenerationResults[blockspergrid, threadsperblock](input)
print(input)
