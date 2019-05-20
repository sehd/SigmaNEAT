import numpy as np
import numba.cuda as c
from neat import NEAT

@c.jit
def GetGenerationResults(input,output):
    pos = c.grid(1)
    if(pos < input.size):
        output[pos] = 1

genes = NEAT().createGenome(3,1)
input = np.zeros(10)
output = np.zeros(10)
GetGenerationResults(input,output)
print(genes)