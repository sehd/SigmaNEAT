import numpy as np
import numba.cuda as c
import neat

@c.jit
def GetGenerationResults(input,output):
    pos = c.grid(1)
    if(pos < input.size):
        output[pos] = 1

genes = neat.createGenome(3,1)
input = np.zeros(10)
output = np.zeros(10)
GetGenerationResults(input,output)
print(genes)