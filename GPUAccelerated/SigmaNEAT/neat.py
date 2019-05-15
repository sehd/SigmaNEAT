import numpy as np
import random

innovationNumber = 0

def getInnovation():
    innovationNumber+=1
    return innovationNumber

def createGenome(inputSize:int, outputSize:int):
    nodeGenes = np.array(range(inputSize + outputSize),int)
    connectionGenes = np.empty
    for i in range(inputSize):
        for j in range(outputSize):
            np.append(connectionGenes,(i,j,random.uniform(-1,1),True,getInnovation()))
    return (nodeGenes,connectionGenes)

