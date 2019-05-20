import numpy as np
import random


class NEAT:
    _innovationNumber = 0

    def getInnovation(self):
        self._innovationNumber += 1
        return self._innovationNumber

    def createGenome(self, inputSize: int, outputSize: int):
        nodeGenes = np.array(range(inputSize + outputSize), int)
        connectionGenes = np.empty
        for i in range(inputSize):
            for j in range(outputSize):
                np.append(connectionGenes,
                          (i, j, random.uniform(-1, 1), True, self.getInnovation()))
        return (nodeGenes, connectionGenes)
        
