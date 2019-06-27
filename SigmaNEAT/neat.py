import numpy as np
import random
from activationFunctions import ActivationFunctions, activate


class NEAT(object):
    '''
    This class holds the logic for NEAT algorithm.
    '''
    # TODO: Activations
    InnovationNumber = 0
    nodeGenes = np.empty
    connectionGenes = np.empty

    _inputSize: int
    _outputSize: int

    def __init__(self, inputSize: int, outputSize: int):
        self._inputSize = inputSize
        self._outputSize = outputSize
        self.nodeGenes = np.array(
            (self._createNode(x) for x in range(inputSize + outputSize)), int)
        for i in range(inputSize):
            for j in range(outputSize):
                np.append(self.connectionGenes, self._createConnection(
                    i, j+inputSize, random.uniform(-1, 1),
                    ActivationFunctions.TanH, True))

    def _createNode(self, id: int, value: float = None):
        return
        {
            "id": id,
            "value": value
        }

    def _createConnection(self,
                          input: int,
                          output: int,
                          weight: float,
                          activationFunction: ActivationFunctions,
                          enabled: bool):
        NEAT.InnovationNumber += 1
        return
        {
            "input": input,
            "output": output,
            "weight": weight,
            "activationFunction": activationFunction,
            "enabled": enabled,
            "innovationNo": NEAT.InnovationNumber
        }

    def _getValueRecursive(self, node):
        if(node["value"] is not None):
            return node["value"]
        res = 0
        for x in self.connectionGenes:
            if x["output"] == node["id"] and x["enabled"]:
                prevNodeValue = self._getValueRecursive(
                    self.nodeGenes[x["input"]])
                res += prevNodeValue*x["weight"]
        node["value"] = activate(
            x["activationFunction"], res)
        return node["value"]

    def mutate(self):
        pass

    @staticmethod
    def crossOver(parent1, parent2):
        pass

    def getValue(self, input):
        if(len(input) != self._inputSize):
            raise Exception("The input size is not right")
        for node in self.nodeGenes:
            if(node["id"] < self._inputSize):
                node["value"] = input[node[id]]
            else:
                node["value"] = None
        res = []
        for x in range(self._outputSize):
            res.append(self._getValueRecursive(
                self.nodeGenes[x+self._inputSize]))
        return
