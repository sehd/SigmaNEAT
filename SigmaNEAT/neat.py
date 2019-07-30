'''
This module holds the logic for NEAT algorithm.
'''

import numpy as np
import random
from activationFunctions import ActivationFunctions, activate
from config import getInnovationNumber


def createDataStructure(inputSize: int, outputSize: int):
    arr_like = [_createNode(x) for x in range(inputSize + outputSize)]
    neatData = {
        # TODO: Activations
        "InnovationNumber": 0,
        "nodeGenes": np.array(arr_like),
        "connectionGenes": np.empty,
        "_inputSize": inputSize,
        "_outputSize": outputSize,
    }
    for i in range(inputSize):
        for j in range(outputSize):
            np.append(neatData["connectionGenes"], _createConnection(
                i, j+inputSize, random.uniform(-1, 1),
                ActivationFunctions.TanH, True))
    return neatData


def _createNode(id: int, value: float = None):
    return {
        "id": id,
        "value": value
    }


def _createConnection(input: int,
                      output: int,
                      weight: float,
                      activationFunction: ActivationFunctions,
                      enabled: bool):
    return
    {
        "input": input,
        "output": output,
        "weight": weight,
        "activationFunction": activationFunction,
        "enabled": enabled,
        "innovationNo": getInnovationNumber()
    }


def _getValueRecursive(self, node):
    if(node["value"] is not None):
        return node["value"]
    res = 0
    for x in self["connectionGenes"]:
        if x["output"] == node["id"] and x["enabled"]:
            prevNodeValue = _getValueRecursive(
                self["nodeGenes"][x["input"]])
            res += prevNodeValue*x["weight"]
    node["value"] = activate(
        x["activationFunction"], res)
    return node["value"]


def mutate(self):
    pass


def crossOver(parent1, parent2):
    pass


def getValue(self, input):
    if(len(input) != self["_inputSize"]):
        raise Exception("The input size is not right")
    for node in self["nodeGene"]:
        if(node["id"] < self["_inputSize"]):
            node["value"] = input[node[id]]
        else:
            node["value"] = None
    res = []
    for x in range(self["_outputSize"]):
        res.append(self["_getValueRecursive"](
            self["nodeGenes"][x+self["_inputSize"]]))
    return
