'''
This module holds the logic for NEAT algorithm.
'''

import numpy as np
import random
import activationFunctions
from config import cudaMethod


def createDataStructure(inputSize: int, outputSize: int):
    node_arr = np.zeros((inputSize + outputSize, 2), dtype=np.float)
    for i in range(inputSize + outputSize):
        node = _createNode(i)
        node_arr[i, 0] = node[0]
        node_arr[i, 1] = node[1]

    conn_arr = np.zeros((inputSize, outputSize, 6), dtype=np.float)
    for i in range(inputSize):
        for j in range(outputSize):
            connection = _createConnection(
                i, j+inputSize, random.uniform(-1, 1),
                activationFunctions.ACTIVATION_FUNCTION__TANH, True)
            for k in range(6):
                conn_arr[i, j, k] = connection[k]
    neatData = (
        # "InnovationNumber":
        0,
        # "nodeGenes":
        node_arr,
        # "connectionGenes":
        conn_arr,
        # "_inputSize":
        inputSize,
        # "_outputSize":
        outputSize
    )
    return neatData


def _createNode(id: int, value: float = None):
    return (id, value)


def _createConnection(input: int,
                      output: int,
                      weight: float,
                      activationFunction: int,
                      enabled: bool):
    return (
        # "input":
        input,
        # "output":
        output,
        # "weight":
        weight,
        # "activationFunction":
        activationFunction,
        # "enabled":
        enabled,
        # "innovationNo":
        # TODO
        0
    )


@cudaMethod()
def _getValueRecursive(self, node):
    if(node["value"] is not None):
        return node["value"]
    res = 0
    for x in self["connectionGenes"]:
        if x["output"] == node["id"] and x["enabled"]:
            prevNodeValue = _getValueRecursive(
                self["nodeGenes"][x["input"]])
            res += prevNodeValue*x["weight"]
    node["value"] = activationFunctions.activate(
        x["activationFunction"], res)
    return node["value"]


@cudaMethod()
def mutate(self):
    pass


@cudaMethod()
def crossOver(parent1, parent2):
    pass


@cudaMethod()
def getValue(neat, input):
    if(len(input) != neat["_inputSize"]):
        raise Exception("The input size is not right")
    for node in neat["nodeGene"]:
        if(node[0] < neat["_inputSize"]):
            node[1] = input[node[id]]
        else:
            node[1] = None
    res = []
    for x in range(neat["_outputSize"]):
        res.append(neat["_getValueRecursive"](
            neat["nodeGenes"][x+neat["_inputSize"]]))
    return
