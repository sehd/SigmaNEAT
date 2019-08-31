'''
This module holds the logic for NEAT algorithm.
'''

import numpy as np
import random
import activationFunctions
import constants
from config import cudaMethod


def createDataStructure(inputSize: int, outputSize: int):
    node_arr = np.zeros((inputSize + outputSize, 2), np.float)
    for i in range(inputSize + outputSize):
        node = _createNode(i)
        node_arr[i, 0] = node[0]
        node_arr[i, 1] = node[1]

    conn_arr = np.zeros((inputSize * outputSize, 6), np.float)
    for i in range(inputSize):
        for j in range(outputSize):
            connection = _createConnection(
                i, j+inputSize, random.uniform(-1, 1),
                activationFunctions.ACTIVATION_FUNCTION__TANH, True)
            for k in range(6):
                conn_arr[(i*outputSize) + j, k] = connection[k]
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
def _getValueRecursive(neat, node):
    if(not np.isnan(node)[1]):
        return node[1]
    res = 0
    for x in neat[constants.NEATDATA__CONNECTION_GENES_INDEX]:
        if (x[constants.CONNECTION_INFO__OUTPUT_INDEX] == node[0]
                and x[constants.CONNECTION_INFO__ENABLED_INDEX]):

            inputIndex = int(x[constants.CONNECTION_INFO__INPUT_INDEX])
            prevNodeValue = _getValueRecursive(
                neat,
                neat[constants.NEATDATA__NODE_GENES_INDEX][inputIndex])
            res += prevNodeValue*x[constants.CONNECTION_INFO__WEIGHT_INDEX]

    node[1] = activationFunctions.activate(
        x[constants.CONNECTION_INFO__ACTIVATIONFUNCTION_INDEX], res)
    return node[1]


@cudaMethod()
def mutate(self):
    pass


@cudaMethod()
def crossOver(parent1, parent2):
    pass


@cudaMethod()
def getValue(neat, input, output):
    if(len(input) != neat[constants.NEATDATA__INPUT_SIZE_INDEX]):
        raise Exception("The input size is not right")
    for node in neat[constants.NEATDATA__NODE_GENES_INDEX]:
        if(node[0] < neat[constants.NEATDATA__INPUT_SIZE_INDEX]):
            node[1] = input[int(node[0])]
        else:
            node[1] = np.nan
    for i in range(neat[constants.NEATDATA__OUTPUT_SIZE_INDEX]):
        output[i] = _getValueRecursive(
            neat, neat[constants.NEATDATA__NODE_GENES_INDEX]
            [neat[constants.NEATDATA__INPUT_SIZE_INDEX]+i])
