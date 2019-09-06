#pragma once
#ifndef NEAT_H
#define NEAT_H

class Neat
{
};

#endif // !NEAT_H

/*
'''
This module holds the logic for NEAT algorithm.
'''

import numba
import numba.cuda as cu
import numpy as np
import random
import activationFunctions
import constants
from config import cudaMethod


def createDataStructure(inputSize: int, outputSize: int):
	node_arr = np.zeros((inputSize + outputSize, 3), np.float)
	for i in range(inputSize + outputSize):
		node_arr[i, 0] = i
		node_arr[i, 1] = 0
		node_arr[i, 2] = 0

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


@cu.jit(numba.f8(numba.types.Tuple(
	(numba.i8,
	 numba.types.Array(numba.f8, 2, 'C'),
	 numba.types.Array(numba.f8, 2, 'C'),
	 numba.i8,
	 numba.i8)),
	numba.types.Array(numba.f8, 1, 'C')), device=True)
def _getValueRecursive(neat, node):
	if(node[2] == 1.0):
		return node[1]
	res = 0
	for conInd in range(neat[constants.NEATDATA__CONNECTION_GENES_INDEX]
						.shape[0]):
		if (neat[constants.NEATDATA__CONNECTION_GENES_INDEX][conInd]
			[constants.CONNECTION_INFO__OUTPUT_INDEX] == node[0]
				and neat[constants.NEATDATA__CONNECTION_GENES_INDEX][conInd]
				[constants.CONNECTION_INFO__ENABLED_INDEX]):

			inputIndex = int(neat[constants.NEATDATA__CONNECTION_GENES_INDEX]
							 [conInd, constants.CONNECTION_INFO__INPUT_INDEX])
			prevNodeValue = 0
			# _getValueRecursive(
			#     neat,
			#     neat[constants.NEATDATA__NODE_GENES_INDEX][inputIndex])
			res += (prevNodeValue *
					neat[constants.NEATDATA__CONNECTION_GENES_INDEX]
					[conInd, constants.CONNECTION_INFO__WEIGHT_INDEX])

	node[1] = activationFunctions.activate(
		neat[constants.NEATDATA__CONNECTION_GENES_INDEX]
		[conInd, constants.CONNECTION_INFO__ACTIVATIONFUNCTION_INDEX], res)
	node[2] = 1.0
	return node[1]


@cudaMethod()
def mutate(val1, val2):
	return 1.0


@cudaMethod()
def crossOver(parent1, parent2):
	pass


@cudaMethod()
def getValue(neat, input, output):
	if(len(input) != neat[constants.NEATDATA__INPUT_SIZE_INDEX]):
		raise Exception("The input size is not right")
	for nodeIndex in range(neat[constants.NEATDATA__NODE_GENES_INDEX]
						   .shape[0]):
		node = neat[constants.NEATDATA__NODE_GENES_INDEX][nodeIndex]
		if(node[0] < neat[constants.NEATDATA__INPUT_SIZE_INDEX]):
			node[1] = input[int(node[0])]
			node[2] = 1.0
		else:
			node[1] = np.nan
			node[2] = 0.0
	for i in range(neat[constants.NEATDATA__OUTPUT_SIZE_INDEX]):
		output[i] = _getValueRecursive(
			neat, neat[constants.NEATDATA__NODE_GENES_INDEX]
			[neat[constants.NEATDATA__INPUT_SIZE_INDEX]+i])

*/