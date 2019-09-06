#pragma once
#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

class Individual
{
};

#endif // !INDIVIDUAL_H

/*
import numba.cuda as cu
import neat
import constants
from config import cudaMethod
import config
import activationFunctions
import numpy as np


@cu.jit
def _getValueKernel(input, output, innovation, nodeGenes, connectionGenes,
					inputSize, outputSize):
	trialIndex = cu.grid(1)
	if(trialIndex < input.shape[0]):
		neatData = (innovation, nodeGenes,
					connectionGenes, inputSize, outputSize)

		# Create a network based on substrate
		network_input = cu.local.array(config.SUBSTRATE__INPUT_SIZE,
									   constants.OPTIONAL_FLOAT)

		network_hidden = cu.local.array((config.SUBSTRATE__LAYERS_COUNT,
										 config.SUBSTRATE__LAYER_SIZE),
										constants.OPTIONAL_FLOAT)

		network_output = cu.local.array(config.SUBSTRATE__OUTPUT_SIZE,
										constants.OPTIONAL_FLOAT)

		# fill the input in the network
		for i in range(config.SUBSTRATE__INPUT_SIZE):
			network_input[i] = input[trialIndex][i]

		# get value for each output node
		for i in range(outputSize):
			output[trialIndex][i] = _getValueRecursive(
				network_input, network_hidden, network_output,
				neatData,
				(1+config.SUBSTRATE__LAYERS_COUNT, i))


@cudaMethod()
def _getValueRecursive(network_input, network_hidden, network_output,
					   neatData, element):
	if(element[0] < 1):
		# We are looking at the input
		return network_input[element[1]]

	elif (element[0] <= config.SUBSTRATE__LAYERS_COUNT):
		# We are looking at the hidden
		if(network_hidden[element[0]-1][element[1]] is not None):
			return network_hidden[element[0]-1][element[1]]

	prevIndices = range(config.SUBSTRATE__INPUT_SIZE) \
		if element[0]-1 < 1 \
		else range(len(network_hidden[element[0]-2][:]))
	value = 0
	for prevElem in prevIndices:
		if(config.SYSTEM__USE_GPU):
			weight = cu.local.array(1, constants.OPTIONAL_FLOAT)
		else:
			# weight = [None]
			pass
		neat.getValue(neatData,
					  (element[0]-1, prevElem, element[0], element[1]),
					  weight)
		if(abs(weight[0]) < config.PARAMS__WEIGHT_THRESHOLD):
			weight[0] = 0
		# TODO: Activation functions
		value += 0
		# _getValueRecursive(network_input,
		#                             network_hidden,
		#                             network_output,
		#                             neatData,
		#                             (element[0]-1, prevElem)) * weight[0]
	result = activationFunctions.activate(
		activationFunctions.ACTIVATION_FUNCTION__TANH,
		value)
	if(element[0] <= config.SUBSTRATE__LAYERS_COUNT):
		network_hidden[element[0]-1][element[1]] = result
	else:
		network_output[element[1]] = result
	return result


class Individual:
	'''
	This class holds the logic for storing the genome, create the phenotype,
	creating final network from substrate and calculating the output of
	that network.
	'''
	neatData = {}

	def __init__(self):
		self.neatData = neat.createDataStructure(
			config.SUBSTRATE__DIMENSION*2, 1)

	def getOutput(self, input):
		'''
		The input should be 2D array of InputSize x TrialCount
		The Output will be the same as input only
		OutputSize x TrialCount dimensions
		'''
		trialCount = np.size(input, 0)
		output = np.zeros((trialCount, config.SUBSTRATE__OUTPUT_SIZE))
		if(config.SYSTEM__USE_GPU):
			# Remember this is different from the bpg in config
			blockspergrid = (trialCount +
							 (config.SYSTEM__THREADS_PER_BLOCK - 1)
							 ) // config.SYSTEM__THREADS_PER_BLOCK
			_getValueKernel[
				blockspergrid, config.SYSTEM__THREADS_PER_BLOCK](
				input,
				output,
				self.neatData[constants.NEATDATA__INNOVATION_NUMBER_INDEX],
				self.neatData[constants.NEATDATA__NODE_GENES_INDEX],
				self.neatData[constants.NEATDATA__CONNECTION_GENES_INDEX],
				self.neatData[constants.NEATDATA__INPUT_SIZE_INDEX],
				self.neatData[constants.NEATDATA__OUTPUT_SIZE_INDEX])
		else:
			self.getValue(input, output)
		return output

	def getValue(self, input, output):
		for trialIndex in range(input.shape[0]):
			# Create a network based on substrate
			network_input = np.empty(config.SUBSTRATE__INPUT_SIZE,
									 np.float_)

			network_hidden = np.empty((config.SUBSTRATE__LAYERS_COUNT,
									   config.SUBSTRATE__LAYER_SIZE),
									  np.float_)

			network_output = np.empty(config.SUBSTRATE__OUTPUT_SIZE,
									  np.float_)

			# fill the input in the network
			for i in range(config.SUBSTRATE__INPUT_SIZE):
				network_input[i] = input[trialIndex][i]

			# get value for each output node
			for i in range(self.neatData[
					constants.NEATDATA__OUTPUT_SIZE_INDEX]):
				output[trialIndex][i] = _getValueRecursive(
					network_input, network_hidden, network_output,
					self.neatData,
					(1+config.SUBSTRATE__LAYERS_COUNT, i))

	def crossOverAndMutate(parent1, parent2):
		child = neat.crossOver(parent1, parent2)
		neat.mutate(child)
		return child

*/

