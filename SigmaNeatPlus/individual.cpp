#include <cuda_runtime.h>
#include <math.h>
#include "Individual.hpp"
#include "Config.hpp"

Individual::Individual()
	:m_neat(SUBSTRATE__DIMENSION * 2, 1) {

}

void Individual::getValueHost(double* t_input, double* t_output) {
	Network network = Network();
	network.input = t_input;

	network.output = t_output;
	for (int i = 0; i < SUBSTRATE__OUTPUT_SIZE; i++)
	{
		network.output[i] = nan("");
	}

	network.hidden = new double* [SUBSTRATE__LAYERS_COUNT];
	for (int h = 0; h < SUBSTRATE__LAYERS_COUNT; h++)
	{
		network.hidden[h] = new double[SUBSTRATE__LAYER_SIZE];
		for (int i = 0; i < SUBSTRATE__LAYER_SIZE; i++)
		{
			network.hidden[h][i] = nan("");
		}
	}
	for (int i = 0; i < SUBSTRATE__OUTPUT_SIZE; i++)
	{
		getValueRecursive(network, &m_neat, SUBSTRATE__LAYERS_COUNT + 1, i);
	}
}

__global__
void Individual::getValueDevice(double* t_input, double* t_output, Neat* t_neat) {
	/*
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
	*/
}

void Individual::getValueRecursive(Network t_network, Neat* t_neat, int t_layerNo, int t_itemIndex) {
	/*
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
	*/
}

double** Individual::getOutput(int t_trialCount, double** t_input) {
	double** output = new double* [t_trialCount];
	if (SYSTEM__USE_GPU) {
		//TODO
		/*
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
		*/
	}
	else {
		for (int trialIndex = 0; trialIndex < t_trialCount; trialIndex++)
		{
			getValueHost(t_input[trialIndex], output[trialIndex]);
		}
	}
	return output;
}

Individual Individual::crossOverAndMutate(Individual t_first, Individual t_second) {
	Neat childGene = Neat::crossOver(t_first.m_neat, t_second.m_neat);
	childGene.mutate();
	Individual* child = new Individual();
	(*child).m_neat = childGene;
	return *child;
}