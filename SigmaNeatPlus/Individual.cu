#include <cuda_runtime.h>
#include <math.h>
#include "Individual.hpp"
#include "Config.hpp"

Individual::Individual()
	:m_neat(SUBSTRATE__DIMENSION * 2, 1, &m_innovationNumber) {
	m_innovationNumber = SUBSTRATE__DIMENSION * 2 + 1;
}

__host__ __device__
double getValueRecursive(Network t_network, Neat* t_neat, int t_layerNo, int t_itemIndex) {
	if (t_layerNo < 1)
		return t_network.input[t_itemIndex];

	if (t_layerNo < SUBSTRATE__LAYERS_COUNT + 1)
		if (!isnan(t_network.hidden[t_layerNo - 1][t_itemIndex]))
			return t_network.hidden[t_layerNo - 1][t_itemIndex];

	int prevLayerLength;
	if (t_layerNo < 1)
		prevLayerLength = SUBSTRATE__INPUT_SIZE;
	else
		prevLayerLength = SUBSTRATE__LAYER_SIZE;
	double value = 0;

	double input[SUBSTRATE__DIMENSION * 2];
	double weight[1];

	for (int prevLayerItemIndex = 0; prevLayerItemIndex < prevLayerLength; prevLayerItemIndex++)
	{
		input[0] = (double)t_layerNo - 1;
		input[1] = (double)prevLayerItemIndex;
		input[2] = (double)t_layerNo;
		input[3] = (double)t_itemIndex;

		t_neat->getValue(input, weight);
		if (weight[0] < PARAMS__WEIGHT_THRESHOLD)
			weight[0] = 0;
		value += getValueRecursive(t_network, t_neat, t_layerNo - 1, prevLayerItemIndex) * weight[0];
	}
	// TODO: Activation functions
	double result = ActivationFunction::activate(ActivationFunction::Identity, value);
	if (t_layerNo < SUBSTRATE__LAYERS_COUNT + 1)
		t_network.hidden[t_layerNo - 1][t_itemIndex] = result;
	else
		t_network.output[t_itemIndex] = result;

	return result;
}

__host__ __device__
void getSingleValue(double* t_input, double* t_output, Neat* t_neat) {
	Network network = Network();
	network.input = t_input;

	network.output = new double[SUBSTRATE__OUTPUT_SIZE];
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
		getValueRecursive(network, t_neat, SUBSTRATE__LAYERS_COUNT + 1, i);
		t_output[i] = network.output[i];
	}
}

__global__
void getAllValuesKernel(int t_trialCount, double** t_input, double** t_output, Neat* t_neat) {
	const int trialIndex = threadIdx.x;
	if (trialIndex < t_trialCount) {
		getSingleValue(t_input[trialIndex], t_output[trialIndex], t_neat);
	}
	/*
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

double** Individual::getOutput(int t_trialCount, double** t_input) {
	double** output = new double* [t_trialCount];
	if (SYSTEM__USE_GPU) {
		int blocksPerGrid =
			(t_trialCount + (SYSTEM__THREADS_PER_BLOCK - 1))
			/ SYSTEM__THREADS_PER_BLOCK;
		getAllValuesKernel <<< blocksPerGrid, SYSTEM__THREADS_PER_BLOCK >>> (
			t_trialCount, t_input, output, &m_neat);
	}
	else {
		for (int trialIndex = 0; trialIndex < t_trialCount; trialIndex++)
		{
			output[trialIndex] = new double[SUBSTRATE__OUTPUT_SIZE];
			getSingleValue(t_input[trialIndex], output[trialIndex], &m_neat);
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