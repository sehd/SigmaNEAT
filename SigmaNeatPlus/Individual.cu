#include <cuda_runtime.h>
#include <math.h>
#include "Individual.hpp"
#include "Config.hpp"

Individual::Individual() :
	m_neat(SUBSTRATE__DIMENSION * 2, 1, &m_innovationNumber),
	m_innovationNumber(SUBSTRATE__DIMENSION * 2 + 1) {}

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
	Network network = Network(t_input);

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
}

double** Individual::getOutput(int t_trialCount, double** t_input) {
	double** output = new double* [t_trialCount];
	if (SYSTEM__USE_GPU) {
		
		//Copy input to device
		double* d_input;
		cudaMalloc(&d_input, t_trialCount * SUBSTRATE__INPUT_SIZE * sizeof(double));
		//cudaMemcpy(d_input, t_input, );
		
		//Create empty output array on device
		double* d_output;

		//Launch the Kernel
		int blocksPerGrid =
			(t_trialCount + (SYSTEM__THREADS_PER_BLOCK - 1))
			/ SYSTEM__THREADS_PER_BLOCK;
		getAllValuesKernel <<< blocksPerGrid, SYSTEM__THREADS_PER_BLOCK >>> (
			t_trialCount, d_input, d_output, &m_neat);

		//Copy back the output from device
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