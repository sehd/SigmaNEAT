#include <cuda_runtime.h>
#include <math.h>
#include <string>
#include <iostream>
#include <exception>
#include "Individual.hpp"
#include "Config.hpp"

static bool sharedMemoryConfigured = !SYSTEM__USE_GPU;

Individual::Individual(int t_idx, int t_speciesId) :
	m_neat(SUBSTRATE__DIMENSION * 2, 1, &m_innovationNumber, SUBSTRATE__DIMENSION* t_idx, t_idx),
	m_innovationNumber(t_idx* SUBSTRATE__DIMENSION * 2 + 1),
	speciesId(t_speciesId),
	isAlive(true) {}

Individual::~Individual() {
	//m_neat.~Neat(); I don't know why but this block automatically calls the destructor of neat !?
}

__host__ __device__
double getValueRecursive(Network* t_network, Neat* t_neat, int t_layerNo, int t_itemIndex) {
	if (t_layerNo < 1)
		return t_network->input[t_itemIndex];

	if (t_layerNo < SUBSTRATE__LAYERS_COUNT + 1)
		if (!isnan(t_network->hidden[t_layerNo - 1][t_itemIndex]))
			return t_network->hidden[t_layerNo - 1][t_itemIndex];

	int prevLayerLength;
	if (t_layerNo <= 1)
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

	double result;
	if (t_layerNo < SUBSTRATE__LAYERS_COUNT + 1) {
		result = ActivationFunction::activate(ActivationFunction::FunctionType::TanH, value);
		t_network->hidden[t_layerNo - 1][t_itemIndex] = result;
	}
	else {
		result = ActivationFunction::activate(ActivationFunction::FunctionType::ReLU, value);
		t_network->output[t_itemIndex] = result;
	}

	return result;
}

__host__ __device__
void getSingleValue(double* t_input, double* t_output, Neat* t_neat) {
	Network network(t_input);

	for (int i = 0; i < SUBSTRATE__OUTPUT_SIZE; i++)
	{
		getValueRecursive(&network, t_neat, SUBSTRATE__LAYERS_COUNT + 1, i);
		t_output[i] = network.output[i];
	}
}

__global__
void getAllValuesKernel(int t_trialCount, double* t_input, double* t_output, Neat* t_neat) {
	const int trialIndex = threadIdx.x;
	if (trialIndex < t_trialCount) {
		getSingleValue(&t_input[trialIndex * SUBSTRATE__INPUT_SIZE],
			&t_output[trialIndex * SUBSTRATE__OUTPUT_SIZE], &t_neat[trialIndex]);
	}
}

double* Individual::getOutput(int t_trialCount, double* t_input) {
	double* output = new double[t_trialCount * SUBSTRATE__OUTPUT_SIZE];
	if (SYSTEM__USE_GPU) {

		if (!sharedMemoryConfigured) {
			cudaFuncSetCacheConfig(getAllValuesKernel, cudaFuncCache::cudaFuncCachePreferL1);
			sharedMemoryConfigured = true;
		}

		//Copy input to device
		double* d_input;
		cudaMalloc(&d_input, t_trialCount * SUBSTRATE__INPUT_SIZE * sizeof(double));
		cudaMemcpy(d_input, t_input, t_trialCount * SUBSTRATE__INPUT_SIZE * sizeof(double),
			cudaMemcpyHostToDevice);

		//Create empty output array on device
		double* d_output;
		cudaMalloc(&d_output, t_trialCount * SUBSTRATE__OUTPUT_SIZE * sizeof(double));

		//Copy neat to device
		Node* d_node = nullptr;
		Connection* d_connection = nullptr;
		Neat* d_neat = m_neat.copyToDevice(t_trialCount, d_node, d_connection);

		//Launch the Kernel
		int blocksPerGrid =
			(t_trialCount + (SYSTEM__THREADS_PER_BLOCK - 1))
			/ SYSTEM__THREADS_PER_BLOCK;
		int threadsPerBlock = fminl(SYSTEM__THREADS_PER_BLOCK, t_trialCount);
		getAllValuesKernel <<< blocksPerGrid, threadsPerBlock >>> (
			t_trialCount, d_input, d_output, d_neat);

		//Check if error
		cudaError_t possibleError = cudaPeekAtLastError();
		if (possibleError == cudaSuccess) {
			//Copy back the output from device
			cudaMemcpy(output, d_output, t_trialCount * SUBSTRATE__OUTPUT_SIZE * sizeof(double),
				cudaMemcpyDeviceToHost);
		}
		else {
#if LOG_ERROR
			std::cout << "Error Occured: " << cudaGetErrorString(possibleError);
#endif
			throw std::exception("Error in kernel", possibleError);
		}
		//Free memory
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_node);
		cudaFree(d_connection);
		cudaFree(d_neat);
	}
	else {
		for (int trialIndex = 0; trialIndex < t_trialCount; trialIndex++)
		{
			getSingleValue(
				&t_input[trialIndex * SUBSTRATE__INPUT_SIZE],
				&output[trialIndex * SUBSTRATE__OUTPUT_SIZE],
				&m_neat);
		}
	}
	return output;
}

void Individual::recreateAsChild(const Individual* t_first, const Individual* t_second) {
	m_neat.crossOver(&t_first->m_neat, &t_second->m_neat);
	m_neat.mutate();
	isAlive = true;
}

std::string Individual::getNeatString() {
	return m_neat.toString();
}