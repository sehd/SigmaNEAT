#include <cuda_runtime.h>
#include <math.h>
#include "Network.hpp"

__host__ __device__
Network::Network(double* t_input) :
	input(t_input) {
	output = new double[SUBSTRATE__OUTPUT_SIZE];
	for (int i = 0; i < SUBSTRATE__OUTPUT_SIZE; i++)
	{
		output[i] = nan("");
	}

	hidden = new double* [SUBSTRATE__LAYERS_COUNT];
	for (int h = 0; h < SUBSTRATE__LAYERS_COUNT; h++)
	{
		hidden[h] = new double[SUBSTRATE__LAYER_SIZE];
		for (int i = 0; i < SUBSTRATE__LAYER_SIZE; i++)
		{
			hidden[h][i] = nan("");
		}
	}
}

__host__ __device__
Network::~Network() {
	delete[] input;
	delete[] output;
	for (int h = 0; h < SUBSTRATE__LAYERS_COUNT; h++)
		delete[] hidden[h];
	delete[] hidden;
}