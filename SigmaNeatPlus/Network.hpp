#pragma once
#ifndef NETWORK_H
#define NETWORK_H

#include <cuda_runtime.h>
#include "Config.hpp"

class Network
{
public:
	double* input;
	double* output;
	double** hidden;

	__device__ __host__
		Network(double* t_input);
	__device__ __host__
		~Network();
};

#endif // !NETWORK_H

