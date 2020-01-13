#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "config.hpp"

class RandomHelper {
	curandState_t* m_curandState;
	int m_seed, m_threadIndex;
	bool m_isInitialized;
	std::random_device m_randomDevice;
	__device__
		float getRandomGpu();
public:
#if USE_GPU
	__device__
#endif
		RandomHelper(int t_seed, int t_threadIndex);

#if USE_GPU
	__device__
#endif
		float getRandom();
	float getRandomCpu();
};

#endif // !RANDOM_H
