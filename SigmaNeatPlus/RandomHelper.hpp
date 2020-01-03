#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

class RandomHelper {
	curandState_t* m_curandState;
	__host__ __device__
		float getRandomGpu();
public:
	RandomHelper();

	__host__ __device__
		float getRandom();
};

#endif // !RANDOM_H
