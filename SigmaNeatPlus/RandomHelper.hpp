#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include <cuda_runtime.h>

class RandomHelper {
public:
	__host__ __device__
		float getRandom();
};

#endif // !RANDOM_H
