#pragma once
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cuda_runtime.h>

class ActivationFunction
{
public:
	enum FunctionType //TODO complete function list. Don't forget "getFromRandom"
	{
		TanH,
		ReLU,
		Identity
	};
	__device__ __host__
		static double activate(FunctionType t_type, double t_input);

	static FunctionType getFromRandom(float t_randomNumber);
};

#endif // !ACTIVATION_FUNCTION_H