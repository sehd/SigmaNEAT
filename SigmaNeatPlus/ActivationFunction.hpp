#pragma once
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cuda_runtime.h>

class ActivationFunction
{
public:
	enum FunctionType
	{
		TanH,
		Identity
	};
	__device__ __host__
		static double activate(FunctionType t_type, double t_input);
};

#endif // !ACTIVATION_FUNCTION_H