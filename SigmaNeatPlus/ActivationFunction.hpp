#pragma once
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cuda_runtime.h>
#include <string>

class ActivationFunction
{
public:
	//TODO complete function list. 
	// Don't forget "getFromRandom"
	// Don't forget "toString"
	enum class FunctionType 
	{
		TanH,
		ReLU,
		Identity
	};
	__device__ __host__
		static double activate(FunctionType t_type, double t_input);

	__device__ __host__
		static FunctionType getFromRandom(float t_randomNumber);

	__host__
		static std::string toString(FunctionType t_type);
};

#endif // !ACTIVATION_FUNCTION_H