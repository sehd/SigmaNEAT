#include <math.h>
#include <cuda_runtime.h>
#include "ActivationFunction.hpp"

__device__ __host__
double ActivationFunction::activate(FunctionType t_type, double t_input) {
	switch (t_type)
	{
	case ActivationFunction::Identity:
		return t_input;
	case ActivationFunction::TanH:
		return tanh(t_input);
	default:
		return nan("");
	}
}