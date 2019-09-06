#include "ActivationFunction.hpp"
#include <math.h>
#include <stdexcept>

double ActivationFunction::activate(FunctionType t_type, double t_input) {
	switch (t_type)
	{
	case ActivationFunction::TanH:
		return tanh(t_input);
	default:
		throw std::invalid_argument("Invalid activation function type");
	}
}