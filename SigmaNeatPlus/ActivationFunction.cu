#include <math.h>
#include "ActivationFunction.hpp"

double ActivationFunction::activate(FunctionType t_type, double t_input) {
	switch (t_type)
	{
	case ActivationFunction::FunctionType::Identity:
		return t_input;
	case ActivationFunction::FunctionType::TanH:
		return tanh(t_input);
	default:
		return nan("");
	}
}

ActivationFunction::FunctionType ActivationFunction::getFromRandom(float t_randomNumber) {
	if (t_randomNumber > 0.5)
		return ActivationFunction::FunctionType::Identity;
	else
		return ActivationFunction::FunctionType::TanH;
}