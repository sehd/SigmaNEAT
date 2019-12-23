#include <math.h>
#include "ActivationFunction.hpp"

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

ActivationFunction::FunctionType ActivationFunction::getFromRandom(float t_randomNumber) {
	if (t_randomNumber > 0.5)
		return ActivationFunction::Identity;
	else
		return ActivationFunction::TanH;
}