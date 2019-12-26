#include <math.h>
#include "ActivationFunction.hpp"

double ActivationFunction::activate(FunctionType t_type, double t_input) {
	switch (t_type)
	{
	case ActivationFunction::Identity:
		return t_input;
	case ActivationFunction::TanH:
		return tanh(t_input);
	case ActivationFunction::ReLU:
		if (t_input < 0)
			return 0;
		else
			return t_input;
	default:
		return nan("");
	}
}

ActivationFunction::FunctionType ActivationFunction::getFromRandom(float t_randomNumber) {
	if (t_randomNumber > 0.5)
		return ActivationFunction::ReLU;
	else
		return ActivationFunction::TanH;
}