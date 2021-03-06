#include <math.h>
#include "ActivationFunction.hpp"

double ActivationFunction::activate(FunctionType t_type, double t_input) {
	switch (t_type)
	{
	case ActivationFunction::FunctionType::Identity:
		return t_input;
	case ActivationFunction::FunctionType::TanH:
		return tanh(t_input);
	case ActivationFunction::FunctionType::ReLU:
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
		return ActivationFunction::FunctionType::ReLU;
	else
		return ActivationFunction::FunctionType::TanH;
}

std::string ActivationFunction::toString(ActivationFunction::FunctionType t_type) {
	switch (t_type)
	{
	case ActivationFunction::FunctionType::Identity:
		return "Identity";
	case ActivationFunction::FunctionType::TanH:
		return "TanH";
	case ActivationFunction::FunctionType::ReLU:
		return "ReLU";
	default:
		return "Unknown";
	}
}