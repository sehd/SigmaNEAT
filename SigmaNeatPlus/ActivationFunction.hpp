#pragma once
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

class ActivationFunction
{
public:
	enum FunctionType
	{
		TanH
	};
	double activate(FunctionType t_type, double t_input);
};

#endif // !ACTIVATION_FUNCTION_H