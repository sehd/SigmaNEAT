#pragma once
#ifndef NODE_H
#define NODE_H

#include "ActivationFunction.hpp"

struct Node
{
	int id;
	double value;
	bool hasValue;
	ActivationFunction::FunctionType activationFunction;
};

#endif // !NODE_H
