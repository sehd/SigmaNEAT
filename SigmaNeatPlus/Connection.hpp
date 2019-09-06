#pragma once
#ifndef CONNECTION_H
#define CONNECTION_H

#include "ActivationFunction.hpp"

struct Connection {
	int input;
	int output;
	double weigh;
	ActivationFunction::FunctionType activationFunction;
	bool enabled;
	int innovationNo;
};

#endif // !CONNECTION_H
