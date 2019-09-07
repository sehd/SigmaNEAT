#pragma once
#ifndef NETWORK_H
#define NETWORK_H

#include "Config.hpp"

class Network
{
public:
	double* input;
	double* output;
	double** hidden;
	
	Network(double* t_input);
	~Network();
};

#endif // !NETWORK_H

