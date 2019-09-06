#pragma once
#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "Neat.hpp"

//This class holds the logic for storing the genome, create the phenotype,
//creating final network from substrateand calculating the output of
//that network.
class Individual
{
	Neat m_neat;
	void getValueHost(double* t_input, double* t_output, Neat* t_neat);
	void getValueDevice(double* t_input, double* t_output, Neat* t_neat);
	double getValueRecursive();
public:
	Individual();
	double getOutput(int t_inputSize, double* t_input);
	Individual crossOverAndMutate(Individual t_first, Individual t_second);
};

#endif // !INDIVIDUAL_H