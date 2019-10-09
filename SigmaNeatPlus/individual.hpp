#pragma once
#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "Neat.hpp"
#include "Network.hpp"

//This class holds the logic for storing the genome, create the phenotype,
//creating final network from substrateand calculating the output of
//that network.
class Individual
{
	int m_innovationNumber;
	Neat m_neat;
public:
	bool isAlive;
	int speciesId;
	Individual(int t_speciesId = 0);

	//The input should be 2D array of InputSize x TrialCount
	//The Output will be the same as input only OutputSize x TrialCount dimensions
	double* getOutput(int t_trialCount, double* t_input);
	void recreateAsChild(const Individual* t_first, const Individual* t_second);
};

#endif // !INDIVIDUAL_H