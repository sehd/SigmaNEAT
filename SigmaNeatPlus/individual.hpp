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
	Individual(int t_idx, int t_speciesId = 0);
	~Individual();

	//The input should be 2D array of InputSize x TrialCount
	//The Output will be the same as input only OutputSize x TrialCount dimensions
	double* getOutput(int t_trialCount, double* t_input);

	//First individual is considered as the more fit one
	void recreateAsChild(const Individual* t_first, const Individual* t_second);

	std::string getNeatString();
};

#endif // !INDIVIDUAL_H