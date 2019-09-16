#include <iostream>
#include <math.h>
#include "Population.hpp"
#include "Config.hpp"

Population::Population() {
	m_individuals = new Individual[PARAMS__POPULATION_SIZE];
	std::cout << "Population initiated." << std::endl;
}

Population::~Population() {
	delete[] m_individuals;
}

void Population::run() {
	if (SYSTEM__USE_GPU)
		std::cout << "Running. (GPU support ENABLED)" << std::endl;
	else
		std::cout << "Running. (GPU support DISABLED)" << std::endl;

	int trialCount = 10000;
	double** input = new double* [trialCount];
	for (int i = 0; i < trialCount; i++)
	{
		input[i] = new double[SUBSTRATE__INPUT_SIZE];
		for (int j = 0; j < SUBSTRATE__INPUT_SIZE; j++)
		{
			input[i][j] = 1;
		}
	}
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		double** output = m_individuals[i].getOutput(trialCount, input);

		//Temp representation of data
		std::cout << "Result for individual " << i << ": ";
		for (int j = 0; j < fminl(trialCount, 10); j++)
		{
			std::cout << "(";
			for (int k = 0; k < fminl(SUBSTRATE__OUTPUT_SIZE, 3); k++)
				std::cout << output[j][k] << ", ";
			std::cout << (SUBSTRATE__OUTPUT_SIZE > 3 ? "..." : "") << ") ";
		}
		std::cout << (trialCount > 10 ? "..." : "") << std::endl;

		if (SYSTEM__USE_GPU) //The whole array of arrays was allocated once
			delete[] output[0];
		else
			for (int j = 0; j < trialCount; j++)
				delete[] output[j];

		delete[] output;
	}

	for (int i = 0; i < trialCount; i++)
		delete[] input[i];
	delete[] input;
}