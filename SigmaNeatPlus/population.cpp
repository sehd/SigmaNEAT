#include <iostream>
#include <math.h>
#include "Population.hpp"
#include "Config.hpp"

bool verbose;

Population::Population(bool t_verbose) {
	verbose = t_verbose;
	m_individuals = new Individual[PARAMS__POPULATION_SIZE];
	if (verbose)
		std::cout << "Population initiated." << std::endl;
}

Population::~Population() {
	delete[] m_individuals;
}

void Population::run() {
	if (verbose) {
		if (SYSTEM__USE_GPU)
			std::cout << "Running. (GPU support ENABLED)" << std::endl;
		else
			std::cout << "Running. (GPU support DISABLED)" << std::endl;
	}

	int trialCount = 10000;
	double* input = new double[trialCount * SUBSTRATE__INPUT_SIZE];
	for (int i = 0; i < trialCount; i++)
		for (int j = 0; j < SUBSTRATE__INPUT_SIZE; j++)
			input[i * SUBSTRATE__INPUT_SIZE + j] = 1;//(double)(i * i) + j * j;

	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		double* output = m_individuals[i].getOutput(trialCount, input);

		//Temp representation of data
		if (verbose) {
			std::cout << "Result for individual " << i << ": ";
			for (int j = 0; j < fminl(trialCount, 10); j++)
			{
				std::cout << "(";
				for (int k = 0; k < fminl(SUBSTRATE__OUTPUT_SIZE, 3); k++)
					std::cout << output[j * SUBSTRATE__OUTPUT_SIZE + k] << ", ";
				std::cout << (SUBSTRATE__OUTPUT_SIZE > 3 ? "..." : "") << ") ";
			}
			std::cout << (trialCount > 10 ? "..." : "") << std::endl;
		}

		delete[] output;
	}
	delete[] input;
}