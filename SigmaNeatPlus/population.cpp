#include <iostream>
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

	int inputSize = 1;
	double** input = new double*[inputSize];
	for (int i = 0; i < inputSize; i++)
	{
		input[i] = new double[SUBSTRATE__INPUT_SIZE];
		for (int j = 0; j < SUBSTRATE__INPUT_SIZE; j++)
		{
			input[i][j] = 1;
		}
	}
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		double** output = m_individuals[i].getOutput(inputSize, input);
		std::cout << output[0][0] << std::endl;
		for (int i = 0; i < inputSize; i++)
			delete[] output[i];
		delete[] output;
	}

	for (int i = 0; i < inputSize; i++)
		delete[] input[i];
	delete[] input;
}