#include <stdio.h>
#include "Population.hpp"
#include "Config.hpp"

Population::Population() {
	m_individuals = new Individual[PARAMS__POPULATION_SIZE];
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
		m_individuals[i] = Individual();
	printf_s("Population initiated.");
}

Population::~Population() {
	delete m_individuals;
}

void Population::run() {
	if (SYSTEM__USE_GPU)
		printf_s("Running. (GPU support ENABLED)");
	else
		printf_s("Running. (GPU support DISABLED)");

	int inputSize = 20000;
	double** input = new double*[inputSize];
	for (int i = 0; i < inputSize; i++)
	{
		input[i] = new double[SUBSTRATE__INPUT_SIZE];
	}
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		double** output = m_individuals[i].getOutput(inputSize, input);
		printf_s("%f", output[1][1]);
	}
}