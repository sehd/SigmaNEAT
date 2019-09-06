#include "Population.hpp"
#include "Config.hpp"
#include <stdio.h>

Population::Population() {
	m_individuals = new Individual[PARAMS__POPULATION_SIZE];
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
		m_individuals[i] = Individual();
	printf_s("Population initiated.");
}

void Population::run() {
	if (SYSTEM__USE_GPU)
		printf_s("Running. (GPU support ENABLED)");
	else
		printf_s("Running. (GPU support DISABLED)");

	float* input = new float[20000, SUBSTRATE__INPUT_SIZE];
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		float* output = m_individuals[i].getOutput(input);
	}
}