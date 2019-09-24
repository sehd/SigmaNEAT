#include <iostream>
#include <fstream>
#include <math.h>
#include "Population.hpp"
#include "Config.hpp"

const char* inputFilePath;

Population::Population(char* t_inputFilePath) {
	inputFilePath = t_inputFilePath;
	m_individuals = new Individual[PARAMS__POPULATION_SIZE];

#if LOG_DEBUG
	std::cout << "Population initiated." << std::endl;
#endif
}

Population::~Population() {
	delete[] m_individuals;
}

double* readInputFromFile(int t_trialCount) {
	double* input = new double[t_trialCount * SUBSTRATE__INPUT_SIZE];
	for (int i = 0; i < t_trialCount; i++)
		for (int j = 0; j < SUBSTRATE__INPUT_SIZE; j++)
			input[i * SUBSTRATE__INPUT_SIZE + j] = 1;//(double)(i * i) + j * j;
	return input;
}

double* Population::trainGeneration(double* t_input) {
	double* performance = new double[PARAMS__POPULATION_SIZE];
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		double* output = m_individuals[i].getOutput(PARAMS__TRAINING_SIZE, t_input);

#if LOG_DEBUG
		//Temp representation of data
		std::cout << "Result for individual " << i << ": ";
		for (int j = 0; j < fminl(PARAMS__TRAINING_SIZE, 10); j++)
		{
			std::cout << "(";
			for (int k = 0; k < fminl(SUBSTRATE__OUTPUT_SIZE, 3); k++)
				std::cout << output[j * SUBSTRATE__OUTPUT_SIZE + k] << ", ";
			std::cout << (SUBSTRATE__OUTPUT_SIZE > 3 ? "..." : "") << ") ";
		}
		std::cout << (PARAMS__TRAINING_SIZE > 10 ? "..." : "") << std::endl;
#endif
		performance[i] = 1;// TODO: Calculate output and expected distance
		delete[] output;
	}
	return performance;
}

void Population::createNextGeneration(double* performance) {

}

void Population::run() {
#if LOG_DEBUG
	if (SYSTEM__USE_GPU)
		std::cout << "Running. (GPU support ENABLED)" << std::endl;
	else
		std::cout << "Running. (GPU support DISABLED)" << std::endl;
#endif
	double* input = readInputFromFile(PARAMS__TRAINING_SIZE + PARAMS__TEST_SIZE);

	for (int generation = 0; generation < PARAMS__TRAINING_GENERATIONS-1; generation++)
	{
		double* performance = trainGeneration(input);
		createNextGeneration(performance);
		delete[] performance;
	}
	double* performance = trainGeneration(input);
	
	//TODO: Move to method
	double maxPerformance = -INFINITY;
	int maxPerformanceIndex = -1;
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		if (performance[i] > maxPerformance) {
			maxPerformance = performance[i];
			maxPerformanceIndex = i;
		}
	}
	for (int i = 0; i < PARAMS__TEST_SIZE; i++)
	{
		double* output = m_individuals[maxPerformanceIndex].
			getOutput(PARAMS__TEST_SIZE, &input[PARAMS__TRAINING_SIZE]);
	}

	delete[] performance;
	delete[] input;
}