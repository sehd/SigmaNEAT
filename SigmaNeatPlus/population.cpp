#include <iostream>
#include <fstream>
#include <sstream>
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

double* readInputFromFile() {
	int trialCount = PARAMS__TRAINING_SIZE + PARAMS__TEST_SIZE;
	double* input = new double[trialCount * SUBSTRATE__INPUT_SIZE];

	std::ifstream file;
	file.open(inputFilePath, std::ios::in);
	if (file.is_open()) {
#if LOG_DEBUG
		std::cout << "Input file opened" << std::endl;
#endif
	}
	else {
#if LOG_ERROR
		std::cout << "ERROR: Couldn't open file" << std::endl;
#endif
		return nullptr;
	}
	std::string line;
	for (int i = 0; i < trialCount; i++)
	{
		std::getline(file, line);
		std::istringstream stringStream(line);
		std::string item;
		for (int j = 0; j < SUBSTRATE__INPUT_SIZE; j++)
		{
			std::getline(stringStream, item, ',');
			input[i * SUBSTRATE__INPUT_SIZE + j] = std::stod(item);
		}
	}
	file.close();

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

double* Population::getBestTestResult(double* t_performances, double* t_input) {
	double maxPerformance = -INFINITY;
	int maxPerformanceIndex = -1;
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		if (t_performances[i] > maxPerformance) {
			maxPerformance = t_performances[i];
			maxPerformanceIndex = i;
		}
	}

	double* result = new double[PARAMS__TEST_SIZE * SUBSTRATE__OUTPUT_SIZE];
	for (int i = 0; i < PARAMS__TEST_SIZE; i++)
	{
		double* output = m_individuals[maxPerformanceIndex].
			getOutput(PARAMS__TEST_SIZE, &t_input[PARAMS__TRAINING_SIZE]);
		for (int j = 0; j < SUBSTRATE__OUTPUT_SIZE; j++)
			result[i * SUBSTRATE__OUTPUT_SIZE + j] = output[j];

		delete[] output;
	}
	return result;
}

void Population::run() {
#if LOG_DEBUG
	if (SYSTEM__USE_GPU)
		std::cout << "Running. (GPU support ENABLED)" << std::endl;
	else
		std::cout << "Running. (GPU support DISABLED)" << std::endl;
#endif
	double* input = readInputFromFile();

	for (int generation = 0; generation < PARAMS__TRAINING_GENERATIONS - 1; generation++)
	{
		double* performance = trainGeneration(input);
		createNextGeneration(performance);
		delete[] performance;
	}
	double* performance = trainGeneration(input);

	double* result = getBestTestResult(performance, input);
	std::cout << "Results:" << std::endl;
	for (int i = 0; i < PARAMS__TEST_SIZE; i++) {
		for (int j = 0; j < SUBSTRATE__OUTPUT_SIZE; j++)
			std::cout << result[i * SUBSTRATE__OUTPUT_SIZE + j] << ", ";
		std::cout << std::endl;
	}


	delete[] result;
	delete[] performance;
	delete[] input;
}