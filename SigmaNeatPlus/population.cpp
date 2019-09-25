#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include "Population.hpp"
#include "Config.hpp"

Population::Population(char* t_inputFilePath, char* t_outputFilePath) {
	m_inputFilePath = t_inputFilePath;
	m_outputFilePath = t_outputFilePath;
	m_individuals = new Individual[PARAMS__POPULATION_SIZE];

#if LOG_DEBUG
	std::cout << "Population initiated." << std::endl;
#endif
}

Population::~Population() {
	delete[] m_individuals;
}

double* readVectorFromFile(int t_count, int t_size, const char* t_filePath) {
	double* result = new double[t_count * t_size];

	std::ifstream file;
	file.open(t_filePath, std::ios::in);
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
	for (int i = 0; i < t_count; i++)
	{
		std::getline(file, line);
		std::istringstream stringStream(line);
		std::string item;
		for (int j = 0; j < t_size; j++)
		{
			std::getline(stringStream, item, ',');
			result[i * t_size + j] = std::stod(item);
		}
	}
	file.close();

#if LOG_DEBUG
	std::cout << "Input read from file successfuly" << std::endl;
#endif

	return result;
}

double* Population::trainGeneration(double* t_input, double* t_expectedOutput) {
	double* error = new double[PARAMS__POPULATION_SIZE];
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		double* output = m_individuals[i].getOutput(PARAMS__TRAINING_SIZE, t_input);

		// TODO: Do this in the GPU kernel along other tasks
		error[i] = 0;
		for (int j = 0; j < PARAMS__TRAINING_SIZE; j++)
		{
			for (int k = 0; k < SUBSTRATE__OUTPUT_SIZE; k++)
			{
				double diff = output[j * SUBSTRATE__OUTPUT_SIZE + k] -
					t_expectedOutput[j * SUBSTRATE__OUTPUT_SIZE + k];
				error[i] += diff * diff;
			}
		}

#if LOG_VERBOSE
		std::cout << "Result for individual " << i << ": ";
		for (int j = 0; j < fminl(PARAMS__TRAINING_SIZE, 10); j++)
		{
			for (int k = 0; k < fminl(SUBSTRATE__OUTPUT_SIZE, 3); k++)
			{

			}
			std::cout << "(";
			for (int k = 0; k < fminl(SUBSTRATE__OUTPUT_SIZE, 3); k++)
				std::cout << output[j * SUBSTRATE__OUTPUT_SIZE + k] << ", ";
			std::cout << (SUBSTRATE__OUTPUT_SIZE > 3 ? "..." : "") << ") ";
		}
		std::cout << (PARAMS__TRAINING_SIZE > 10 ? "..." : "") << std::endl;
#endif

		delete[] output;
	}

#if LOG_DEBUG
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		std::cout << "Error for individual " << i << " = " << error[i] << std::endl;
	}
#endif

	return error;
}

void Population::createNextGeneration(double* error) {

}

double* Population::getBestTestResult(double* t_errors, double* t_input) {
	double maxError = -INFINITY;
	int maxErrorIndex = -1;
	for (int i = 0; i < PARAMS__POPULATION_SIZE; i++)
	{
		if (t_errors[i] > maxError) {
			maxError = t_errors[i];
			maxErrorIndex = i;
		}
	}

	double* result = new double[PARAMS__TEST_SIZE * SUBSTRATE__OUTPUT_SIZE];
	for (int i = 0; i < PARAMS__TEST_SIZE; i++)
	{
		double* output = m_individuals[maxErrorIndex].
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
	double* input = readVectorFromFile(
		PARAMS__TRAINING_SIZE + PARAMS__TEST_SIZE,
		SUBSTRATE__INPUT_SIZE,
		m_inputFilePath);
	double* expOutput = readVectorFromFile(
		PARAMS__TRAINING_SIZE + PARAMS__TEST_SIZE,
		SUBSTRATE__OUTPUT_SIZE,
		m_outputFilePath);

	for (int generation = 0; generation < PARAMS__TRAINING_GENERATIONS - 1; generation++)
	{
		double* error = trainGeneration(input, expOutput);
		createNextGeneration(error);
		delete[] error;
	}
	double* error = trainGeneration(input, expOutput);

	double* result = getBestTestResult(error, input);
	std::cout << "Results:" << std::endl;
	for (int i = 0; i < PARAMS__TEST_SIZE; i++) {
		for (int j = 0; j < SUBSTRATE__OUTPUT_SIZE; j++)
			std::cout << result[i * SUBSTRATE__OUTPUT_SIZE + j] << ", ";
		std::cout << std::endl;
	}

	delete[] result;
	delete[] error;
	delete[] input;
}