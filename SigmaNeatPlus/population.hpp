#pragma once
#ifndef POPULATION_H
#define POPULATION_H

#include "Individual.hpp"

//A population is responsible to handle and control individuals.
//This includes mutationand cross - over and other GA operations.
class Population
{
	const char* m_inputFilePath, * m_outputFilePath;
	Individual* m_individuals;
	double* Population::trainGeneration(double* t_input, double* t_expectedOutput);
	void createNextGeneration(double* error);
	double* getBestTestResult(double* t_errors, double* t_input);
public:
	Population(char* t_inputFilePath, char* t_outputFilePath);
	~Population();
	void run();
};

#endif // !POPULATION_H