#pragma once
#ifndef POPULATION_H
#define POPULATION_H

#include "Individual.hpp"

//A population is responsible to handle and control individuals.
//This includes mutationand cross - over and other GA operations.
class Population
{
	Individual* m_individuals;
	double* Population::trainGeneration(double* t_input);
	void createNextGeneration(double* performance);
	double* getBestTestResult(double* t_performances, double* t_input);
public:
	Population(char* t_inputFilePath);
	~Population();
	void run();
};

#endif // !POPULATION_H