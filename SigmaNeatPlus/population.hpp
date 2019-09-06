#pragma once
#ifndef POPULATION_H
#define POPULATION_H

#include "Individual.hpp"

//A population is responsible to handle and control individuals.
//This includes mutationand cross - over and other GA operations.
class Population
{
	Individual* m_individuals;
public:
	Population();
	void run();
};

#endif // !POPULATION_H