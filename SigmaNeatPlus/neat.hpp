#pragma once
#ifndef NEAT_H
#define NEAT_H

#include "Node.hpp"
#include "Connection.hpp"

//This class holds the logic for NEAT algorithm.
class Neat
{
	int m_innovationNumber;
	Node* m_nodeGenes;
	Connection* m_connectionGenes;
	int m_inputSize;
	int m_outputSize;

	double getValueRecursive(Node t_node);
public:
	Neat();
	//SUBSTRATE__DIMENSION*2 is input size
	//Single output.
	void getValue(double* t_input, double* t_output);
	void crossOver();
	void mutate();
};

#endif // !NEAT_H