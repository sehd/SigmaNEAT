#pragma once
#ifndef NEAT_H
#define NEAT_H

#include "Node.hpp"
#include "Connection.hpp"

//This class holds the logic for NEAT algorithm.
class Neat
{
	int* m_innovationNumber;
	
	Node* m_nodeGenes;
	int m_nodeCount;

	Connection* m_connectionGenes;
	int m_connectionCount;

	int m_inputSize;
	int m_outputSize;

	double getValueRecursive(Node t_node);
public:
	Neat(int t_inputSize, int t_outputSize, int* t_innovationNumber);
	~Neat();
	//Sizes should match input size and output size
	void getValue(double* t_input, double* t_output);
	Neat* copyToDevice();
	static Neat crossOver(Neat t_parent1, Neat t_parent2);
	void mutate();
};

#endif // !NEAT_H