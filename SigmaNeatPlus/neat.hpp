#pragma once
#ifndef NEAT_H
#define NEAT_H

#include <cuda_runtime.h>
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

	__device__ __host__
		double getValueRecursive(Node t_node);
public:
	__device__ __host__
		Neat(int t_inputSize, int t_outputSize, int* t_innovationNumber);
	__device__ __host__
		~Neat();
	//Sizes should match input size and output size
	__device__ __host__
		void getValue(double* t_input, double* t_output);
	Neat* copyToDevice();
	void crossOver(const Neat* t_parent1, const Neat* t_parent2);
	void mutate();
};

#endif // !NEAT_H