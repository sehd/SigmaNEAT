#pragma once
#ifndef NEAT_H
#define NEAT_H

#include <cuda_runtime.h>
#include "Node.hpp"
#include "Connection.hpp"
#include "RandomHelper.hpp"

//This class holds the logic for NEAT algorithm.
class Neat
{
	RandomHelper m_randomHelper;
	int* m_innovationNumber;

	Node* m_nodeGenes;
	int m_nodeCount;

	Connection* m_connectionGenes;
	int m_connectionCount;

	int m_inputSize;
	int m_outputSize;

	void addConnection(Connection* newConnection);

	void mutateWeights();
	void mutateAddNode();
	void mutateAddConnection();

	__device__ __host__
		double getValueRecursive(Node* t_node);
public:
	__device__ __host__
		Neat(int t_inputSize, int t_outputSize, int* t_innovationNumber, int t_randomSeed, int t_randomState);
	__device__ __host__
		~Neat();
	//Sizes should match input size and output size
	__device__ __host__
		void getValue(double* t_input, double* t_output);
	__host__
		Neat* copyToDevice(int t_trialCount, Node* nodes, Connection* connections);
	//Parent1 is always considered the more fit one.
	__host__
		void crossOver(const Neat* t_parent1, const Neat* t_parent2);
	__host__
		void mutate();

	__host__
		std::string toString();
};

#endif // !NEAT_H