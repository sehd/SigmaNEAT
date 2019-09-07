#include <random>
#include <cuda_runtime.h>
#include "Neat.hpp"

__device__ __host__
Neat::Neat(int t_inputSize, int t_outputSize, int* t_innovationNumber) {
	m_inputSize = t_inputSize;
	m_outputSize = t_outputSize;
	m_nodeCount = t_inputSize + t_outputSize;
	m_nodeGenes = new Node[m_nodeCount];
	for (int i = 0; i < t_inputSize + t_outputSize; i++) {
		m_nodeGenes[i].id = i;
		m_nodeGenes[i].value = 0;
		m_nodeGenes[i].hasValue = false;
		m_nodeGenes[i].activationFunction = ActivationFunction::Identity;
	}
	m_connectionCount = t_inputSize * t_outputSize;
	m_connectionGenes = new Connection[m_connectionCount];
	for (int i = 0; i < t_inputSize; i++)
	{
		for (int j = 0; j < t_outputSize; j++)
		{
			m_connectionGenes[(i * t_outputSize) + j].input = i;
			m_connectionGenes[(i * t_outputSize) + j].output = j + t_inputSize;
			m_connectionGenes[(i * t_outputSize) + j].weight = 0.5; //TODO: Random
			m_connectionGenes[(i * t_outputSize) + j].enabled = true;
			m_connectionGenes[(i * t_outputSize) + j].innovationNo = (i * t_outputSize) + j;
		}
	}
	m_innovationNumber = t_innovationNumber;
}

__device__ __host__
double Neat::getValueRecursive(Node t_node) {
	if (t_node.hasValue)
		return t_node.value;
	double nodeInputValue = 0;
	for (int connectionIndex = 0; connectionIndex < m_connectionCount; connectionIndex++)
	{
		Connection connection = m_connectionGenes[connectionIndex];
		if (connection.output == t_node.id && connection.enabled)
		{
			double prevNodeValue = getValueRecursive(m_nodeGenes[connection.input]);
			nodeInputValue = prevNodeValue * connection.weight;
		}
	}
	t_node.hasValue = true;
	t_node.value = ActivationFunction::activate(t_node.activationFunction, nodeInputValue);
	return t_node.value;
}

__device__ __host__
void Neat::getValue(double* t_input, double* t_output) {
	for (int nodeIndex = 0; nodeIndex < m_nodeCount; nodeIndex++)
	{
		if (m_nodeGenes[nodeIndex].id < m_inputSize) {
			m_nodeGenes[nodeIndex].value = t_input[nodeIndex];
			m_nodeGenes[nodeIndex].hasValue = true;
		}
		else {
			m_nodeGenes[nodeIndex].hasValue = false;
		}
	}
	for (int outputIndex = 0; outputIndex < m_outputSize; outputIndex++)
	{
		t_output[outputIndex] = getValueRecursive(m_nodeGenes[m_inputSize + outputIndex]);
	}
}

Neat Neat::crossOver(Neat t_parent1, Neat t_parent2) {
	//TODO:
	return t_parent1;
}

void Neat::mutate() {

}
