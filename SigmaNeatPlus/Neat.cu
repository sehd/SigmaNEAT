#include <random>
#include <string>
#include "Neat.hpp"

Neat::Neat(int t_inputSize, int t_outputSize, int* t_innovationNumber) :
	m_inputSize(t_inputSize),
	m_outputSize(t_outputSize),
	m_nodeCount(t_inputSize + t_outputSize),
	m_connectionCount(t_inputSize* t_outputSize),
	m_innovationNumber(t_innovationNumber) {

	m_nodeGenes = new Node[m_nodeCount];
	for (int i = 0; i < t_inputSize + t_outputSize; i++) {
		m_nodeGenes[i].id = i;
		m_nodeGenes[i].value = 0;
		m_nodeGenes[i].hasValue = false;
		m_nodeGenes[i].activationFunction = ActivationFunction::Identity;
	}
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
}

Neat::~Neat() {
	delete[] m_nodeGenes;
	delete[] m_connectionGenes;
}

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
			nodeInputValue += prevNodeValue * connection.weight;
		}
	}
	t_node.hasValue = true;
	t_node.value = ActivationFunction::activate(t_node.activationFunction, nodeInputValue);
	return t_node.value;
}

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

Neat* Neat::copyToDevice() {
	//TODO: Use unified memory. cudaMallocManaged
	Node* nodes;
	cudaMalloc(&nodes, m_nodeCount * sizeof(Node));
	cudaMemcpy(nodes, m_nodeGenes,
		m_nodeCount * sizeof(Node), cudaMemcpyHostToDevice);

	Connection* connections;
	cudaMalloc(&connections, m_connectionCount * sizeof(Connection));
	cudaMemcpy(connections, m_connectionGenes,
		m_connectionCount * sizeof(Connection), cudaMemcpyHostToDevice);

	Neat* d_neat;
	cudaMalloc(&d_neat, sizeof(Neat));
	cudaMemcpy(d_neat, this, sizeof(Neat), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_neat->m_nodeGenes), &nodes, sizeof(nodes), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_neat->m_connectionGenes), &connections, sizeof(connections), cudaMemcpyHostToDevice);

	return d_neat;
}

void Neat::crossOver(const Neat* t_parent1, const Neat* t_parent2) {
	//TODO
}

void Neat::mutate() {
	//TODO
}

std::string Neat::toString() {
	std::string res = "";
	for (int i = 0; i < m_connectionCount; i++)
	{
		res.append(std::to_string(m_connectionGenes[i].input));
		if (m_connectionGenes[i].enabled) {
			res.append(" --- ");
			res.append(std::to_string(m_connectionGenes[i].weight));
			res.append(" ---> ");
		}
		else {
			res.append(" -X- ");
			res.append(std::to_string(m_connectionGenes[i].weight));
			res.append(" -X-> ");
		}
		res.append(std::to_string(m_connectionGenes[i].output));
		res.append("\n");
	}
	return res;
}
