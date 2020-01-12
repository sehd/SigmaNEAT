#include <string>
#include <map>
#include <vector>
#include "Neat.hpp"
#include "config.hpp"

Neat::Neat(int t_inputSize, int t_outputSize, int* t_innovationNumber, int t_randomSeed, int t_randomState) :
	m_inputSize(t_inputSize),
	m_outputSize(t_outputSize),
	m_nodeCount(t_inputSize + t_outputSize),
	m_connectionCount(t_inputSize* t_outputSize),
	m_innovationNumber(t_innovationNumber),
	m_randomHelper(t_randomSeed, t_randomState) {

	m_nodeGenes = new Node[m_nodeCount];
	for (int i = 0; i < t_inputSize + t_outputSize; i++) {
		m_nodeGenes[i].id = i;
		m_nodeGenes[i].value = 0;
		m_nodeGenes[i].hasValue = false;
		m_nodeGenes[i].activationFunction = ActivationFunction::getFromRandom(m_randomHelper.getRandomCpu());
	}
	m_connectionGenes = new Connection[m_connectionCount];
	for (int i = 0; i < t_inputSize; i++)
	{
		for (int j = 0; j < t_outputSize; j++)
		{
			m_connectionGenes[(i * t_outputSize) + j].input = i;
			m_connectionGenes[(i * t_outputSize) + j].output = j + t_inputSize;
			m_connectionGenes[(i * t_outputSize) + j].weight = m_randomHelper.getRandomCpu() - 0.5; //Between -0.5 and 0.5
			m_connectionGenes[(i * t_outputSize) + j].enabled = true;
			m_connectionGenes[(i * t_outputSize) + j].innovationNo = i * t_outputSize + j; //Innovation number of the default connections should be the same.
		}
	}
}

Neat::~Neat() {
	delete[] m_nodeGenes;
	delete[] m_connectionGenes;
}

double Neat::getValueRecursive(Node* t_node) {
	if (t_node->hasValue)
		return t_node->value;

	// To handle architectural loops
	t_node->hasValue = true;
	t_node->value = 0;

	double nodeInputValue = 0;
	for (int connectionIndex = 0; connectionIndex < m_connectionCount; connectionIndex++)
	{
		Connection connection = m_connectionGenes[connectionIndex];
		if (connection.output == t_node->id && connection.enabled)
		{
			Node* n = &m_nodeGenes[connection.input];
			double prevNodeValue = getValueRecursive(n);
			nodeInputValue += prevNodeValue * connection.weight;
		}
	}
	t_node->value = ActivationFunction::activate(t_node->activationFunction, nodeInputValue);
	return t_node->value;
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
		t_output[outputIndex] = getValueRecursive(&m_nodeGenes[m_inputSize + outputIndex]);
	}
}

Neat* Neat::copyToDevice(int t_trialCount) {
	//TODO: Use unified memory. cudaMallocManaged
	Neat* d_neat;
	cudaMalloc(&d_neat, sizeof(Neat) * t_trialCount);

	for (int i = 0; i < t_trialCount; i++)
	{
		Node* nodes;
		cudaMalloc(&nodes, m_nodeCount * sizeof(Node));
		cudaMemcpy(nodes, m_nodeGenes,
			m_nodeCount * sizeof(Node), cudaMemcpyHostToDevice);

		Connection* connections;
		cudaMalloc(&connections, m_connectionCount * sizeof(Connection));
		cudaMemcpy(connections, m_connectionGenes,
			m_connectionCount * sizeof(Connection), cudaMemcpyHostToDevice);

		cudaMemcpy(&d_neat[i], this, sizeof(Neat), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_neat[i].m_nodeGenes), &nodes, sizeof(nodes), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_neat[i].m_connectionGenes), &connections, sizeof(connections), cudaMemcpyHostToDevice);
	}
	return d_neat;
}

void Neat::crossOver(const Neat* t_parent1, const Neat* t_parent2) {
	delete[] m_nodeGenes;
	delete[] m_connectionGenes;
	this->m_connectionCount = t_parent1->m_connectionCount;
	this->m_connectionGenes = new Connection[t_parent1->m_connectionCount];

	//Create connections
	for (int i = 0; i < t_parent1->m_connectionCount; i++)
	{
		bool assigned = false;
		for (int j = 0; j < t_parent2->m_connectionCount; j++)
		{
			if (t_parent1->m_connectionGenes[i].innovationNo ==
				t_parent2->m_connectionGenes[j].innovationNo)
			{
				assigned = true;
				if (m_randomHelper.getRandomCpu() > 0.5) {
					this->m_connectionGenes[i].input = t_parent2->m_connectionGenes[i].input;
					this->m_connectionGenes[i].output = t_parent2->m_connectionGenes[i].output;
					this->m_connectionGenes[i].weight = t_parent2->m_connectionGenes[i].weight;
					this->m_connectionGenes[i].innovationNo = t_parent2->m_connectionGenes[i].innovationNo;
					this->m_connectionGenes[i].enabled = t_parent2->m_connectionGenes[i].enabled;
				}
				else {
					this->m_connectionGenes[i].input = t_parent1->m_connectionGenes[i].input;
					this->m_connectionGenes[i].output = t_parent1->m_connectionGenes[i].output;
					this->m_connectionGenes[i].weight = t_parent1->m_connectionGenes[i].weight;
					this->m_connectionGenes[i].innovationNo = t_parent1->m_connectionGenes[i].innovationNo;
					this->m_connectionGenes[i].enabled = t_parent1->m_connectionGenes[i].enabled;
				}
			}
		}
		// Excess or disjoint
		if (!assigned) {
			this->m_connectionGenes[i].input = t_parent1->m_connectionGenes[i].input;
			this->m_connectionGenes[i].output = t_parent1->m_connectionGenes[i].output;
			this->m_connectionGenes[i].weight = t_parent1->m_connectionGenes[i].weight;
			this->m_connectionGenes[i].innovationNo = t_parent1->m_connectionGenes[i].innovationNo;
			this->m_connectionGenes[i].enabled = t_parent1->m_connectionGenes[i].enabled;
		}
	}

	//Create nodes
	std::map<int, Node>newNodes;
	for (int i = 0; i < this->m_connectionCount; i++)
	{
		int inputId = m_connectionGenes[i].input;
		if (newNodes.count(inputId) == 0) {
			Node node;
			node.id = -1;
			for (int j = 0; j < t_parent1->m_nodeCount; j++)
			{
				if (t_parent1->m_nodeGenes[j].id == inputId) {
					node.id = inputId;
					node.activationFunction = t_parent1->m_nodeGenes[j].activationFunction;
					break;
				}
			}
			if (node.id == -1)
				for (int j = 0; j < t_parent2->m_nodeCount; j++)
				{
					if (t_parent2->m_nodeGenes[j].id == inputId) {
						node.id = inputId;
						node.activationFunction = t_parent2->m_nodeGenes[j].activationFunction;
						break;
					}
				}
			newNodes.insert({ inputId, node });
		}
		int outputId = m_connectionGenes[i].output;
		if (newNodes.count(outputId) == 0) {
			Node node;
			node.id = -1;
			for (int j = 0; j < t_parent1->m_nodeCount; j++)
			{
				if (t_parent1->m_nodeGenes[j].id == outputId) {
					node.id = outputId;
					node.activationFunction = t_parent1->m_nodeGenes[j].activationFunction;
					break;
				}
			}
			if (node.id == -1)
				for (int j = 0; j < t_parent2->m_nodeCount; j++)
				{
					if (t_parent2->m_nodeGenes[j].id == outputId) {
						node.id = outputId;
						node.activationFunction = t_parent2->m_nodeGenes[j].activationFunction;
						break;
					}
				}
			newNodes.insert({ outputId, node });
		}
	}

	//Assign nodes based on the map
	this->m_nodeCount = (int)newNodes.size();
	this->m_nodeGenes = new Node[newNodes.size()];
	int i = 0;
	for (const auto& item : newNodes) {
		this->m_nodeGenes[i].id = item.second.id;
		this->m_nodeGenes[i].activationFunction = item.second.activationFunction;
		this->m_nodeGenes[i].value = 0;
		this->m_nodeGenes[i].hasValue = false;
		i++;
	}
}

void Neat::mutateWeights() {
	for (int i = 0; i < m_connectionCount; i++)
	{
		if (m_randomHelper.getRandomCpu() < MUTATION__WEIGHT_RATE)
		{
			float amount = (m_randomHelper.getRandomCpu() / 5) - 0.1f; //Between -0.1 and 0.1
			m_connectionGenes[i].weight += amount;
		}
	}
}

void Neat::addConnection(Connection* newConnection) {
	Connection* newConnections = new Connection[m_connectionCount + 1l];
	for (int i = 0; i < m_connectionCount; i++)
	{
		newConnections[i] = m_connectionGenes[i];
	}
	newConnections[m_connectionCount] = *newConnection;
	m_connectionCount++;
	delete[] m_connectionGenes;
	m_connectionGenes = newConnections;
}

void Neat::mutateAddNode() {
	std::vector<int> selectableConnections;
	for (int i = 0; i < m_connectionCount; i++)
	{
		if (m_connectionGenes[i].enabled)
			selectableConnections.push_back(i);
	}
	if (selectableConnections.size() > 0)
	{
		int index = selectableConnections[(int)(m_randomHelper.getRandomCpu() * selectableConnections.size())];
		m_connectionGenes[index].enabled = false;
		Node* newNodes = new Node[m_nodeCount + 1l];
		int maxId = 0;
		for (int i = 0; i < m_nodeCount; i++)
		{
			newNodes[i] = m_nodeGenes[i];
			if (newNodes[i].id > maxId)
				maxId = newNodes[i].id;
		}
		newNodes[m_nodeCount].id = maxId + 1;
		newNodes[m_nodeCount].activationFunction = ActivationFunction::getFromRandom(m_randomHelper.getRandomCpu());
		newNodes[m_nodeCount].value = 0;
		newNodes[m_nodeCount].hasValue = false;
		m_nodeCount++;
		delete[] m_nodeGenes;
		m_nodeGenes = newNodes;

		Connection newCon1;
		newCon1.input = m_connectionGenes[index].input;
		newCon1.output = newNodes[m_nodeCount - 1].id;
		newCon1.innovationNo = ++ * m_innovationNumber;
		newCon1.enabled = true;
		newCon1.weight = 1;
		addConnection(&newCon1);

		Connection newCon2;
		newCon2.input = newNodes[m_nodeCount - 1].id;
		newCon2.output = m_connectionGenes[index].output;
		newCon2.innovationNo = ++ * m_innovationNumber;
		newCon2.enabled = true;
		newCon2.weight = m_connectionGenes[index].weight;
		addConnection(&newCon2);
	}
}

void Neat::mutateAddConnection() {
	std::vector<Connection> possibleConnections;
	for (int i = 0; i < m_nodeCount - m_outputSize; i++) //Every possible connection from input layer...
	{
		for (int j = m_inputSize; j < m_nodeCount; j++) //... upto output layer
		{
			if (i != j)
			{
				bool hit = false;
				for (int k = 0; k < m_connectionCount; k++)
				{
					if (m_connectionGenes[k].input == i &&
						m_connectionGenes[k].output == j &&
						m_connectionGenes[k].enabled)
					{
						hit = true;
						break;
					}
				}
				if (!hit) //Not exists
				{
					Connection newCon;
					newCon.input = i;
					newCon.output = j;
					newCon.enabled = true;
					newCon.weight = m_randomHelper.getRandomCpu() - 0.5;
					newCon.innovationNo = ++ * m_innovationNumber;
					possibleConnections.push_back(newCon);
				}
			}
		}
	}
	if (possibleConnections.size() > 0) {
		int index = (int)(m_randomHelper.getRandomCpu() * possibleConnections.size());
		addConnection(&possibleConnections[index]);
	}
}

void Neat::mutate() {
	mutateWeights();

	if (m_randomHelper.getRandomCpu() < MUTATION__ADD_NODE)
		mutateAddNode();

	if (m_randomHelper.getRandomCpu() < MUTATION__ADD_CONNECTION)
		mutateAddConnection();
}

std::string Neat::toString() {
	std::string res = "";
	for (int i = 0; i < m_nodeCount; i++)
	{
		res.append("f(");
		res.append(std::to_string(m_nodeGenes[i].id));
		res.append(") = ");
		res.append(ActivationFunction::toString(m_nodeGenes[i].activationFunction));
		res.append("\n");
	}
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