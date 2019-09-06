#pragma once
#ifndef CONNECTION_H
#define CONNECTION_H

struct Connection {
	int input;
	int output;
	double weight;
	bool enabled;
	int innovationNo;
};

#endif // !CONNECTION_H
