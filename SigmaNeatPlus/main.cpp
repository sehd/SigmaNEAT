#include <iostream>
#include <conio.h>
#include <chrono>
#include "Population.hpp"

bool hasArgumentFlag(int argc, char** argv, std::string flag) {
	for (int i = 0; i < argc; i++)
	{
		if (flag.compare(argv[i]) == 0) {
			return true;
		}
	}
	return false;
}

void runPopulation(bool t_verbose = true) {
	Population population = Population(t_verbose);
	population.run();
}

void timeRange() {
	const int startPopSize = 5;
	const int endPopSize = 200;
	for (int i = startPopSize; i <= endPopSize; i++)
	{
		PARAMS__POPULATION_SIZE = i;
		SYSTEM__USE_GPU = false;
		auto startTime = std::chrono::high_resolution_clock::now();
		runPopulation(false);
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::high_resolution_clock::now() - startTime).count();
		std::cout << PARAMS__POPULATION_SIZE << "\tNo GPU\t" << duration << "\t\t";
		SYSTEM__USE_GPU = true;
		startTime = std::chrono::high_resolution_clock::now();
		runPopulation(false);
		duration = std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::high_resolution_clock::now() - startTime).count();
		std::cout << PARAMS__POPULATION_SIZE << "\tGPU\t" << duration << std::endl;
	}
}

int main(int argc, char** argv)
{
	if (hasArgumentFlag(argc, argv, "--time-range"))
		timeRange();
	if (!hasArgumentFlag(argc, argv, "--no-time")) {
		auto startTime = std::chrono::high_resolution_clock::now();
		runPopulation();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::high_resolution_clock::now() - startTime).count();
		std::cout << "Total operation done in " << duration << " micro seconds." << std::endl
			<< "Average of " << duration / PARAMS__POPULATION_SIZE << " micro seconds per individual"
			<< std::endl;
	}
	else
		runPopulation();

	if (!hasArgumentFlag(argc, argv, "--profiler")) {
		std::cout << "Press any key to exit..." << std::endl;
		int _ = _getch();
	}
	else {
		std::cout << "Profiler flag found. Exiting." << std::endl;
	}
	return 0;
}