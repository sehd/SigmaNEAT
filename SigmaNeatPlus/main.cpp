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

void runPopulation() {
	Population population = Population("Input.txt", "Output.txt");
	population.run();
}

void timeOne() {
	auto startTime = std::chrono::high_resolution_clock::now();
	runPopulation();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
		std::chrono::high_resolution_clock::now() - startTime).count();
	std::cout << "Total operation done in " << duration << " micro seconds." << std::endl
		<< "Average of " << duration / PARAMS__POPULATION_SIZE << " micro seconds per individual"
		<< std::endl;
}

int main(int argc, char** argv)
{
	if (hasArgumentFlag(argc, argv, "--no-time"))
		runPopulation();
	else
		timeOne();

	if (!hasArgumentFlag(argc, argv, "--profiler")) {
		std::cout << "Press any key to exit..." << std::endl;
		int _ = _getch();
	}
	else {
		std::cout << "Profiler flag found. Exiting.";
	}

	return 0;
}