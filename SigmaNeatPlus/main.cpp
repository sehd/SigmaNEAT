#include <iostream>
#include <conio.h>
#include "Population.hpp"

int main()
{
	Population population = Population();
	population.run();
	std::cout << "Press any key to exit..." << std::endl;
	int _ = _getch();
	return 0;
}