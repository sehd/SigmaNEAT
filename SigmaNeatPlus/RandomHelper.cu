#include <random>
#include "config.hpp"
#include "RandomHelper.hpp"

__device__
float getRandomGpu() {
	float val = 1;//TODO curand_uniform;
	return -(val - 1); //Amazingly curand_uniform includes 1 bug excludes 0 this bit fixes that
}

float getRandomCpu() {
	static std::default_random_engine e;
	static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
	return (float)dis(e);
}

float RandomHelper::getRandom()
{
	if (SYSTEM__USE_GPU)
	{
		return getRandomGpu();
	}
	else
	{
		return getRandomCpu();
	}
}