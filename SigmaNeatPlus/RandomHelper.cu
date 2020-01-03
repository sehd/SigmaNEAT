#include <random>
#include "config.hpp"
#include "RandomHelper.hpp"

float getRandomCpu() {
	static std::default_random_engine e;
	static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
	return (float)dis(e);
}

RandomHelper::RandomHelper() {
	curand_init(1234, idx, 0, &m_curandState[idx]);
}

float RandomHelper::getRandom()
{
#if USE_GPU
	return getRandomGpu();
#else
	return getRandomCpu();
#endif
}

float RandomHelper::getRandomGpu()
{

	float val = curand_uniform(m_curandState);
	return -(val - 1); //Amazingly curand_uniform includes 1 bug excludes 0 this bit fixes that
}