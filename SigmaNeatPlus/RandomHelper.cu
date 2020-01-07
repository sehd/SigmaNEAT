#include "config.hpp"
#include "RandomHelper.hpp"

RandomHelper::RandomHelper(int t_seed, int t_threadIndex) :
	m_seed(t_seed),
	m_threadIndex(t_threadIndex),
	m_isInitialized(false),
	m_randomDevice() {}

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
	if (!m_isInitialized) {
		curand_init(m_seed, m_threadIndex, 0, &m_curandState[m_threadIndex]);
		m_isInitialized = true;
	}

	float val = curand_uniform(m_curandState);
	return -(val - 1); //Amazingly curand_uniform includes 1 bug excludes 0 this bit fixes that
}

float RandomHelper::getRandomCpu() {
	static std::uniform_real_distribution<> dis(0, 1);
	return (float)dis(m_randomDevice);
}