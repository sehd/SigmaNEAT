#pragma once
#ifndef CONFIGURED
#define CONFIGURED

// System
constexpr auto SYSTEM__USE_GPU = false;
constexpr auto SYSTEM__THREADS_PER_BLOCK = 512;

// Substrate
constexpr auto SUBSTRATE__DIMENSION = 2;
constexpr auto SUBSTRATE__INPUT_SIZE = 8;
constexpr auto SUBSTRATE__OUTPUT_SIZE = 1;
constexpr auto SUBSTRATE__LAYERS_COUNT = 5;
constexpr auto SUBSTRATE__LAYER_SIZE = 5;

// Params
constexpr auto PARAMS__POPULATION_SIZE = 100;
constexpr auto PARAMS__EVICTION_RATE = 0.5;
constexpr auto PARAMS__WEIGHT_THRESHOLD = 0.05;
constexpr auto PARAMS__TRAINING_GENERATIONS = 100;
constexpr auto PARAMS__TRAINING_SIZE = 2506;
constexpr auto PARAMS__TEST_SIZE = 1671;

// Mutation
constexpr auto MUTATION__WEIGHT_RATE = 0.1;
constexpr auto MUTATION__ADD_NODE = 0.02;
constexpr auto MUTATION__ADD_CONNECTION = 0.05;

// Log
#define LOG_LEVEL 3

#define LOG_VERBOSE LOG_LEVEL < 2
#define LOG_DEBUG LOG_LEVEL < 3
#define LOG_INFO LOG_LEVEL < 4
#define LOG_WARNING LOG_LEVEL < 5
#define LOG_ERROR LOG_LEVEL < 6

#endif // !CONFIGURED
