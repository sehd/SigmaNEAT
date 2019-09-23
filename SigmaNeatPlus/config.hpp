#pragma once
#ifndef CONFIGURED
#define CONFIGURED

// System
constexpr auto SYSTEM__USE_GPU = true;
constexpr auto SYSTEM__MAX_GENERATION_COUNT = 1000;
constexpr auto SYSTEM__THREADS_PER_BLOCK = 512;
constexpr auto SYSTEM__LOG_LEVEL = 1; //1 = Verbose -> 5 = Error

// Substrate
constexpr auto SUBSTRATE__DIMENSION = 2;
constexpr auto SUBSTRATE__INPUT_SIZE = 2;
constexpr auto SUBSTRATE__OUTPUT_SIZE = 1;
constexpr auto SUBSTRATE__LAYERS_COUNT = 5;
constexpr auto SUBSTRATE__LAYER_SIZE = 5;

// Params
constexpr auto PARAMS__POPULATION_SIZE = 10;
constexpr auto PARAMS__WEIGHT_THRESHOLD = 0.05;

#endif // !CONFIGURED
