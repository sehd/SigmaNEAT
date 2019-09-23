#pragma once
#ifndef LOGGER_H
#define LOGGER_H

#include <cuda_runtime.h>

static class Logger
{
public:
	enum LogLevel {
		Verbose = 1,
		Debug = 2,
		Info = 3,
		Warning = 4,
		Error = 5
	};

	template<class T>
	__device__ __host__
		void Log(LogLevel t_level, char* t_logIdentifier, char* t_format, T param, bool t_goNextLine = true);
};

#endif // !LOGGER_H

