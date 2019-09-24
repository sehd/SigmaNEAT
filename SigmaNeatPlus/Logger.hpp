#pragma once
#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <cuda_runtime.h>
#include "config.hpp"

static class Logger
{
	static char* logLevelNames[];

	__device__ __host__
		static tm* getTime();
public:
	enum LogLevel {
		Verbose = 1,
		Debug = 2,
		Info = 3,
		Warning = 4,
		Error = 5
	};
	
	template<typename T>
	inline
		__device__ __host__
		static void Log(LogLevel t_level, char* t_logIdentifier, char* t_format, T param, bool t_goNextLine = true) {
		if (t_level >= SYSTEM__LOG_LEVEL) {
			tm* timeInfo = getTime();
			printf("%02d:%02d:%02d: (%s) [%s] ", timeInfo->tm_hour, timeInfo->tm_min,
				timeInfo->tm_sec, logLevelNames[t_level], t_logIdentifier);
			printf(t_value, param);
			if (t_goNextLine)
				printf("\n");
		}
	}

	__device__ __host__
		static void Log(LogLevel t_level, char* t_logIdentifier, char* t_format, bool t_goNextLine = true);
};

#endif // !LOGGER_H

