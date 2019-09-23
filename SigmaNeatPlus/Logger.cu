#include <stdio.h>
#include "Logger.hpp"
#include "config.hpp"

template<class T>
void Logger::Log(LogLevel t_level, char* t_logIdentifier, char* t_format, T param, bool t_goNextLine = true) {
	if (t_level >= SYSTEM__LOG_LEVEL) {
		printf(t_value, param);
		if (t_goNextLine)
			printf("\n");
	}
}