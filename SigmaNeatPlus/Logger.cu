#include <time.h>
#include "Logger.hpp"

char* Logger::logLevelNames[] = { "","Verbose","Debug","Info","Warning","Error" };

tm* Logger::getTime() {
	time_t rawtime;
	time(&rawtime);
	return localtime(&rawtime);
}

void Logger::Log(Logger::LogLevel t_level, char* t_logIdentifier, char* t_format, bool t_goNextLine) {
	if (t_level >= SYSTEM__LOG_LEVEL) {
		tm* timeInfo = getTime();
		printf("%02d:%02d:%02d: (%s) [%s] %s", timeInfo->tm_hour, timeInfo->tm_min,
			timeInfo->tm_sec, logLevelNames[t_level], t_logIdentifier, t_format);
		if (t_goNextLine)
			printf("\n");
	}
}