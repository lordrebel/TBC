#pragma once
#include "llvm/Support/raw_ostream.h"
#include <string>

#define LOG_PREFIX(LOGLEVEL)                                                   \
  "[" + std::string(tbc::support::getLogLevelName(LOGLEVEL)) + " " +           \
      std::string(tbc::support::getFileName(__FILE__)) + ":" + std::to_string(__LINE__) +   \
      " " + std::string("]: ") \

#define VLOG_PREFIX(LOGLEVEL)                                                  \
  "[" + std::string(tbc::support::getVLogLevelName(LOGLEVEL)) + " " +          \
      std::string(tbc::support::getFileName(__FILE__)) + ":" + std::to_string(__LINE__) +   \
      " " + std::string("]: ") \

#define LOG(LOGLEVEL)                                                          \
  tbc::support::getLogStream(LOGLEVEL) << LOG_PREFIX(LOGLEVEL) \

#define VLOG(i)                                                         \
  tbc::support::getVLogStream(tbc::support::int2Vloglevel(i)) << VLOG_PREFIX(tbc::support::int2Vloglevel(i)) \

#define LOGF LOG(tbc::support::LogLevel::FATAL)

#define LOGE LOG(tbc::support::LogLevel::ERROR)

#define LOGW LOG(tbc::support::LogLevel::WARNING)

#define LOGI LOG(tbc::support::LogLevel::INFO)

#define LOGD LOG(tbc::support::LogLevel::DEBUG)
#define LOGT LOG(tbc::support::LogLevel::TRACE)

namespace tbc {
namespace support {
auto constexpr LOG_ENV = "TBC_LOG_LEVEL";
auto constexpr VLOG_ENV = "TBC_VLOG_LEVEL";
auto constexpr LOG_COLORED = "TBC_LOG_COLOR";
enum LogLevel {
  MAX = 3,
  TRACE = 3,
  DEBUG = 2,
  INFO = 1,
  WARNING = 0,
  ERROR = -1,
  FATAL = -2,
};
enum VLogLevel {
  VLOG9 = 9,
  VLOG8 = 8,
  VLOG7 = 7,
  VLOG6 = 6,
  VLOG5 = 5,
  VLOG4 = 4,
  VLOG3 = 3,
  VLOG2 = 2,
  VLOG1 = 1,
  NON_VLOG=0,
};
std::string getLogLevelName(LogLevel level);
std::string getVLogLevelName(VLogLevel level);
void initLogger();
llvm::raw_ostream &getLogStream(LogLevel level);
llvm::raw_ostream &getVLogStream(VLogLevel level);
std::string_view getFileName(std::string_view path);

VLogLevel int2Vloglevel(int level);

} // namespace support
} // namespace tbc
