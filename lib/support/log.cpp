#include "support/log.h"
#include <climits>

namespace tbc {
namespace support {

static LogLevel currentLogLevel = LogLevel::WARNING;
static VLogLevel currentVLogLevel = VLogLevel::NON_VLOG;
static bool logColored = true;
std::string getLogLevelName(LogLevel level) {
  switch (level) {
  case TRACE:
    return "TRACE";
  case DEBUG:
    return "DEBUG";
  case INFO:
    return "INFO";
  case WARNING:
    return "WARNING";
  case ERROR:
    return "ERROR";
  case FATAL:
    return "FATAL";
  default:
    return "UNKNOWN";
  }
}
std::string getVLogLevelName(VLogLevel level) {
  switch (level) {
  case VLOG9:
    return "VLOG9";
  case VLOG8:
    return "VLOG8";
  case VLOG7:
    return "VLOG7";
  case VLOG6:
    return "VLOG6";
  case VLOG5:
    return "VLOG5";
  case VLOG4:
    return "VLOG4";
  case VLOG3:
    return "VLOG3";
  case VLOG2:
    return "VLOG2";
  case VLOG1:
    return "VLOG1";
  default:
    return "NON_VLOG";
  }
}
int getEnvVar(const std::string &varName, int defaultValue) {
  const char *value = std::getenv(varName.c_str());
  if (!value) {
    return defaultValue;
  }

  // 使用 strtol 替代 std::stoi，不会抛异常
  char *endptr;
  errno = 0; // 重置errno
  long result = std::strtol(value, &endptr, 10);

  // 检查转换是否成功
  if (errno == ERANGE || result > INT_MAX || result < INT_MIN) {
    // 数值超出int范围
    return defaultValue;
  }

  if (endptr == value || *endptr != '\0') {
    // 转换失败或者字符串包含非数字字符
    return defaultValue;
  }

  return static_cast<int>(result);
}
void initLogger() {
  // for log
  int value = getEnvVar(LOG_ENV, static_cast<int>(LogLevel::WARNING));
  if (value > static_cast<int>(LogLevel::MAX)) {
    value = static_cast<int>(LogLevel::MAX);
  } else if (value < static_cast<int>(LogLevel::WARNING)) {
    value = static_cast<int>(LogLevel::WARNING);
  }
  currentLogLevel = static_cast<LogLevel>(value);

  // for vlog
  value = getEnvVar(VLOG_ENV, static_cast<int>(VLogLevel::NON_VLOG));
  if (value > static_cast<int>(VLogLevel::VLOG9)) {
    value = static_cast<int>(VLogLevel::VLOG9);
  } else if (value < static_cast<int>(VLogLevel::NON_VLOG)) {
    value = static_cast<int>(VLogLevel::NON_VLOG);
  }
  currentVLogLevel = static_cast<VLogLevel>(value);
  // for log color
  logColored = getEnvVar(LOG_COLORED, 1) > 0;
}

std::string_view getFileName(std::string_view path) {
  return path.substr(path.find_last_of("/\\") + 1);
}
VLogLevel int2Vloglevel(int level) {
  if (level >= static_cast<int>(VLogLevel::VLOG9))
    return VLogLevel::VLOG9;
  if (level <= static_cast<int>(VLogLevel::VLOG1))
    return VLogLevel::VLOG1;
  return static_cast<VLogLevel>(level);
}
llvm::raw_ostream::Colors getLogLevelColor(LogLevel level) {
  switch (level) {
  case FATAL:
    return llvm::raw_ostream::Colors::BRIGHT_RED;
  case ERROR:
    return llvm::raw_ostream::Colors::RED;
  case WARNING:
    return llvm::raw_ostream::Colors::YELLOW;
  case INFO:
    return llvm::raw_ostream::Colors::GREEN;
  case DEBUG:
    return llvm::raw_ostream::Colors::CYAN;
  case TRACE:
    return llvm::raw_ostream::Colors::BLUE;
  default:
    return llvm::raw_ostream::Colors::WHITE;
  }
}
llvm::raw_ostream::Colors getVLogLevelColor(VLogLevel level) {
  switch (level) {
  case VLOG9:
  case VLOG8:
  case VLOG7:
    return llvm::raw_ostream::Colors::BRIGHT_MAGENTA;
  case VLOG6:
  case VLOG5:
  case VLOG4:
    return llvm::raw_ostream::Colors::MAGENTA;
  case VLOG3:
  case VLOG2:
  case VLOG1:
    return llvm::raw_ostream::Colors::BRIGHT_BLUE;
  default:
    return llvm::raw_ostream::Colors::WHITE;
  }
}

llvm::raw_ostream &getLogStream(LogLevel level) {
  auto &&error_stram = llvm::errs();
  auto &&out_stram = llvm::outs();
  auto &&null_stram = llvm::nulls();
  if (logColored) {
    error_stram.changeColor(getLogLevelColor(level));
    out_stram.changeColor(getLogLevelColor(level));
  }
  if (level < LogLevel::WARNING)
    return error_stram;

  if (level > currentLogLevel)
    return null_stram;
  if (level <= currentLogLevel)
    return out_stram;
  return null_stram;
}
llvm::raw_ostream &getVLogStream(VLogLevel level) {
  auto &&out_stram = llvm::outs();
  auto &&null_stram = llvm::nulls();
  if (logColored) {
    out_stram.changeColor(getVLogLevelColor(level));
  }

  if (level > currentVLogLevel)
    return null_stram;
  if (level <= currentVLogLevel)
    return out_stram;
  return llvm::nulls();
}
} // namespace support
} // namespace tbc
