#ifndef SIMILIA_COMMON_UTILS_METRICS_H_
#define SIMILIA_COMMON_UTILS_METRICS_H_

#include <stdint.h>
#include <chrono>
#include <string>


namespace common_utils {

class MetricsImplementation {
 public:
  virtual void Time(const std::string& name, int64_t millis) {};

  static MetricsImplementation* instance_;
};

// Helper class to time events: the timer starts when the object is created, and calls MetricsImplementation::Time
// when it is destroyed. Thus all timers must be destroyed before MetricsImplementation::instance_ is destroyed.
// It is not necessary to manually stop the timer but it is possible in case one wants
// to manually access the value.
// The timer is not thread-safe and shouldn't be started/stopped in 2 different threads.
class Timer {
 public:
  explicit Timer(const std::string& name);

  ~Timer();

  // stops the timer and returns the time elapsed in millis.
  int64_t Stop();

 private:
  int64_t ElapsedMs();

  const std::string name_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point end_;
  bool stopped_{false};

};
}  // namespace common_utils

#endif  // SIMILIA_COMMON_UTILS_METRICS_H_

