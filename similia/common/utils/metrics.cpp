#include "similia/common/utils/metrics.h"

#include <gflags/gflags.h>
#include <glog/logging.h>


namespace common_utils {

MetricsImplementation* MetricsImplementation::instance_ = nullptr;


Timer::Timer(const std::string& name)
    : name_(name),
      start_(std::chrono::steady_clock::now()) {
}

int64_t Timer::Stop() {
  if (!stopped_) {
    stopped_ = true;
    end_ = std::chrono::steady_clock::now();
  }
  return ElapsedMs();
}

int64_t Timer::ElapsedMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      end_ - start_).count();
}

Timer::~Timer() {
  int64_t elapsed_ms;
  if (!stopped_) {
    elapsed_ms = Stop();
  } else {
    elapsed_ms = ElapsedMs();
  }
  CHECK(MetricsImplementation::instance_ != nullptr) << "All timers must be destroyed before MetricsImplementation::instance_ is destroyed";
  MetricsImplementation::instance_->Time(name_, elapsed_ms);
}
}  // namespace common_utils
