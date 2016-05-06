// This file defines various basic utilities (for strings, maps, etc...)
#ifndef COMMON_UTILS_BASE_H_
#define COMMON_UTILS_BASE_H_

#include <string>

#include <glog/logging.h>

// Serializes a protocol buffer message to string.
template <typename M>
std::string SerializeToStringOrDie(const M& message) {
  std::string r;
  CHECK(message.SerializeToString(&r));
  return r;
}

#endif  // COMMON_UTILS_BASE_H_
