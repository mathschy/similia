// This file defines various basic utilities (for strings, maps, etc...)
#pragma once

#include <string>

#include <glog/logging.h>

// Serializes a protocol buffer message to string.
template <typename M>
std::string SerializeToStringOrDie(const M& message) {
  std::string r;
  CHECK(message.SerializeToString(&r));
  return r;
}
