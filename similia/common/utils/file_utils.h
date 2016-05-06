#ifndef SIMILIA_COMMON_UTILS_FILE_UTILS_H_
#define SIMILIA_COMMON_UTILS_FILE_UTILS_H_

#include <string>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/path.hpp> 
#include <google/protobuf/message.h>

namespace common_utils {

void WriteToFileOrDie(const boost::filesystem::path& file, const std::string& data);

void WriteToFileOrDie(const boost::filesystem::path& file, const google::protobuf::Message& message);

// reads the content of 'file' as a binary string.
std::string ReadFromFileOrDie(const boost::filesystem::path& file);

}  // namespace common_utils

#endif  // SIMILIA_COMMON_UTILS_FILE_UTILS_H_

