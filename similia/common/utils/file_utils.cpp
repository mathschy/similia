#include "similia/common/utils/file_utils.h"

#include <stdio.h>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/path.hpp>
#include <glog/logging.h>

namespace common_utils {
using boost::filesystem::path;

void WriteToFileOrDie(const path& file, const std::string& data) {
  FILE* fp = fopen(file.string().c_str(), "wb");
  CHECK(fp != nullptr) << "couldn't open file: " << file;
  PCHECK(fwrite(data.data(), sizeof(char), data.size(), fp) == data.size());
  PCHECK(fclose(fp) == 0);
}

void WriteToFileOrDie(const path& file, const google::protobuf::Message& message) {
  std::string data;
  CHECK(message.SerializeToString(&data));
  WriteToFileOrDie(file, data);
}

std::string ReadFromFileOrDie(const path& file) {
  boost::filesystem::ifstream in;
  in.open(file, std::ios::binary | std::ios::in);
  CHECK(in) << "file couldn't be opened for reading: " << file.string();

  std::stringstream out;
  out << in.rdbuf();
  in.close();
  return out.str();
}

}  // namespace common_utils
