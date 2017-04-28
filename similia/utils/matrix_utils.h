#pragma once

#include <fstream>

#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>

#include "similia/common/utils/file_utils.h"
#include "similia/proto/similia.pb.h"

namespace similia {

template<typename Derived>
void LoadFloatMatrixFromFileOrDie(const std::string& pb_matrix_file, Eigen::MatrixBase<Derived>* float_matrix) {
  std::string data = common_utils::ReadFromFileOrDie(pb_matrix_file);
  proto::FloatMatrix float_matrix_pb;
  float_matrix_pb.ParseFromString(data);
  CHECK_EQ(float_matrix_pb.num_rows() * float_matrix_pb.num_cols(), float_matrix_pb.element_size());
  float_matrix->derived().resize(float_matrix_pb.num_rows(), float_matrix_pb.num_cols());
  for (int i = 0; i < float_matrix_pb.num_rows(); ++i) {
    for (int j = 0; j < float_matrix_pb.num_cols(); ++j) {
      float_matrix->derived()(i, j) = float_matrix_pb.element(i * float_matrix_pb.num_rows() + j);
    }
  }
}

template<typename Derived>
void SaveFloatMatrixToFileOrDie(const Eigen::MatrixBase<Derived>& float_matrix, const std::string& pb_matrix_file) {
  proto::FloatMatrix float_matrix_pb;
  int num_rows = float_matrix.derived().rows();
  int num_cols = float_matrix.derived().cols();
  float_matrix_pb.set_num_rows(num_rows);
  float_matrix_pb.set_num_cols(num_cols);
  float_matrix_pb.mutable_element()->Reserve(num_rows * num_cols);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      float_matrix_pb.add_element(float_matrix.derived()(i, j));
    }
  }
  CHECK_EQ(float_matrix_pb.num_rows() * float_matrix_pb.num_cols(), float_matrix_pb.element_size());
  common_utils::WriteToFileOrDie(pb_matrix_file, float_matrix_pb);
}

}  // namespace similia
