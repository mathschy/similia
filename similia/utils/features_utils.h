#ifndef SIMILIA_UTILS_FEATURES_UTILS_H
#define SIMILIA_UTILS_FEATURES_UTILS_H

#include <fstream>

#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>

namespace similia {

static const int kFeatureDimensions = 1024;
static const int kIndexingDimensionDivision = 2;  // don't modify this.
static const int kCompressingDimensionDivision = 32;
static const int kNumIndexingClustersPerDimensionDivision = 16384;
static const int kNumCompressingClustersPerDimensionDivision = 256;
static const int kIndexingSubfeaturesDimensions = kFeatureDimensions / kIndexingDimensionDivision;
static const int kCompressingSubfeaturesDimensions = kFeatureDimensions / kCompressingDimensionDivision;

// parse the line from a csv file with id, feature1, ... featureN
void ReadFeaturesLineOrDie(const std::string& line, std::vector<float>* features, std::string* id);

// parse a csv file with id, feature1,.., featureN into a matrix features(num_features, dim_features) and a list of ids.
template <typename Derived>
void LoadFeaturesFromfileOrDie(const std::string &path_to_features_file, int dim_features, int num_features,
                               bool normalize, Eigen::MatrixBase<Derived>* features, std::vector<std::string>* ids) {
  using boost::filesystem::path;

  path features_path(path_to_features_file);
  LOG(INFO) << "loading features from file: " << features_path;
  std::vector<std::vector<float>> features_vec;
  int count_features = 0;
  std::ifstream data(features_path.string());
  std::string line;
  while (std::getline(data, line) && count_features < num_features) {
    ids->emplace_back();
    features_vec.emplace_back();
    features_vec.back().resize(dim_features);
    ReadFeaturesLineOrDie(line, &features_vec.back(), &ids->back());
    count_features++;
  }
  CHECK_EQ(count_features, num_features);
  features->derived().resize(features_vec.size(), dim_features);
  for (std::size_t i_row = 0; i_row < features_vec.size(); ++i_row) {
    features->derived().row(i_row) =
        Eigen::RowVectorXf::Map(&(features_vec[i_row][0]), features_vec[i_row].size());
    if (normalize) {
      features->derived().row(i_row).normalize();
    } else {
      // we don't normalize.
    }
  }
  LOG(INFO) << "loaded features matrix of size: " << features->derived().rows() << "x" << features->derived().cols();
  LOG(INFO) << "frobenius norm: " << features->derived().norm();
};

void LoadClustersFeatures(const std::string& root_path,
                          int feature_dimensions,
                          int dimension_division,
                          int num_clusters_per_dimension_division,
                          std::vector<Eigen::MatrixXf>* clusters_features);

}  // namespace similia

#endif  // SIMILIA_UTILS_FEATURES_UTILS_H
