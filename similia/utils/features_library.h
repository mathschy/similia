#ifndef SIMILIA_UTILS_FEATURES_LIBRARY_H
#define SIMILIA_UTILS_FEATURES_LIBRARY_H

#include <string>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

namespace similia {
struct IdAndComputation {
  std::string id;
  float dot_product;  // dot product between features and the (euclidean) nearest features in the library
  float squared_distance;  // squared euclidean distance between features and the nearest features in the library
};

class FeaturesLibrary {
 public:
  FeaturesLibrary(const std::string& path_to_features_file,
                  const int max_num_features,
                  const int dim_features,
                  bool normalize_features_on_loading,
                  bool normalize_features_on_searching);

  // takes a vector of features and return the k_nearest ids ordered by increased euclidean distance.
  std::vector<IdAndComputation> GetNearestNeighbors(const std::vector<float>& features, int k_nearest) const;

 private:
  void LoadFeatures(bool normalize);
  void ComputeNorms();

  std::string path_to_features_file_;
  Eigen::MatrixXf features_;  // shape min(num_images, max_num_features), dim_features
  Eigen::VectorXf features_norms_;  // length min(num_images, max_num_features)
  int max_num_features_;
  std::vector<std::string> feature_ids_;  // length num_images
  int dim_features_;  // dimension of the features
  bool normalize_features_on_searching_;

};
}  // namespace similia

#endif  // SIMILIA_UTILS_FEATURES_LIBRARY_H
