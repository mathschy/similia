#include "similia/utils/features_utils.h"

#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <glog/logging.h>


namespace similia {
using boost::filesystem::path;

void ReadFeaturesLineOrDie(const std::string& line, std::vector<float>* features, std::string* id) {
  int dim_features = features->size();
  boost::char_separator<char> sep(",");
  boost::tokenizer<boost::char_separator<char>> values_str = boost::tokenizer<boost::char_separator<char>>(line, sep);
  int i = 0;
  for (auto it : values_str) {
    if (i == 0) {
      path pathit(it);
      *id = pathit.filename().stem().string();
    } else if (i <= dim_features) {  // 1 offset for the id
      features->at(i - 1) = (float) atof(it.c_str());
    } else {
      LOG(FATAL) << "the features in features_file are too high dimensional, "
          << "they should be " << dim_features << " dimensional";
    }
    ++i;
  }
  CHECK_EQ(i - 1, dim_features) << "feature in features_file don't have the right dimension";  // 1 offset for the id
}

// Load the clusters features from multiple .csv file named root_path_i.csv where 0 <= i < dimension_division.
// don't normalize
void LoadClustersFeatures(const std::string& root_path,
                          int feature_dimensions,
                          int dimension_division,
                          int num_clusters_per_dimension_division,
                          std::vector<Eigen::MatrixXf>* clusters_features) {
  std::string path_clusters_feature;
  std::vector<std::string> ids;  // this is not used but needed to use features_utils
  for (int i = 0; i < dimension_division; ++i) {
    path_clusters_feature = root_path + std::to_string(i) + ".csv";
    clusters_features->emplace_back();
    LoadFeaturesFromfileOrDie(path_clusters_feature, feature_dimensions / dimension_division,
                              num_clusters_per_dimension_division, false,
                              &clusters_features->back(), &ids);
  }
}


}  // namespace similia
