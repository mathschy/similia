#include <similia/utils/features_library.h>

#include <chrono>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <glog/logging.h>

#include <similia/common/utils/metrics.h>
#include <similia/proto/similia.pb.h>
#include <similia/utils/features_utils.h>



namespace similia {

using std::chrono::steady_clock;

using boost::filesystem::directory_iterator;
using boost::filesystem::path;

using common_utils::Timer;

template <typename Derived>
std::vector<int> SortIndexes(const Eigen::MatrixBase<Derived>& v, bool decreasing, int limit) {
  steady_clock::time_point before_index_vector_creation = steady_clock::now();
  std::vector<int> idx(v.derived().size());
  steady_clock::time_point after_index_vector_creation = steady_clock::now();
  VLOG(1) << "index vector creation took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_index_vector_creation-before_index_vector_creation).count()
      << " us.";
  for (int i = 0; i != idx.size(); ++i) {
    idx[i] = i;
  }
  steady_clock::time_point after_index_vector_initialization = steady_clock::now();
  VLOG(1) << "index vector initialization took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_index_vector_initialization-after_index_vector_creation).count()
      << " us.";
  // sort indexes based on comparing values in v
  if (decreasing) {
    std::nth_element(idx.begin(), idx.begin() + limit, idx.end(), [&v](int i1, int i2) { return v.derived()(i1) > v.derived()(i2); });
    idx.resize(limit);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v.derived()(i1) > v.derived()(i2); });
  } else {
    std::nth_element(idx.begin(), idx.begin() + limit, idx.end(), [&v](int i1, int i2) { return v.derived()(i1) < v.derived()(i2); });
    idx.resize(limit);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v.derived()(i1) < v.derived()(i2); });
  }
  steady_clock::time_point after_index_vector_sort = steady_clock::now();
  VLOG(1) << "index vector sorting took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_index_vector_sort-after_index_vector_initialization).count()
      << " us.";
  return idx;
}

FeaturesLibrary::FeaturesLibrary(const std::string& path_to_features_file,
                                 const int max_num_features,
                                 const int dim_features,
                                 bool normalize_features_on_loading,
                                 bool normalize_features_on_searching) {
  // set the path variable after checking that the paths are OK.
  path path_features(path_to_features_file);
  CHECK(boost::filesystem::is_regular_file(path_features)) << path_features;
  path_to_features_file_ = path_to_features_file;
  CHECK_GT(max_num_features, 0);
  max_num_features_ = max_num_features;
  CHECK_GT(dim_features, 0);
  dim_features_ = dim_features;
  normalize_features_on_searching_ = normalize_features_on_searching;

  // load the features in memory
  Timer timer_load_features("similia.images.loading_features");
  LoadFeatures(normalize_features_on_loading);
  ComputeNorms();
  LOG(INFO) << "Took " << timer_load_features.Stop() << "ms to load features.";
}

std::vector<IdAndComputation> FeaturesLibrary::GetNearestNeighbors(const std::vector<float>& features,
                                                                   int k_nearest) const {
  CHECK_EQ(dim_features_, features.size());
  CHECK_LE(k_nearest, features_.rows());
  Timer timer_get_nn("similia.images.get_nearest_neighbors");
  std::vector<IdAndComputation> nearest_ids;
  steady_clock::time_point before_f_init = steady_clock::now();
  Eigen::VectorXf f = Eigen::VectorXf::Map(&(features[0]), features.size());
  steady_clock::time_point after_f_init = steady_clock::now();
  VLOG(1) << "f init took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_f_init-before_f_init).count()
      << " us.";
  if (normalize_features_on_searching_) {
    f.normalize();
    steady_clock::time_point after_f_normalization = steady_clock::now();
    LOG(INFO) << "f normalization took: "
        << std::chrono::duration_cast<std::chrono::microseconds>(after_f_normalization-after_f_init).count()
        << " us.";
    after_f_init = after_f_normalization;
  } else {
    // don't normalize.
  }

  // we compute the dot product and the squared distances separately because we sort based on squared distances and
  // we return the dot product for further computations.
  Eigen::VectorXf dot = features_ * f;
  Eigen::VectorXf squared_distances = (features_norms_ - 2 * dot).array() + f.squaredNorm();
  steady_clock::time_point after_d_computation = steady_clock::now();
  LOG(INFO) << "d computation took: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(after_d_computation-after_f_init).count()
      << " ms.";
  // we sort by increasing euclidean distances
  std::vector<int> order = SortIndexes(squared_distances, /*decreasing=*/false, k_nearest);
  steady_clock::time_point after_d_sorting = steady_clock::now();
  VLOG(1) << "d sorting took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_d_sorting-after_d_computation).count()
      << " us.";
  IdAndComputation id_and_computation;
  for (int i = 0; i < k_nearest; ++i) {
    id_and_computation.id = feature_ids_[order[i]];
    id_and_computation.dot_product = dot(order[i]);  // we put the dot product here.
    id_and_computation.squared_distance = squared_distances[order[i]];
    nearest_ids.push_back(id_and_computation);
  }
  steady_clock::time_point after_nearest_ids = steady_clock::now();
  VLOG(1) << "nearest_ids building took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_nearest_ids-after_d_sorting).count()
      << " us.";
  LOG(INFO) << "Took " << timer_get_nn.Stop() << "ms to get nearest_neighbors.";
  return nearest_ids;
}

void FeaturesLibrary::LoadFeatures(bool normalize) {
  LoadFeaturesFromfileOrDie(path_to_features_file_, dim_features_, max_num_features_, normalize,
                            &features_, &feature_ids_);
}
void FeaturesLibrary::ComputeNorms() {
  features_norms_ = features_.rowwise().squaredNorm();
  CHECK_EQ(features_norms_.size(), features_.rows());
}
}  // namespace similia
