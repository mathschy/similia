#include <similia/utils/candidates_reranker.h>

#include <chrono>
#include <fstream>

#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <boost/token_functions.hpp>
#include <boost/tokenizer.hpp>

#include <similia/utils/features_utils.h>
#include <similia/utils/matrix_utils.h>


namespace similia {
using std::chrono::steady_clock;

using boost::filesystem::path;

template<typename T>
std::vector<int> SortIndexes(const std::vector<T>& v) {
  // Sort by increasing values of v
  // initialize original index locations
  std::vector<int> idx(v.size());
  for (int i = 0; i != idx.size(); ++i) {
    idx[i] = i;
  }

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
            [&v](int i1, int i2) { return v[i1] < v[i2]; });

  return idx;
}

CandidatesReranker::CandidatesReranker(const std::string& root_path_to_indexing_clusters_features,
                                       const std::string& root_path_to_compressing_clusters_features,
                                       const std::string& path_to_indexing_rotation_matrix,
                                       const std::string& path_to_compressing_rotation_matrix) {
  LoadFloatMatrixFromFileOrDie(path_to_indexing_rotation_matrix, &indexing_rotation_matrix_);
  LoadFloatMatrixFromFileOrDie(path_to_compressing_rotation_matrix, &compressing_rotation_matrix_);
  LoadClustersFeatures(root_path_to_indexing_clusters_features, kFeatureDimensions, kIndexingDimensionDivision,
                       kNumIndexingClustersPerDimensionDivision, &indexing_clusters_features_);

  // the features rotated are computed by padding the subfeatures with 0 and applying Rc. The resulting features are of size 1024.
  std::vector<Eigen::MatrixXf> indexing_clusters_features_rotated;
  indexing_clusters_features_rotated.resize(2);  // 2 dimension divisions.
  Eigen::MatrixXf indexing_features_padded = Eigen::MatrixXf::Zero(kNumIndexingClustersPerDimensionDivision,
                                                                   kFeatureDimensions);
  indexing_features_padded.leftCols(kIndexingSubfeaturesDimensions) = indexing_clusters_features_[0];
  indexing_clusters_features_rotated[0] = indexing_features_padded * compressing_rotation_matrix_;
  indexing_features_padded = Eigen::MatrixXf::Zero(kNumIndexingClustersPerDimensionDivision, kFeatureDimensions);
  indexing_features_padded.rightCols(kIndexingSubfeaturesDimensions) = indexing_clusters_features_[1];
  indexing_clusters_features_rotated[1] = indexing_features_padded * compressing_rotation_matrix_;

  LoadClustersFeatures(root_path_to_compressing_clusters_features, kFeatureDimensions, kCompressingDimensionDivision,
                       kNumCompressingClustersPerDimensionDivision, &compressing_clusters_features_);
  LOG(INFO) << "loaded compressing_clusters_features_";
  // We loop over the subfeatures with the smallest dimension (the compressing ones).
  // For clarity we declare 4 eigen variables and then do the computation. This limits the potential optimization done
  // by eigen but this is done only once at the server start.
  // We do this 2 times, because we cache results for both indexing-level dimension divisions.
  std::vector<std::vector<Eigen::MatrixXf>> norm_tables;
  norm_tables.resize(kIndexingDimensionDivision);
  for (int i_index = 0; i_index < kIndexingDimensionDivision; ++i_index) {
    VLOG(1) << "i_index=" << i_index;
    norm_tables[i_index].resize(kCompressingDimensionDivision);
    for (int i_compress = 0; i_compress < kCompressingDimensionDivision; ++i_compress) {
      VLOG(1) << "i_compress=" << i_compress;
      int start_column = i_compress * kCompressingSubfeaturesDimensions;

      Eigen::MatrixXf indexing_sub_features = indexing_clusters_features_rotated[i_index]
          .block(0, start_column, kNumIndexingClustersPerDimensionDivision, kCompressingSubfeaturesDimensions);
      Eigen::MatrixXf compressing_sub_features = compressing_clusters_features_[i_compress];
      Eigen::VectorXf indexing_sub_norms;
      // The indexing sub norms are computed in the indexing-level rotated space.
      // We compute them only one time depending on values of indexes.
      if (i_index == 0) {
        if (i_compress < kCompressingDimensionDivision / kIndexingDimensionDivision) {
          indexing_sub_norms = indexing_clusters_features_[i_index]
              .block(0, start_column, kNumIndexingClustersPerDimensionDivision, kCompressingSubfeaturesDimensions)
              .rowwise().squaredNorm();
        } else {
          indexing_sub_norms = Eigen::VectorXf::Zero(kNumIndexingClustersPerDimensionDivision);
        }
      } else {
        if (i_compress >= kCompressingDimensionDivision / kIndexingDimensionDivision) {
          indexing_sub_norms = indexing_clusters_features_[i_index]
              .block(0, start_column - kIndexingSubfeaturesDimensions,
                     kNumIndexingClustersPerDimensionDivision, kCompressingSubfeaturesDimensions)
              .rowwise().squaredNorm();
        } else {
          indexing_sub_norms = Eigen::VectorXf::Zero(kNumIndexingClustersPerDimensionDivision);
        }
      }

      Eigen::VectorXf compressing_sub_norms = compressing_sub_features.rowwise().squaredNorm();
      norm_tables[i_index][i_compress] = ((2 * indexing_sub_features * compressing_sub_features.transpose())
          .colwise() + indexing_sub_norms)
          .rowwise() + 0.5 * compressing_sub_norms.transpose();  // 0.5 because we counted 2 times the compressing norms
    }
  }
  norm_table_.reserve(kCompressingDimensionDivision);
  for (int i_compress = 0; i_compress < kCompressingDimensionDivision; ++i_compress) {
    norm_table_.push_back(norm_tables[0][i_compress] + norm_tables[1][i_compress]);
  }
}

std::vector<IdAndComputation> CandidatesReranker::RerankCandidates(const std::vector<Candidate>& candidates,
                                                                   const std::vector<float>& query_features,
                                                                   int num_nearest) {
  steady_clock::time_point before_table = steady_clock::now();
  // We rotate the query features with Rc (they already have been rotated with Ri in candidates_finder).
  Eigen::Matrix<float, kFeatureDimensions, 1> features = Eigen::VectorXf::Map(&query_features[0], query_features.size());
  // We want features to be a column vector. x^T * R = R^T * x
  features = compressing_rotation_matrix_.transpose() * features;
  // for each vector of 128 compute scalar product with its corresponding compressing_clusters_features_
  // store the results in a table
  // scalar product between query subdim i with each of its compressing cluster j.
  Eigen::Matrix<float, kCompressingDimensionDivision, kNumCompressingClustersPerDimensionDivision> table;
  for (int i = 0; i < kCompressingDimensionDivision; ++i) {
    table.row(i) = compressing_clusters_features_[i] * features.segment(i * kCompressingSubfeaturesDimensions,
                                                                        kCompressingSubfeaturesDimensions);
  }
  steady_clock::time_point after_table = steady_clock::now();
  LOG(INFO) << "precomputed table in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(after_table - before_table).count()
            << "ms";

  VLOG(2) << "table: " << table;
  // compute the scalar product between query features and each of the candidate using the precomputed table
  std::vector<float> squared_distances(candidates.size(), 1.0);
  int i_index;
  int ratio = kCompressingDimensionDivision / kIndexingDimensionDivision;
  // squared_distance is ||candidate_features||^2 - 2 * <candidate_features, query_features> + ||query_features||^2
  // candidate_features is approximated by its closest clusters assignments.
  // ||candidate_features||^2 is read from a table that was precomputed at class construction.
  for (int i_candidate = 0; i_candidate < candidates.size(); ++i_candidate) {
    // This is ||query_features||^2 - part of 2 * <candidate_features, query_features>
    squared_distances[i_candidate] -= 2 * candidates[i_candidate].dot_product;
  }
  for (int i_compress = 0; i_compress < kCompressingDimensionDivision; ++i_compress) {
    i_index = i_compress / ratio;
    for (int i_candidate = 0; i_candidate < candidates.size(); ++i_candidate) {
      int j = static_cast<int>(candidates[i_candidate].compressing_clusters_ids[i_compress]);
      // This is part of 2 * <candidate_features, query_features>
      squared_distances[i_candidate] -= 2 * table(i_compress, j);
      // This is part of ||candidate_features||^2
      squared_distances[i_candidate] +=
          norm_table_[i_compress](candidates[i_candidate].indexing_clusters_ids[i_index], j);
    }
  }
  steady_clock::time_point after_distances = steady_clock::now();
  LOG(INFO) << "computed distances in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(after_distances - after_table).count()
            << "ms";
  std::vector<int> sort = SortIndexes(squared_distances);
  IdAndComputation id_and_computation;
  std::vector<IdAndComputation> nearest_candidates = {};
  for (int s : sort) {
    id_and_computation.id = candidates[s].image_id;
    id_and_computation.squared_distance = squared_distances[s];
    nearest_candidates.push_back(id_and_computation);
    if (nearest_candidates.size() == num_nearest) {
      break;
    }
  }

  return nearest_candidates;
}



}  // namespace similia
