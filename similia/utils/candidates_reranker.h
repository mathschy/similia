#pragma once

#include "similia/utils/candidates_finder.h"

namespace similia {

class CandidatesReranker {
 public:
  CandidatesReranker(const std::string& root_path_to_indexing_clusters_features,
                     const std::string& root_path_to_compressing_clusters_features,
                     const std::string& path_to_indexing_rotation_matrix,
                     const std::string& path_to_compressing_rotation_matrix);

  std::vector<IdAndComputation> RerankCandidates(const std::vector<Candidate>& candidates,
                                                 const std::vector<float>& query_features,
                                                 std::size_t num_nearest);

 private:
  // 8 matrices of shape 2^8 = 256, 1024/8 = 128
  // These features are in the compressing-level-rotated space. (They are stored in this space.)
  std::vector<Eigen::MatrixXf> compressing_clusters_features_;

  // 2 matrices of shape 2^14 = 16384, 1024/2 = 512
  // Those features are rotated on loading to be in the compressing-level-rotated space. (They are stored in the
  // indexing-level-rotated space.)
  std::vector<Eigen::MatrixXf> indexing_clusters_features_;

  // kCompressingDimensionDivision-vector of
  // (kNumIndexingClustersPerDimensionDivision * kNumCompressingClustersPerDimensionDivision) matrices.
  // norm_table_[j](indexing_id(i(j)), compressing_id(j)) is the norm of the corresponding subvector
  // (of length kFeatureDimensions / kCompressingDimensionDivision).
  // This contains the sum of both indexing-level divisions multiplied by Rc corresponding norms.
  std::vector<Eigen::MatrixXf> norm_table_;

  // Rotation matrix for the indexing-level rotation
  Eigen::MatrixXf indexing_rotation_matrix_;

  // Rotation matrix for the compressing-level rotation
  Eigen::MatrixXf compressing_rotation_matrix_;
};

}  // namespace similia
