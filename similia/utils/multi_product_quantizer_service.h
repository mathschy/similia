#ifndef SIMILIA_UTILS_MULTI_PRODUCT_QUANTIZER_SERVICE_H_
#define SIMILIA_UTILS_MULTI_PRODUCT_QUANTIZER_SERVICE_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <similia/proto/similia.grpc.pb.h>
#include <similia/utils/features_utils.h>
#include <similia/utils/matrix_utils.h>


namespace similia {

class MultiProductQuantizerService final: public proto::MultiProductQuantizer::Service {
 public:
  MultiProductQuantizerService(const std::string& root_path_to_indexing_clusters_features,
                               const std::string& root_path_to_compressing_clusters_features,
                               const std::string& path_to_indexing_rotation_matrix,
                               const std::string& path_to_compressing_rotation_matrix) {
    LoadClustersFeatures(root_path_to_indexing_clusters_features, kFeatureDimensions, kIndexingDimensionDivision,
                         kNumIndexingClustersPerDimensionDivision, &indexing_clusters_features_);
    LoadClustersFeatures(root_path_to_compressing_clusters_features, kFeatureDimensions, kCompressingDimensionDivision,
                         kNumCompressingClustersPerDimensionDivision, &compressing_clusters_features_);
    LoadFloatMatrixFromFileOrDie(path_to_indexing_rotation_matrix, &indexing_rotation_matrix_);
    LoadFloatMatrixFromFileOrDie(path_to_compressing_rotation_matrix, &compressing_rotation_matrix_);
    std::chrono::steady_clock::time_point before_norms = std::chrono::steady_clock::now();
    for (int i = 0; i < indexing_clusters_features_.size(); ++i) {
      indexing_clusters_norms_.push_back(indexing_clusters_features_[i].rowwise().squaredNorm());
      CHECK_EQ(kNumIndexingClustersPerDimensionDivision, indexing_clusters_norms_.back().rows());
      CHECK_EQ(1, indexing_clusters_norms_.back().cols());
    }
    for (int i = 0; i < compressing_clusters_features_.size(); ++i) {
      compressing_clusters_norms_.push_back(compressing_clusters_features_[i].rowwise().squaredNorm());
      CHECK_EQ(kNumCompressingClustersPerDimensionDivision, compressing_clusters_norms_.back().rows());
      CHECK_EQ(1, compressing_clusters_norms_.back().cols());
    }
    std::chrono::steady_clock::time_point after_norms = std::chrono::steady_clock::now();
    LOG(INFO) << "norms computation took: "
        << std::chrono::duration_cast<std::chrono::microseconds>(after_norms-before_norms).count()
        << " us.";
  };

  grpc::Status Quantize(grpc::ServerContext* context,
                        const proto::QuantizationRequest* request,
                        proto::QuantizationResponse* response) override;

 private:
  // 8 matrices of shape 2^8 = 256, 1024/8 = 128
  std::vector<Eigen::MatrixXf> compressing_clusters_features_;

  // 2 matrices of shape 2^14 = 16384, 1024/2 = 512
  std::vector<Eigen::MatrixXf> indexing_clusters_features_;

  // 2 vector of length 16384 that store the norms of the indexing clusters features
  std::vector<Eigen::VectorXf> indexing_clusters_norms_;

  // 8 vector of length 256 that store the norms of the compressing clusters features
  std::vector<Eigen::VectorXf> compressing_clusters_norms_;

  // Rotation matrix for the indexing-level rotation
  Eigen::MatrixXf indexing_rotation_matrix_;

  // Rotation matrix for the compressing-level rotation
  Eigen::MatrixXf compressing_rotation_matrix_;
};
}  // namespace similia

#endif  // SIMILIA_UTILS_MULTI_PRODUCT_QUANTIZER_SERVICE_H_

