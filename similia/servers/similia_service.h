#ifndef SIMILIA_SERVERS_SIMILIA_SERVICE_H_
#define SIMILIA_SERVERS_SIMILIA_SERVICE_H_

#include <similia/proto/similia.grpc.pb.h>
#include <similia/utils/candidates_finder.h>
#include <similia/utils/candidates_reranker.h>
#include <similia/utils/features_library.h>
#include <similia/utils/matrix_utils.h>

namespace similia {

class SimiliaService final: public proto::Similia::Service {
 public:
  SimiliaService(CandidatesFinder* cf, CandidatesReranker* cr,
                 const std::string& path_to_indexing_rotation_matrix,
                 const std::string& path_to_compressing_rotation_matrix) :
      candidates_finder_(cf), candidates_reranker_(cr) {
    LoadFloatMatrixFromFileOrDie(path_to_indexing_rotation_matrix, &indexing_rotation_matrix_);
    LoadFloatMatrixFromFileOrDie(path_to_compressing_rotation_matrix, &compressing_rotation_matrix_);
  }

  grpc::Status SimiliaSearch(grpc::ServerContext* context,
                             const proto::SimiliaSearchRequest* request,
                             proto::SimiliaSearchResponse* response) override;

 private:
  FeaturesLibrary* features_library_;  // not owned
  CandidatesFinder* candidates_finder_;  // not owned
  CandidatesReranker* candidates_reranker_;  // not owned

  // Rotation matrix for the indexing-level rotation
  Eigen::MatrixXf indexing_rotation_matrix_;
  // Rotation matrix for the compressing-level rotation
  Eigen::MatrixXf compressing_rotation_matrix_;
};
}  // namespace similia

#endif  // SIMILIA_SERVERS_SIMILIA_SERVICE_H_

