#include <similia/servers/similia_service.h>

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <similia/common/utils/metrics.h>

using grpc::Status;
using grpc::StatusCode;
using grpc::ServerContext;
using common_utils::Timer;


namespace similia {

static const int kCandidatesListLength = 50000;

Status SimiliaService::SimiliaSearch(ServerContext* context,
                                     const proto::SimiliaSearchRequest* request,
                                     proto::SimiliaSearchResponse* response) {
  LOG(INFO) << "we are looking for " << request->num_nearest() << " similar images of : "
      << request->image_id();
  if (!request->has_features()) {
    // not implemented for now.
    const std::string error_message = "there are no features in the request.";
    LOG(ERROR) << error_message;
    return Status(StatusCode::INVALID_ARGUMENT, error_message);
  }
  if (request->features().element_size() != kFeatureDimensions) {
    std::stringstream error_message_stream;
    error_message_stream << "Features have " << request->features().element_size() << " dimensions instead of " <<
        kFeatureDimensions << " required.";
    const std::string error_message = error_message_stream.str();
    LOG(ERROR) << error_message;
    return Status(StatusCode::INVALID_ARGUMENT, error_message);
  }

  Timer timer("similia.images.search_processing");
  LOG(INFO) << "there are features in the request so we work directly with these.";
  std::vector<float> features_vec(request->features().element().begin(), request->features().element().end());

  // Prepare features
  Eigen::RowVectorXf features_eig = Eigen::RowVectorXf::Map(&(features_vec[0]), features_vec.size());

  Timer timer_normalization("similia.images.normalization");
  features_eig.normalize();
  LOG(INFO) << "elapsed time for normalization: " << timer_normalization.Stop() << " ms";

  features_eig = features_eig * indexing_rotation_matrix_;
  std::vector<float> features_normalized(features_eig.data(), features_eig.data() + features_eig.size());

  // Find candidates
  Timer timer_candidates_finder("similia.images.candidates_finder");
  std::vector<Candidate> candidates_list = candidates_finder_->GetCandidates(features_normalized,
                                                                             kCandidatesListLength,
                                                                             *context);
  LOG(INFO) << "elapsed time for candidates finder: " << timer_candidates_finder.Stop() << " ms";
  LOG(INFO) << "got a list of " << candidates_list.size() << " candidates. reranking it...";

  // Rerank candidates
  Timer timer_candidates_reranker("similia.images.candidates_reranker");
  std::vector<IdAndComputation> nearest_candidates = candidates_reranker_->RerankCandidates(
      candidates_list, features_normalized, request->num_nearest());
  LOG(INFO) << "elapsed time for candidates reranker: " << timer_candidates_reranker.Stop() << " ms";

  for (const auto& it : nearest_candidates) {
    response->add_image_id(it.id);
    response->add_squared_distance(it.squared_distance);
  }
  int64_t elapsed_time = timer.Stop();
  LOG(INFO) << "total elapsed time: " << elapsed_time;
  response->set_processing_time_ms(elapsed_time);

  return Status::OK;
}

}  // namespace similia

